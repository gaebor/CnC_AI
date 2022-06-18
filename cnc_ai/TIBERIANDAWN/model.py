# https://github.com/eriklindernoren/PyTorch-GAN

from torch import nn
import torch
from torch.distributions.beta import Beta

from cnc_ai.nn import (
    DoubleEmbedding,
    DownScaleLayer,
    HiddenLayer,
    ConvolutionLayer,
    log_beta,
    interflatten,
    take_along_first_dim,
)

from cnc_ai.TIBERIANDAWN.cnc_structs import (
    all_asset_num_shapes,
    static_tile_names,
    dynamic_object_names,
    MAX_OBJECT_PIPS,
)

from cnc_ai.common import multi_sample


class TD_GamePlay(nn.Module):
    def __init__(self, embedding_dim=1024, n_lstm=1):
        super().__init__()
        self.reset()
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, n_lstm)
        self.game_state = TD_GameEmbedding(embedding_dim)
        self.actions = TD_Action(embedding_dim)

    def forward(
        self,
        dynamic_mask,
        sidebar_mask,
        StaticAssetName,
        StaticShapeIndex,
        AssetName,
        ShapeIndex,
        Owner,
        Pips,
        ControlGroup,
        Cloak,
        Continuous,
        SidebarInfos,
        SidebarAssetName,
        SidebarContinuous,
    ):
        latent_embedding = interflatten(
            self.game_state,
            dynamic_mask,
            sidebar_mask,
            StaticAssetName,
            StaticShapeIndex,
            AssetName,
            ShapeIndex,
            Owner,
            Pips,
            ControlGroup,
            Cloak,
            Continuous,
            SidebarInfos,
            SidebarAssetName,
            SidebarContinuous,
        )
        time_progress, cell_states = self.lstm(latent_embedding, self.cell_states)
        self.cell_states = cell_states[0].detach(), cell_states[1].detach()
        actions = interflatten(
            self.actions, time_progress, sidebar_mask, SidebarAssetName, SidebarContinuous
        )
        return actions

    def reset(self):
        self.cell_states = None


class TD_GameEmbedding(nn.Module):
    def __init__(self, embedding_dim=1024):
        super().__init__()
        self.map_embedding = MapEmbedding_62_62(embedding_dim)

        self.unit_embedding = DoubleEmbedding(calculate_asset_num_shapes(dynamic_object_names), 16)
        self.owner_embedding = nn.Embedding(256, 3)  # 0-8 and 255 for default value
        self.pip_embedding = nn.Sequential(nn.Embedding(10, 3), nn.Flatten(-2))
        self.control_embedding = nn.Embedding(256, 3)  # 0-9 and 255 for default value
        self.cloak_embedding = nn.Embedding(5, 2)

        self.buildable_embedding = nn.Embedding(len(dynamic_object_names), 7)

        dynamic_object_dim = (
            sum(
                layer.embedding_dim
                for layer in [
                    self.unit_embedding,
                    self.owner_embedding,
                    self.control_embedding,
                    self.cloak_embedding,
                ]
            )
            + self.pip_embedding[0].embedding_dim * MAX_OBJECT_PIPS
            + 5  # dynamic object continuous
        )

        self.dynamic_object_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dynamic_object_dim,
                nhead=1,
                batch_first=True,
                layer_norm_eps=0,
                dim_feedforward=128,
            ),
            num_layers=1,
        )

        self.siderbar_entries_encoder = SidebarEntriesEncoder(num_layers=1)
        self.dense = nn.Sequential(
            HiddenLayer(
                self.map_embedding.embedding_dim
                + dynamic_object_dim
                + self.siderbar_entries_encoder.embedding_dim
                + 6,  # sidebar members
                embedding_dim,
            ),
            HiddenLayer(embedding_dim, embedding_dim),
        )

    def forward(
        self,
        dynamic_mask,
        sidebar_mask,
        StaticAssetName,
        StaticShapeIndex,
        AssetName,
        ShapeIndex,
        Owner,
        Pips,
        ControlGroup,
        Cloak,
        Continuous,
        SidebarInfos,
        SidebarAssetName,
        SidebarContinuous,
    ):
        map_embedding = self.map_embedding(StaticAssetName, StaticShapeIndex)
        dynamic_objects = torch.cat(
            [
                self.unit_embedding(AssetName, ShapeIndex),
                self.owner_embedding(Owner),
                self.pip_embedding(Pips),
                self.control_embedding(ControlGroup),
                self.cloak_embedding(Cloak),
                Continuous,
            ],
            axis=-1,
        )
        internal_vectors = [
            map_embedding,
            self.dynamic_object_encoder(dynamic_objects, src_key_padding_mask=dynamic_mask).sum(
                axis=1
            ),
            SidebarInfos,
            self.siderbar_entries_encoder(sidebar_mask, SidebarAssetName, SidebarContinuous).sum(
                axis=1
            ),
        ]
        internal_state = torch.cat(internal_vectors, 1)
        game_state_embedding = self.dense(internal_state)
        return game_state_embedding


class MapEmbedding_62_62(nn.Module):
    def __init__(self, embedding_dim=1024, static_embedding_dim=10):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.asset_embedding = DoubleEmbedding(
            calculate_asset_num_shapes(static_tile_names), static_embedding_dim
        )
        self.convolutions = nn.Sequential(
            nn.Conv2d(static_embedding_dim, static_embedding_dim, 3, padding=2),  # 1x64x64
            nn.LeakyReLU(),
            DownScaleLayer(static_embedding_dim, 2 * static_embedding_dim, 2),  # 2x32x32
            nn.LeakyReLU(),
            ConvolutionLayer(2 * static_embedding_dim),
            DownScaleLayer(2 * static_embedding_dim, 4 * static_embedding_dim, 2),  # 4x16x16
            nn.LeakyReLU(),
            ConvolutionLayer(4 * static_embedding_dim),
            DownScaleLayer(4 * static_embedding_dim, 8 * static_embedding_dim, 2),  # 8x8x8
            nn.LeakyReLU(),
            ConvolutionLayer(8 * static_embedding_dim),
            DownScaleLayer(8 * static_embedding_dim, 16 * static_embedding_dim, 2),  # 16x4x4
            nn.Flatten(),
            HiddenLayer(16 * static_embedding_dim * 4 * 4, embedding_dim),
            HiddenLayer(embedding_dim, embedding_dim),
        )

    def forward(self, asset_indices, shape_indices):
        map_embedding = self.asset_embedding(asset_indices, shape_indices).permute(0, 3, 1, 2)
        output = self.convolutions(map_embedding)
        return output


def calculate_asset_num_shapes(names_list):
    names_dict = {v: i for i, v in enumerate(names_list)}
    asset_num_shapes = {
        names_dict[k]: v for k, v in all_asset_num_shapes.items() if k in names_dict
    }
    return asset_num_shapes


class SidebarEntriesEncoder(nn.Module):
    def __init__(self, num_layers=1, embedding_dim=7):
        super().__init__()
        self.buildable_embedding = nn.Embedding(len(dynamic_object_names), embedding_dim)
        self.embedding_dim = self.buildable_embedding.embedding_dim + 6  # sidebar continuous
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=1,
                batch_first=True,
                layer_norm_eps=0,
                dim_feedforward=16,
            ),
            num_layers=num_layers,
        )

    def forward(self, sidebar_mask, SidebarAssetName, SidebarContinuous):
        siderbar_entries = torch.cat(
            [self.buildable_embedding(SidebarAssetName), SidebarContinuous],
            axis=2,
        )
        sidebar_embeddings = self.encoder(siderbar_entries, src_key_padding_mask=sidebar_mask)
        sidebar_embeddings[sidebar_mask] = 0
        return sidebar_embeddings


class TD_Action(nn.Module):
    def __init__(self, embedding_dim=1024):
        super().__init__()
        self.per_tile_actions = (5, 12)
        self.mouse_action = MouseAction(embedding_dim)
        self.sidebar_decoder = SidebarDecoder(
            embedding_dim=embedding_dim, num_layers=1, out_dim=12
        )
        self.flatten = nn.Flatten()

    def forward(self, game_state, sidebar_mask, SidebarAssetName, SidebarContinuous):
        mouse_positional_params, mouse_button_logits = self.mouse_action(game_state)
        sidebar = self.sidebar_decoder(
            game_state, sidebar_mask, SidebarAssetName, SidebarContinuous
        )
        sidebar[sidebar_mask] = float('-inf')
        action_distribution = nn.functional.log_softmax(
            torch.cat([mouse_button_logits, self.flatten(sidebar)], 1), -1
        )

        return mouse_positional_params, action_distribution

    def sample(self, mouse_positional_params, action_distribution):
        chosen_actions = multi_sample(torch.exp(action_distribution))
        chosen_mouse_positional_params = self.mouse_action.choose_beta_parameters(
            mouse_positional_params, chosen_actions
        )
        alpha_x, beta_x, alpha_y, beta_y = (
            chosen_mouse_positional_params[:, 0],
            chosen_mouse_positional_params[:, 1],
            chosen_mouse_positional_params[:, 2],
            chosen_mouse_positional_params[:, 3],
        )
        mouse_x, mouse_y = Beta(alpha_x, beta_x).sample(), Beta(alpha_y, beta_y).sample()

        return chosen_actions, mouse_x * 1488, mouse_y * 1488

    def evaluate(
        self, mouse_positional_params, action_distribution, chosen_actions, mouse_x, mouse_y
    ):
        chosen_mouse_positional_params = self.mouse_action.choose_beta_parameters(
            mouse_positional_params, chosen_actions
        )
        button_prob = take_along_first_dim(action_distribution, chosen_actions)

        mouse_x_prob = log_beta(
            chosen_mouse_positional_params[:, 0],
            chosen_mouse_positional_params[:, 1],
            mouse_x / 1488,
        )
        mouse_y_prob = log_beta(
            chosen_mouse_positional_params[:, 2],
            chosen_mouse_positional_params[:, 3],
            mouse_y / 1488,
        )
        return button_prob + mouse_x_prob + mouse_y_prob


class MouseAction(nn.Module):
    def __init__(self, embedding_dim=1024, n_layers=2):
        super().__init__()
        self.n_buttons = 12  # types of mouse action, including None
        self.ff = nn.Sequential(
            *[HiddenLayer(embedding_dim) for _ in range(n_layers)],
            HiddenLayer(embedding_dim, 4 * (self.n_buttons - 1) + self.n_buttons),
        )

    def forward(self, latent_embedding):
        logits = self.ff(latent_embedding)
        mouse_parameters = torch.cat(
            [
                torch.ones(logits.shape[0], 1, 4, dtype=logits.dtype, device=logits.device),
                torch.exp(
                    logits[:, : 4 * (self.n_buttons - 1)].reshape(-1, self.n_buttons - 1, 4)
                ),
            ],
            axis=1,
        )
        mouse_buttons = logits[:, -self.n_buttons :]
        return mouse_parameters, mouse_buttons

    def choose_beta_parameters(self, mouse_positional_params, chosen_actions):
        return take_along_first_dim(
            mouse_positional_params,
            torch.where(chosen_actions >= self.n_buttons, 0, chosen_actions),
        )


class SidebarDecoder(nn.Module):
    def __init__(self, embedding_dim=1024, num_layers=1, out_dim=12):
        super().__init__()
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embedding_dim,
                nhead=1,
                batch_first=True,
                layer_norm_eps=0,
                dim_feedforward=embedding_dim,
            ),
            num_layers=num_layers,
        )
        self.sidebar_embedding = SidebarEntriesEncoder(num_layers=0)
        self.linear_in = HiddenLayer(self.sidebar_embedding.embedding_dim, embedding_dim)
        self.embedding_dim = out_dim
        self.linear_out = HiddenLayer(embedding_dim, self.embedding_dim)

    def forward(self, game_state, sidebar_mask, SidebarAssetName, SidebarContinuous):
        sidebar_in = self.linear_in(
            self.sidebar_embedding(sidebar_mask, SidebarAssetName, SidebarContinuous)
        )
        decoded = self.linear_out(
            self.decoder(sidebar_in, game_state[:, None, :], tgt_key_padding_mask=sidebar_mask)
        )
        return decoded
