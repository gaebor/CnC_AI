from torch import nn
import torch

from cnc_ai.nn import (
    DoubleEmbedding,
    DownScaleLayer,
    HiddenLayer,
    ConvolutionLayer,
    MultiChoiceSamplerWithLogits,
    TwoParameterContinuousSampler,
)

from cnc_ai.TIBERIANDAWN.cnc_structs import (
    all_asset_num_shapes,
    static_tile_names,
    dynamic_object_names,
    MAX_OBJECT_PIPS,
)


class TD_GamePlay(nn.Module):
    def __init__(self, embedding_dim=128, n_lstm=1, dropout=0.1):
        super().__init__()
        self.reset()
        self.game_state = TD_GameEmbedding(embedding_dim, dropout=dropout)
        # self.lstm = nn.LSTM(
        #     embedding_dim,
        #     embedding_dim,
        #     n_lstm,
        #     batch_first=False,
        #     bidirectional=False,
        #     dropout=dropout,
        # )
        self.actions = TD_Action(embedding_dim, dropout=dropout)

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
        button,
        mouse_x,
        mouse_y,
    ):
        latent_embedding = self.game_state(
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
            button,
            mouse_x,
            mouse_y,
        )
        # time_progress, cell_states = self.lstm(latent_embedding, self.cell_states)
        # self.cell_states = cell_states[0].detach(), cell_states[1].detach()
        actions = self.actions(latent_embedding, sidebar_mask, SidebarAssetName, SidebarContinuous)
        return actions

    def reset(self):
        self.cell_states = None


class TD_GameEmbedding(nn.Module):
    def __init__(self, embedding_dim=1024, dropout=0.1):
        super().__init__()
        # self.map_embedding = MapEmbedding_62_62(embedding_dim, dropout=0.1)

        self.dynamic_object_embedding = DynamicObjectEmbedding()
        # self.dynamic_object_transformer = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         d_model=self.dynamic_object_embedding.embedding_dim,
        #         nhead=1,
        #         batch_first=True,
        #         layer_norm_eps=0,
        #         dim_feedforward=128,
        #         dropout=dropout,
        #     ),
        #     num_layers=1,
        # )

        # self.siderbar_embedding = SidebarEmbedding()
        # self.sidebar_transformer = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         d_model=self.siderbar_embedding.embedding_dim,
        #         nhead=1,
        #         batch_first=True,
        #         layer_norm_eps=0,
        #         dim_feedforward=16,
        #         dropout=dropout,
        #     ),
        #     num_layers=1,
        # )

        # self.previous_action_embedding = nn.Embedding(len(dynamic_object_names) * 12, 16)

        self.dense = nn.Sequential(
            HiddenLayer(
                # self.map_embedding.embedding_dim
                self.dynamic_object_embedding.embedding_dim,
                # + self.siderbar_embedding.embedding_dim
                # + 6  # sidebar members
                # + self.previous_action_embedding.embedding_dim
                # + 2,  # previous mouse position
                embedding_dim,
                dropout=dropout,
            ),
            HiddenLayer(embedding_dim, embedding_dim, dropout=dropout),
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
        button,
        mouse_x,
        mouse_y,
    ):
        # map_embedding = self.map_embedding(StaticAssetName, StaticShapeIndex)

        dynamic_objects = self.dynamic_object_embedding(
            AssetName, ShapeIndex, Owner, Pips, ControlGroup, Cloak, Continuous
        )[:, :, 0, :]
        # sidebar = self.sidebar_transformer(
        #     self.siderbar_embedding(SidebarAssetName, SidebarContinuous),
        #     src_key_padding_mask=sidebar_mask,
        # ).sum(dim=1)

        # previous_actions = self.previous_action_embedding(
        #     previous_action_item * 12 + previous_action_type
        # )

        # internal_state = torch.cat(
        #     [
        #         map_embedding,
        #         dynamic_objects,
        #         SidebarInfos,
        #         sidebar,
        #         previous_actions,
        #         previous_mouse_x[:, None],
        #         previous_mouse_y[:, None],
        #     ],
        #     dim=1,
        # )
        game_state_embedding = self.dense(dynamic_objects)
        return game_state_embedding


class MapEmbedding_62_62(nn.Module):
    def __init__(self, embedding_dim=1024, static_embedding_dim=10, dropout=0.1):
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
            ConvolutionLayer(2 * static_embedding_dim, dropout=dropout),
            DownScaleLayer(2 * static_embedding_dim, 4 * static_embedding_dim, 2),  # 4x16x16
            nn.LeakyReLU(),
            ConvolutionLayer(4 * static_embedding_dim, dropout=dropout),
            DownScaleLayer(4 * static_embedding_dim, 8 * static_embedding_dim, 2),  # 8x8x8
            nn.LeakyReLU(),
            ConvolutionLayer(8 * static_embedding_dim, dropout=dropout),
            DownScaleLayer(8 * static_embedding_dim, 16 * static_embedding_dim, 2),  # 16x4x4
            nn.Flatten(),
            HiddenLayer(16 * static_embedding_dim * 4 * 4, embedding_dim, dropout=dropout),
            HiddenLayer(embedding_dim, embedding_dim, dropout=dropout),
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


class DynamicObjectEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.unit_embedding = DoubleEmbedding(calculate_asset_num_shapes(dynamic_object_names), 16)
        self.owner_embedding = nn.Embedding(256, 3)  # 0-8 and 255 for default value
        self.pip_embedding = nn.Sequential(nn.Embedding(10, 3), nn.Flatten(-2))
        self.control_embedding = nn.Embedding(256, 3)  # 0-9 and 255 for default value
        self.cloak_embedding = nn.Embedding(5, 2)

        self.embedding_dim = (
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

    def forward(self, AssetName, ShapeIndex, Owner, Pips, ControlGroup, Cloak, Continuous):
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
        return dynamic_objects


class SidebarEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.buildable_embedding = nn.Embedding(len(dynamic_object_names), 7)
        self.embedding_dim = self.buildable_embedding.embedding_dim + 6  # sidebar continuous

    def forward(self, SidebarAssetName, SidebarContinuous):
        sidebar_embeddings = torch.cat(
            [self.buildable_embedding(SidebarAssetName), SidebarContinuous], axis=-1
        )
        return sidebar_embeddings


class TD_Action(nn.Module):
    def __init__(self, embedding_dim=1024, n_actions=12, dropout=0.1):
        super().__init__()
        self.n_actions = n_actions
        self.mouse_parameters = MouseParameters(embedding_dim, dropout=dropout)
        self.mouse_x = TwoParameterContinuousSampler(torch.distributions.beta.Beta)
        self.mouse_y = TwoParameterContinuousSampler(torch.distributions.beta.Beta)
        self.sidebar_embedding = SidebarEmbedding()
        self.dense = nn.Sequential(
            HiddenLayer(
                self.sidebar_embedding.embedding_dim, n_actions * embedding_dim, dropout=dropout
            ),
            nn.Unflatten(-1, (n_actions, embedding_dim)),
        )
        # self.transformer_in = HiddenLayer(
        #     self.sidebar_embedding.embedding_dim, embedding_dim, dropout=dropout
        # )
        # self.action_transformer = nn.TransformerDecoder(
        #     nn.TransformerDecoderLayer(
        #         d_model=embedding_dim,
        #         nhead=1,
        #         batch_first=True,
        #         layer_norm_eps=0,
        #         dim_feedforward=embedding_dim,
        #         dropout=dropout,
        #     ),
        #     num_layers=1,
        # )
        # self.transformer_out = nn.Linear(embedding_dim, n_actions)

        self.button_sampler = MultiChoiceSamplerWithLogits()

    def forward(self, game_state, sidebar_mask, SidebarAssetName, SidebarContinuous):
        mouse_positional_params = self.mouse_parameters(game_state)
        sidebar = self.dense(self.sidebar_embedding(SidebarAssetName, SidebarContinuous))
        # actions_input = self.transformer_in(sidebar)
        # transformed = self.action_transformer(
        #     actions_input, game_state[:, None, :], tgt_key_padding_mask=sidebar_mask
        # )
        # action_logits = self.transformer_out(transformed)
        action_logits = (sidebar * game_state[:, :, None, None, :]).sum(dim=-1)
        action_logits[sidebar_mask] = float('-inf')

        return mouse_positional_params, action_logits

    def sample(self, mouse_parameters, action_logits):
        chosen_actions_mask = self.button_sampler.sample(action_logits.flatten(-2)).unflatten(
            -1, (-1, self.n_actions)
        )
        chosen_mouse_parameters = self.mouse_parameters.choose_parameters(
            mouse_parameters, chosen_actions_mask
        )
        mouse_x, mouse_y = (
            self.mouse_x.sample(
                [chosen_mouse_parameters[:, :, 0], chosen_mouse_parameters[:, :, 1]]
            ),
            self.mouse_y.sample(
                [chosen_mouse_parameters[:, :, 2], chosen_mouse_parameters[:, :, 3]]
            ),
        )
        return chosen_actions_mask, mouse_x * 1488, mouse_y * 1488

    def surprise(self, mouse_parameters, action_logits, chosen_actions_mask, mouse_x, mouse_y):
        chosen_mouse_parameters = self.mouse_parameters.choose_parameters(
            mouse_parameters, chosen_actions_mask
        )
        button_surprise = self.button_sampler.surprise(
            action_logits.flatten(-2), chosen_actions_mask.flatten(-2)
        )
        mouse_x_surprise = self.mouse_x.surprise(
            [chosen_mouse_parameters[:, :, 0], chosen_mouse_parameters[:, :, 1]], mouse_x / 1488
        )
        mouse_y_surprise = self.mouse_y.surprise(
            [chosen_mouse_parameters[:, :, 2], chosen_mouse_parameters[:, :, 3]], mouse_y / 1488
        )
        return button_surprise + mouse_x_surprise + mouse_y_surprise

    def get_probabilities(self, mouse_parameters, action_logits, x):
        button_probs = torch.softmax(action_logits.flatten(-2), -1).unflatten(
            -1, (-1, self.n_actions)
        )
        mouse_x_probs = torch.exp(
            -self.mouse_x.surprise([mouse_parameters[:, 0], mouse_parameters[:, 1]], x)
        )
        mouse_y_probs = torch.exp(
            -self.mouse_y.surprise([mouse_parameters[:, 2], mouse_parameters[:, 3]], x)
        )
        return button_probs, mouse_x_probs, mouse_y_probs


class MouseParameters(nn.Module):
    def __init__(self, embedding_dim=1024, n_layers=1, dropout=0.1):
        super().__init__()
        self.ff = nn.Sequential(
            *[HiddenLayer(embedding_dim, dropout=dropout) for _ in range(n_layers)],
            nn.Linear(embedding_dim, 4),
        )

    def forward(self, latent_embedding):
        raw_params = self.ff(latent_embedding)
        u1, v1, u2, v2 = (
            torch.sigmoid(raw_params[:, :, 0]),
            torch.sigmoid(raw_params[:, :, 1]) * 998 + 2,
            torch.sigmoid(raw_params[:, :, 2]),
            torch.sigmoid(raw_params[:, :, 3]) * 998 + 2,
        )
        alpha_beta_params = torch.stack([u1 * v1, (1 - u1) * v1, u2 * v2, (1 - u2) * v2], dim=-1)
        return alpha_beta_params

    @staticmethod
    def choose_parameters(mouse_parameters, chosen_actions_mask):
        # INPUT_REQUEST_MOUSE_MOVE
        # INPUT_REQUEST_MOUSE_AREA
        # INPUT_REQUEST_MOUSE_AREA_ADDITIVE
        move_mouse = chosen_actions_mask[:, :, 0, [1, 5, 6]].any(dim=-1)
        chosen_mouse_positional_params = mouse_parameters.where(
            move_mouse[:, :, None], torch.ones_like(mouse_parameters)
        )
        return chosen_mouse_positional_params
