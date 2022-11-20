from torch import nn
import torch

from cnc_ai.nn import (
    DoubleEmbedding,
    DownScaleLayer,
    HiddenLayer,
    ConvolutionLayer,
    MultiChoiceSamplerWithLogits,
    TwoParameterContinuousSampler,
    interflatten,
)

from cnc_ai.TIBERIANDAWN.cnc_structs import (
    all_asset_num_shapes,
    static_tile_names,
    dynamic_object_names,
    MAX_OBJECT_PIPS,
)


class TD_GamePlay(nn.Module):
    def __init__(self, embedding_dim=1024, n_lstm=1, dropout=0.1):
        super().__init__()
        self.reset()
        self.game_state = TD_GameEmbedding(embedding_dim, dropout=dropout)
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
        previous_action_item,
        previous_action_type,
        previous_mouse_x,
        previous_mouse_y,
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
            previous_action_item,
            previous_action_type,
            previous_mouse_x,
            previous_mouse_y,
        )
        actions = interflatten(
            self.actions, latent_embedding, sidebar_mask, SidebarAssetName, SidebarContinuous
        )
        return actions

    def reset(self):
        pass


class TD_GameEmbedding(nn.Module):
    def __init__(self, embedding_dim=1024, dropout=0.1):
        super().__init__()
        self.dynamic_object_embedding = DynamicObjectEmbedding()
        self.dense = nn.Sequential(
            HiddenLayer(
                self.dynamic_object_embedding.embedding_dim,
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
        previous_action_item,
        previous_action_type,
        previous_mouse_x,
        previous_mouse_y,
    ):
        dynamic_objects = self.dynamic_object_embedding(
            AssetName, ShapeIndex, Owner, Pips, ControlGroup, Cloak, Continuous
        )[:, 0, :]

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
        dynamic_object = torch.cat(
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
        return dynamic_object


class SidebarEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.buildable_embedding = nn.Embedding(len(dynamic_object_names), 7)
        self.embedding_dim = self.buildable_embedding.embedding_dim + 6  # sidebar continuous

    def forward(self, SidebarAssetName, SidebarContinuous):
        sidebar_embeddings = torch.cat(
            [self.buildable_embedding(SidebarAssetName), SidebarContinuous],
            axis=2,
        )
        return sidebar_embeddings


class TD_Action(nn.Module):
    def __init__(self, embedding_dim=1024, n_actions=12, dropout=0.1):
        super().__init__()
        self.mouse_parameters = MouseParameters(embedding_dim, dropout=dropout)
        self.mouse_x = TwoParameterContinuousSampler(torch.distributions.beta.Beta)
        self.mouse_y = TwoParameterContinuousSampler(torch.distributions.beta.Beta)

        self.sidebar_embedding = SidebarEmbedding()
        self.transformer_in = HiddenLayer(
            self.sidebar_embedding.embedding_dim, embedding_dim, dropout=dropout
        )
        self.transformer_out = nn.Linear(embedding_dim, n_actions)

        self.button_sampler = MultiChoiceSamplerWithLogits()

    def forward(self, game_state, sidebar_mask, SidebarAssetName, SidebarContinuous):
        mouse_positional_params = self.mouse_parameters(game_state)
        sidebar = self.sidebar_embedding(SidebarAssetName, SidebarContinuous)
        actions_input = self.transformer_in(sidebar[:, [0], :])
        action_logits = self.transformer_out(actions_input)
        # action_logits[sidebar_mask] = float('-inf')
        return mouse_positional_params, action_logits

    def sample(self, mouse_parameters, action_logits):
        chosen_actions_mask = self.button_sampler.sample(
            action_logits.reshape(action_logits.shape[0], -1)
        ).reshape(action_logits.shape[0], -1, action_logits.shape[2])
        chosen_mouse_parameters = self.mouse_parameters.choose_parameters(
            mouse_parameters, chosen_actions_mask
        )
        mouse_x, mouse_y = (
            self.mouse_x.sample(chosen_mouse_parameters[:, :2]),
            self.mouse_y.sample(chosen_mouse_parameters[:, 2:]),
        )
        chosen_actions = chosen_actions_mask.nonzero()
        chosen_item, action_type = chosen_actions[:, 1], chosen_actions[:, 2]
        return chosen_item, action_type, mouse_x * 1488, mouse_y * 1488

    def surprise(
        self, mouse_parameters, action_logits, chosen_item, action_type, mouse_x, mouse_y
    ):
        action_mask = torch.zeros(action_logits.shape, dtype=torch.bool, device=chosen_item.device)
        action_mask[torch.arange(action_logits.shape[0]), chosen_item, action_type] = True
        chosen_mouse_parameters = self.mouse_parameters.choose_parameters(
            mouse_parameters, action_mask
        )
        button_surprise = self.button_sampler.surprise(
            action_logits.reshape(action_logits.shape[0], -1),
            action_mask.reshape(action_logits.shape[0], -1),
        )
        mouse_x_surprise = self.mouse_x.surprise(chosen_mouse_parameters[:, :2], mouse_x / 1488)
        mouse_y_surprise = self.mouse_y.surprise(chosen_mouse_parameters[:, 2:], mouse_y / 1488)
        return button_surprise + mouse_x_surprise + mouse_y_surprise


class MouseParameters(nn.Module):
    def __init__(self, embedding_dim=1024, n_layers=1, dropout=0.1):
        super().__init__()
        self.ff = nn.Sequential(
            *[HiddenLayer(embedding_dim, dropout=dropout) for _ in range(n_layers)],
            nn.Linear(embedding_dim, 4),
            nn.ELU()
        )

    def forward(self, latent_embedding):
        alpha_beta_params = self.ff(latent_embedding) + 2
        mouse_parameters = torch.stack(
            [
                torch.ones(
                    alpha_beta_params.shape[0],
                    4,
                    dtype=alpha_beta_params.dtype,
                    device=alpha_beta_params.device,
                ),
                alpha_beta_params,
            ],
            dim=1,
        )
        return mouse_parameters

    @staticmethod
    def choose_parameters(mouse_parameters, chosen_actions_mask):
        # INPUT_REQUEST_MOUSE_MOVE
        # INPUT_REQUEST_MOUSE_AREA
        # INPUT_REQUEST_MOUSE_AREA_ADDITIVE
        move_mouse = chosen_actions_mask[:, 0, [1, 5, 6]].any(dim=1)
        parameter_index = torch.stack([~move_mouse, move_mouse], dim=1)
        chosen_mouse_positional_params = mouse_parameters[parameter_index]
        return chosen_mouse_positional_params
