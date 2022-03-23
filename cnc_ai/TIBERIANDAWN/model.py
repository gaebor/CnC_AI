# https://github.com/eriklindernoren/PyTorch-GAN
from operator import mul
from functools import reduce

from torch import nn
import torch

from cnc_ai.nn import (
    DoubleEmbedding,
    DownScaleLayer,
    UpscaleLayer,
    HiddenLayer,
    SoftmaxReadout,
    ConvolutionLayer,
)

from cnc_ai.TIBERIANDAWN.cnc_structs import (
    all_asset_num_shapes,
    static_tile_names,
    dynamic_object_names,
    MAX_OBJECT_PIPS,
)


def calculate_asset_num_shapes(names_list):
    names_dict = {v: i for i, v in enumerate(names_list)}
    asset_num_shapes = {
        names_dict[k]: v for k, v in all_asset_num_shapes.items() if k in names_dict
    }
    return asset_num_shapes


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
            HiddenLayer(160 * 4 * 4, embedding_dim),
            HiddenLayer(embedding_dim, embedding_dim),
        )

    def forward(self, asset_indices, shape_indices):
        map_embedding = self.asset_embedding(asset_indices, shape_indices).permute(0, 3, 1, 2)
        output = self.convolutions(map_embedding)
        return output


class MapGenerator_62_62(nn.Sequential):
    def __init__(self, embedding_dim=1024, out_channels=5 * 12):
        super().__init__(
            HiddenLayer(embedding_dim, embedding_dim),
            HiddenLayer(embedding_dim, 16 * 4 * 4 * out_channels),
            nn.Unflatten(-1, (16 * out_channels, 4, 4)),  # 16x4x4
            UpscaleLayer(16 * out_channels, 8 * out_channels, 3, 2),  # 8x8x8
            nn.LeakyReLU(),
            ConvolutionLayer(8 * out_channels),
            UpscaleLayer(8 * out_channels, 4 * out_channels, 3, 2),  # 4x16x16
            nn.LeakyReLU(),
            ConvolutionLayer(4 * out_channels),
            UpscaleLayer(4 * out_channels, 2 * out_channels, 3, 2),  # 2x32x32
            nn.LeakyReLU(),
            ConvolutionLayer(2 * out_channels),
            UpscaleLayer(2 * out_channels, out_channels, 3, 2),  # 1x64x64
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=0),  # 1x62x62
        )


class SidebarEntriesEncoder(nn.Module):
    def __init__(self, num_layers=2, embedding_dim=7):
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
        return sidebar_embeddings


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
            num_layers=2,
        )

        self.siderbar_entries_encoder = SidebarEntriesEncoder(num_layers=2)
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
            axis=2,
        )
        internal_vectors = [
            map_embedding,
            self.dynamic_object_encoder(dynamic_objects, src_key_padding_mask=dynamic_mask)[:, 0],
            SidebarInfos,
        ]
        sidebar_embeddings = self.siderbar_entries_encoder(
            sidebar_mask, SidebarAssetName, SidebarContinuous
        )
        if SidebarAssetName.shape[1] > 0:
            sidebar_embedding = sidebar_embeddings[:, 0, :]
            sidebar_embedding[sidebar_mask[:, 0], :] = 0
            internal_vectors.append(sidebar_embedding)
        else:
            internal_vectors.append(
                torch.zeros(
                    (dynamic_objects.shape[0], self.siderbar_entries_encoder.embedding_dim),
                    dtype=dynamic_objects.dtype,
                    device=dynamic_objects.device,
                )
            )
        internal_state = torch.cat(internal_vectors, 1)
        game_state_embedding = self.dense(internal_state)
        return game_state_embedding


class SidebarDecoder(nn.Module):
    def __init__(self, embedding_dim=1024, num_layers=2, out_dim=12):
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


class TD_Action(nn.Module):
    def __init__(self, embedding_dim=1024):
        super().__init__()
        self.per_tile_actions = (5, 12)
        self.mouse_action_size = (62, 62) + self.per_tile_actions
        self.mouse_action_dim = reduce(mul, self.mouse_action_size, 1)
        self.mouse_position = MapGenerator_62_62(embedding_dim)
        self.sidebar_decoder = SidebarDecoder(
            embedding_dim=embedding_dim, num_layers=2, out_dim=12
        )
        self.flatten = nn.Flatten()
        self.mouse_unflatten = nn.Unflatten(-1, self.mouse_action_size)
        self.sidebar_unflatten = nn.Unflatten(-1, (-1, self.sidebar_decoder.embedding_dim))

    def forward(self, game_state, sidebar_mask, SidebarAssetName, SidebarContinuous):
        mouse = self.mouse_position(game_state).permute(0, 2, 3, 1)
        sidebar = self.sidebar_decoder(
            game_state, sidebar_mask, SidebarAssetName, SidebarContinuous
        )
        sidebar[sidebar_mask] = torch.tensor(
            float('-inf'), dtype=game_state.dtype, device=game_state.device
        )
        actions = nn.functional.softmax(
            torch.cat([self.flatten(mouse), self.flatten(sidebar)], 1), -1
        )
        mouse_actions = self.mouse_unflatten(actions[:, : self.mouse_action_dim])
        sidebar_actions = self.sidebar_unflatten(actions[:, self.mouse_action_dim :])
        return mouse_actions, sidebar_actions


class TD_GamePlay(nn.Module):
    def __init__(self, embedding_dim=1024):
        super().__init__()
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
        )
        actions = self.actions(latent_embedding, sidebar_mask, SidebarAssetName, SidebarContinuous)
        return actions
