# https://github.com/eriklindernoren/PyTorch-GAN

from torch import nn
import torch

from cnc_ai.nn import DoubleEmbedding, DownScaleLayer, HiddenLayer

from cnc_ai.TIBERIANDAWN.cnc_structs import (
    all_asset_num_shapes,
    static_tile_names,
    dynamic_object_names,
)


def calculate_asset_num_shapes(names_list):
    names_dict = {v: i for i, v in enumerate(names_list)}
    asset_num_shapes = {
        names_dict[k]: v for k, v in all_asset_num_shapes.items() if k in names_dict
    }
    return asset_num_shapes


class MapEmbedding_62_62(nn.Module):
    def __init__(self, embedding_dim=1024):
        super().__init__()
        self.asset_embedding = DoubleEmbedding(calculate_asset_num_shapes(static_tile_names), 10)
        self.convolutions = nn.Sequential(
            nn.Conv2d(10, 10, 3, padding=2),  # 10x64x64
            nn.LeakyReLU(),
            DownScaleLayer(10, 20, 2),  # 20x32x32
            nn.LeakyReLU(),
            nn.Conv2d(20, 20, 3, padding=1),
            nn.LeakyReLU(),
            DownScaleLayer(20, 40, 2),  # 40x16x16
            nn.LeakyReLU(),
            nn.Conv2d(40, 40, 3, padding=1),
            nn.LeakyReLU(),
            DownScaleLayer(40, 80, 2),  # 80x8x8
            nn.LeakyReLU(),
            nn.Conv2d(80, 80, 3, padding=1),
            nn.LeakyReLU(),
            DownScaleLayer(80, 160, 2),  # 160x4x4
            nn.Flatten(),
            HiddenLayer(160 * 4 * 4, embedding_dim),
            HiddenLayer(embedding_dim, embedding_dim),
        )

    def forward(self, asset_indices, shape_indices):
        map_embedding = self.asset_embedding(asset_indices, shape_indices).permute(0, 3, 1, 2)
        output = self.convolutions(map_embedding)
        return output


class TD_GameEmbedding(nn.Module):
    def __init__(self, embedding_dim=1024):
        super().__init__()
        map_dim = 1024
        self.map_embedding = MapEmbedding_62_62(map_dim)

        self.unit_embedding = DoubleEmbedding(calculate_asset_num_shapes(dynamic_object_names), 16)
        self.owner_embedding = nn.Embedding(256, 3)  # 0-8 and 255 for default value
        self.pip_embedding = nn.Sequential(nn.Embedding(10, 3), nn.Flatten(start_dim=2))
        self.control_embedding = nn.Embedding(256, 3)  # 0-9 and 255 for default value
        self.cloak_embedding = nn.Embedding(5, 2)
        dynamic_object_dim = 16 + 3 + 3 * 18 + 3 + 2 + 5

        self.buildable_embedding = nn.Embedding(len(dynamic_object_names), 7)
        siderbar_entries_dim = 7 + 6

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

        self.siderbar_entries_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=siderbar_entries_dim,
                nhead=1,
                batch_first=True,
                layer_norm_eps=0,
                dim_feedforward=16,
            ),
            num_layers=2,
        )
        self.dense = nn.Sequential(
            HiddenLayer(map_dim + dynamic_object_dim + siderbar_entries_dim + 6, embedding_dim),
            HiddenLayer(embedding_dim, embedding_dim),
        )

    def forward(
        self,
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
        siderbar_entries = torch.cat(
            [self.buildable_embedding(SidebarAssetName), SidebarContinuous],
            axis=2,
        )
        internal_state = torch.cat(
            [
                map_embedding,
                torch.sum(self.dynamic_object_encoder(dynamic_objects), 1),
                SidebarInfos,
                torch.sum(self.siderbar_entries_encoder(siderbar_entries), 1),
            ],
            1,
        )
        game_state_embedding = self.dense(internal_state)
        return game_state_embedding


def pad_game_states(list_of_game_states, device=None):
    dynamic_lengths = torch.tensor(
        [len(game_state['AssetName']) for game_state in list_of_game_states]
    ).to(device)
    sidebar_lengths = torch.tensor(
        [len(game_state['SidebarAssetName']) for game_state in list_of_game_states]
    ).to(device)
    tensors = {
        **{
            key: torch.stack(
                [torch.tensor(game_state[key]) for game_state in list_of_game_states], 0
            ).to(device)
            for key in ['StaticAssetName', 'StaticShapeIndex', 'SidebarInfos']
        },
        **{
            key: torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(game_state[key]) for game_state in list_of_game_states],
                batch_first=True,
            ).to(device)
            for key in [
                'AssetName',
                'ShapeIndex',
                'Owner',
                'Pips',
                'ControlGroup',
                'Cloak',
                'Continuous',
                'SidebarAssetName',
                'SidebarContinuous',
            ]
        },
    }
    return dynamic_lengths, sidebar_lengths, tensors
