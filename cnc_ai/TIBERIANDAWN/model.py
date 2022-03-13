# https://github.com/eriklindernoren/PyTorch-GAN

import numpy
from torch import nn
import torch

from cnc_ai.nn import DoubleEmbedding, DownScaleLayer, HiddenLayer, SoftmaxReadout

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
    def __init__(self, embedding_dim=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
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


class SidebarEntriesEncoder(nn.Module):
    def __init__(self, num_layers=2):
        super().__init__()
        self.buildable_embedding = nn.Embedding(len(dynamic_object_names), 7)
        SidebarEntriesEncoder.embedding_dim = (
            self.buildable_embedding.embedding_dim + 6
        )  # sidebar continuous
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
        self.map_embedding = MapEmbedding_62_62(1024)

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


class SidebarReadout(nn.Module):
    def __init__(self, embedding_dim=1024, n_readout=10):
        super().__init__()
        self.n_readout = n_readout
        self.attention = nn.MultiheadAttention(
            SidebarEntriesEncoder.embedding_dim, 1, batch_first=True
        )

        self.entries_embedding = SidebarEntriesEncoder(num_layers=0)

        self.memory = nn.Sequential(
            HiddenLayer(embedding_dim, self.entries_embedding.embedding_dim),
            nn.Unflatten(-1, (1, -1)),  # insert a one-long dimension before the last dimension
        )
        self.entries_blowup = nn.Sequential(
            HiddenLayer(
                self.entries_embedding.embedding_dim,
                self.entries_embedding.embedding_dim * n_readout,
            ),
            nn.Unflatten(-1, (n_readout, self.entries_embedding.embedding_dim)),
            nn.Flatten(-3, -2),  # adds n_readout extra time dimension (batch_first=True)
        )
        self.register_buffer('mask_blowup', torch.ones((1, 1, n_readout)))

    def forward(self, game_state, sidebar_mask, SidebarAssetName, SidebarContinuous):
        memory = self.memory(game_state)
        sidebar_entries = self.entries_blowup(
            self.entries_embedding(sidebar_mask, SidebarAssetName, SidebarContinuous)
        )
        sidebar_mask = (
            torch.matmul(sidebar_mask.to(torch.float32)[:, :, None], self.mask_blowup)
            .flatten(-2)
            .to(torch.bool)
        )
        result = self.attention(
            memory, sidebar_entries, sidebar_entries, key_padding_mask=sidebar_mask
        )[1].unflatten(-1, (-1, self.n_readout))[:, 0]
        return result


class TD_Action(nn.Module):
    def __init__(self, embedding_dim=1024):
        super().__init__()
        self.main_action = SoftmaxReadout(3, embedding_dim)
        self.sidebar_action = SidebarReadout(embedding_dim)

        n_input_request_type = 12
        self.input_request_type = SoftmaxReadout(n_input_request_type, embedding_dim)
        self.mouse_position = nn.Sequential(
            nn.Linear(embedding_dim, n_input_request_type * 2),
            nn.Sigmoid(),
            nn.Unflatten(-1, (n_input_request_type, 2)),
        )

    def forward(self, game_state, sidebar_mask, SidebarAssetName, SidebarContinuous):
        main_action = self.main_action(game_state)
        sidebar_action = self.sidebar_action(
            game_state, sidebar_mask, SidebarAssetName, SidebarContinuous
        )
        input_request_type = self.input_request_type(game_state)
        mouse_position = self.mouse_position(game_state) * (62 * 24)
        return main_action, sidebar_action, input_request_type, mouse_position


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


def pad_game_states(list_of_game_states, device=None):
    dynamic_objects_mask = compute_key_padding_mask(
        [len(game_state['AssetName']) for game_state in list_of_game_states], device
    )
    sidebar_entries_mask = compute_key_padding_mask(
        [len(game_state['SidebarAssetName']) for game_state in list_of_game_states], device
    )

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

    return dynamic_objects_mask, sidebar_entries_mask, tensors


_static_masks = numpy.zeros((0, 0), dtype=bool)


def compute_key_padding_mask(lengths, device):
    global _static_masks
    """https://discuss.pytorch.org/t/create-a-mask-tensor-using-index/97303/6"""
    max_length = max(lengths)
    if max_length > _static_masks.shape[1]:
        _static_masks = numpy.triu(numpy.ones((max_length + 1, max_length), dtype=bool))
    return torch.tensor(_static_masks[lengths, :max_length], device=device, requires_grad=False)
