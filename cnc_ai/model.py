# https://github.com/eriklindernoren/PyTorch-GAN

from torch import nn
import torch

from cnc_ai.TIBERIANDAWN.cnc_structs import (
    all_asset_num_shapes,
    static_tile_names,
    dynamic_object_names,
)


def soft_inverse_norm(length):
    return (torch.exp(-length) - 1) / -length


class ReshapeLayer(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return torch.reshape(x, self.shape)


class DownScaleLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, downscale):
        super().__init__(
            nn.MaxPool2d(downscale, downscale),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )


class DownScaleLayer(nn.Conv2d):
    def __init__(self, in_channels, out_channels, downscale):
        super().__init__(
            in_channels, out_channels, kernel_size=downscale, stride=downscale, padding=0
        )


class UpscaleLayer(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, upscale):
        super().__init__(
            in_channels,
            out_channels * upscale ** 2,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.out_channels = out_channels
        self.upscale = upscale

    def forward(self, x):
        conved = super().forward(x)
        batch_size, _, height, width = conved.shape
        conved = conved.view(
            (batch_size, self.out_channels, self.upscale, self.upscale, height, width)
        )
        upscaled = torch.zeros(
            (conved.shape[0], self.out_channels, height * self.upscale, width * self.upscale),
            dtype=conved.dtype,
            device=conved.device,
        )
        for i in range(self.upscale):
            for j in range(self.upscale):
                upscaled[:, :, i :: self.upscale, j :: self.upscale] = conved[:, :, i, j, :, :]
        return upscaled


class ImageEmbedding(nn.Sequential):
    def __init__(self, n_embedding=1024):
        super().__init__(
            nn.Conv2d(3, 4, 5, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(4, 4, 5, padding=2),
            nn.LeakyReLU(),
            DownScaleLayer(4, 8, 3),
            nn.LeakyReLU(),
            nn.Conv2d(8, 8, 5, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(8, 8, 5, padding=2),
            nn.LeakyReLU(),
            DownScaleLayer(8, 16, 3),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 5, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 5, padding=2),
            nn.LeakyReLU(),
            DownScaleLayer(16, 32, 5),
            ReshapeLayer((-1, 32 * 9 * 16)),
            nn.Linear(32 * 9 * 16, n_embedding),
            nn.LeakyReLU(),
            nn.Linear(n_embedding, n_embedding),
            nn.LeakyReLU(),
        )


class Generator(nn.Sequential):
    def __init__(self, activation, n_embedding=1024):
        super().__init__(
            nn.Linear(n_embedding, n_embedding),
            nn.LeakyReLU(),
            nn.Linear(n_embedding, 32 * 9 * 16),
            nn.LeakyReLU(),
            ReshapeLayer((-1, 32, 9, 16)),
            UpscaleLayer(32, 16, 5, 5),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 5, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 5, padding=2),
            nn.LeakyReLU(),
            UpscaleLayer(16, 8, 3, 3),
            nn.LeakyReLU(),
            nn.Conv2d(8, 8, 5, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(8, 8, 5, padding=2),
            nn.LeakyReLU(),
            UpscaleLayer(8, 4, 3, 3),
            nn.LeakyReLU(),
            nn.Conv2d(4, 4, 5, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(4, 3, 5, padding=2),
            activation,
        )


class DoubleEmbedding(nn.Module):
    def __init__(self, asset_indices, embedding_dim):
        super().__init__()
        self.sub_embedding_sizes = torch.tensor(
            [asset_indices.get(k, 1) for k in range(max(asset_indices.keys()) + 1)]
        )
        self.embedding = nn.Embedding(self.sub_embedding_sizes.sum(), embedding_dim)

        self.offsets = torch.cat([torch.tensor([0]), self.sub_embedding_sizes[:-1].cumsum(0)], 0)

    def forward(self, asset_index, shape_index):
        assert (shape_index < self.sub_embedding_sizes[asset_index]).all()
        indices = self.offsets[asset_index] + shape_index
        embedding = self.embedding(indices)
        return embedding


class MapEmbedding_62_62(nn.Module):
    def __init__(self, n_embedding=1024):
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
            ReshapeLayer((-1, 160 * 4 * 4)),
            nn.Linear(160 * 4 * 4, n_embedding),
            nn.LeakyReLU(),
            nn.Linear(n_embedding, n_embedding),
            nn.LeakyReLU(),
        )

    def forward(self, asset_indices, shape_indices):
        map_embedding = self.asset_embedding(asset_indices, shape_indices).permute(0, 3, 1, 2)
        output = self.convolutions(map_embedding)
        return output


class Predictor(nn.Module):
    def __init__(self, activation=nn.Sigmoid(), n_embedding=1024):
        super().__init__()
        self.embedding = ImageEmbedding(n_embedding=n_embedding)
        self.generator = Generator(activation, n_embedding=n_embedding)

    def forward(self, x):
        return self.generator(self.embedding(x))


class GamePlay(nn.Module):
    def __init__(self, latent_size=1024, n_button=3, num_layers=2, num_ff=1, dropout=0.1):
        super().__init__()
        self.button_embedding = nn.Embedding(n_button, n_button)
        hidden_size = latent_size + 2 + n_button
        self.encoder_layer = nn.LSTM(
            hidden_size, hidden_size, dropout=dropout, num_layers=num_layers
        )

        readout_layers = []
        for _ in range(num_ff):
            readout_layers += [
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout),
            ]
        readout_layers.append(nn.Linear(hidden_size, 2 + n_button))
        self.readout_layer = nn.Sequential(*readout_layers)

    def forward(self, latent_embedding, cursor, button, hidden_state=None, limit=360.0):
        input_tensor = torch.cat([latent_embedding, cursor, self.button_embedding(button)], dim=1)
        hidden_tensor, hidden_state = self.encoder_layer(input_tensor[:, None, :], hidden_state)
        output_tensor = self.readout_layer(hidden_tensor[:, 0, :])
        return (
            cursor_speed_limit(output_tensor[:, :2], limit=limit),
            output_tensor[:, 2:] @ self.button_embedding.weight.t(),
            (hidden_state[0].detach(), hidden_state[1].detach()),
        )


def cursor_speed_limit(predicted_movement, limit=360.0):
    speed = torch.norm(predicted_movement, dim=1)
    return predicted_movement * soft_inverse_norm(speed)[:, None] * limit


def cursor_pos_loss(target_cursor, predicted_cursor):
    return nn.functional.l1_loss(target_cursor, predicted_cursor)


def button_loss(target_button, predicted_button_probabilities):
    return nn.functional.cross_entropy(predicted_button_probabilities, target_button)


class Optimizer(torch.optim.RMSprop):
    def __init__(self, params, lr, weight_decay):
        super().__init__(params, lr=lr, alpha=0.5, weight_decay=weight_decay)


def calculate_asset_num_shapes(names_list):
    names_dict = {v: i for i, v in enumerate(names_list)}
    asset_num_shapes = {
        names_dict[k]: v for k, v in all_asset_num_shapes.items() if k in names_dict
    }
    return asset_num_shapes


class TD_GamePlay(nn.Module):
    def __init__(self):
        super().__init__()
        self.map_embedding = MapEmbedding_62_62(1024)
        self.unit_embedding = DoubleEmbedding(calculate_asset_num_shapes(dynamic_object_names), 16)
        self.owner_embedding = nn.Embedding(10, 3)
        self.pip_embedding = nn.Sequential(nn.Embedding(10, 3), nn.Flatten(start_dim=2))
        self.control_embedding = nn.Embedding(256, 3)  # 0-9 and 255 for default value
        self.cloak_embedding = nn.Embedding(5, 2)
        self.buildable_embedding = nn.Embedding(len(dynamic_object_names), 7)

        self.dynamic_object_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=16 + 3 + 3 * 18 + 3 + 2 + 5,
                nhead=1,
                batch_first=True,
                layer_norm_eps=0,
                dim_feedforward=128,
            ),
            num_layers=2,
        )
        self.siderbar_entries_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=7 + 6, nhead=1, batch_first=True, layer_norm_eps=0, dim_feedforward=16
            ),
            num_layers=2,
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
                torch.mean(self.dynamic_object_encoder(dynamic_objects), 1),
                SidebarInfos,
                torch.mean(self.siderbar_entries_encoder(siderbar_entries), 1),
            ],
            1,
        )
        return internal_state


def pad_game_states(list_of_game_states):
    dynamic_lengths = torch.tensor(
        [len(game_state['AssetName']) for game_state in list_of_game_states]
    )
    sidebar_lengths = torch.tensor(
        [len(game_state['SidebarAssetName']) for game_state in list_of_game_states]
    )
    tensors = {
        **{
            key: torch.stack(
                [torch.tensor(game_state[key]) for game_state in list_of_game_states], 0
            )
            for key in ['StaticAssetName', 'StaticShapeIndex', 'SidebarInfos']
        },
        **{
            key: torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(game_state[key]) for game_state in list_of_game_states],
                batch_first=True,
            )
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
