# https://github.com/eriklindernoren/PyTorch-GAN

import torchvision.transforms as transforms
from torch.autograd import Variable

from torch import nn
import torch


def soft_inverse_norm(length):
    return (torch.exp(-length) - 1) / -length


class ReshapeLayer(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


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


class ImageEmbedding(nn.Module):
    def __init__(self, n_embedding=1024, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 4, (4, 3), padding=(3, 1)),  # this produces a 408x720 image
            nn.LeakyReLU(),
            nn.Conv2d(4, 4, 3, padding=1),
            nn.LeakyReLU(),
            DownScaleLayer(4, 8, 2),
            nn.LeakyReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.LeakyReLU(),
            DownScaleLayer(8, 16, 2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.LeakyReLU(),
            DownScaleLayer(16, 32, 2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(),
            DownScaleLayer(32, 64, 3),  # at this point 30x17 with 64 channels
            nn.LeakyReLU(),
            ReshapeLayer((-1, 30 * 17 * 64)),
            nn.Linear(30 * 17 * 64, n_embedding),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(n_embedding, n_embedding),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(n_embedding, n_embedding),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class Generator(nn.Module):
    def __init__(self, activation, n_embedding=1024, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_embedding, n_embedding),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(n_embedding, n_embedding),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(n_embedding, 30 * 17 * 64),
            nn.LeakyReLU(),
            ReshapeLayer((-1, 64, 17, 30)),
            UpscaleLayer(64, 32, 3, 3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(),
            UpscaleLayer(32, 16, 3, 2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.LeakyReLU(),
            UpscaleLayer(16, 8, 3, 2),
            nn.LeakyReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.LeakyReLU(),
            UpscaleLayer(8, 4, 3, 2),
            nn.LeakyReLU(),
            nn.Conv2d(4, 4, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(4, 3, (4, 3), padding=(0, 1)),
            activation,
        )

    def forward(self, z):
        return self.layers(z)


class Predictor(nn.Module):
    def __init__(self, activation=nn.Sigmoid(), n_embedding=1024, dropout=0.2):
        super().__init__()
        self.embedding = ImageEmbedding(n_embedding=n_embedding, dropout=dropout)
        self.generator = Generator(activation, n_embedding=n_embedding, dropout=dropout)

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
        input_tensor = torch.cat([latent_embedding, cursor, self.button_embedding(button)], dim=1,)
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
