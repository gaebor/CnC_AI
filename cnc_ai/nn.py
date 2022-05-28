from torch import nn
import torch


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


class UpscaleLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, upscale):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels * upscale ** 2,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
            nn.PixelShuffle(upscale),
        )


class HiddenLayer(nn.Sequential):
    def __init__(self, in_features, out_features=None, dropout=0.1):
        if out_features is None:
            out_features = in_features
        super().__init__(
            nn.Linear(in_features, out_features), nn.LeakyReLU(), nn.Dropout(p=dropout)
        )


class ConvolutionLayer(nn.Sequential):
    def __init__(self, n_features, dropout=0):
        super().__init__(
            nn.Dropout(p=dropout),
            nn.Conv2d(n_features, n_features, 3, padding=1),
            nn.LeakyReLU(),
        )


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
        self.register_buffer(
            'sub_embedding_sizes',
            torch.tensor([asset_indices.get(k, 1) for k in range(max(asset_indices.keys()) + 1)]),
        )
        self.embedding = nn.Embedding(self.sub_embedding_sizes.sum(), embedding_dim)
        self.embedding_dim = self.embedding.embedding_dim
        self.register_buffer(
            'offsets', torch.cat([torch.tensor([0]), self.sub_embedding_sizes[:-1].cumsum(0)], 0)
        )

    def forward(self, asset_index, shape_index):
        assert (shape_index < nn.functional.embedding(asset_index, self.sub_embedding_sizes)).all()
        indices = nn.functional.embedding(asset_index, self.offsets) + shape_index
        embedding = self.embedding(indices)
        return embedding


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


class SoftmaxReadout(nn.Module):
    def __init__(self, n_readout, in_features, dim=-1):
        super().__init__()
        self.readout = nn.Embedding(n_readout, in_features)
        self.dim = dim

    def forward(self, state):
        logits = torch.matmul(state, self.readout.weight.t())
        weights = torch.nn.functional.softmax(logits.flatten(self.dim), -1)
        result = weights.unflatten(-1, logits.shape[self.dim :])
        return result


def log_beta(alpha, beta, x):
    return (
        torch.log(torch.tensor(x)) * (alpha - 1)
        + torch.log(torch.tensor(1 - x)) * (beta - 1)
        + torch.lgamma(alpha + beta)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
    )
