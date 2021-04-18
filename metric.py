import torch
from torch import nn
from torchvision import transforms

from model import ReshapeLayer


def _max(m, axes):
    for axis in axes:
        m = m.max(dim=axis).values
    return m


def color_difference1(a, b):
    # naive L2 in RGB
    return torch.norm(a - b, dim=1)


_rgb_weights = torch.Tensor([3, 4, 3]).to('cuda')


def color_difference2(a, b):
    # 3*|dR| + 4*|dG|+ 3*|dB|
    return (torch.abs(a - b) * _rgb_weights.to(a.device)[:, None, None]).sum(dim=1)


def calculate_lipsitz_constant_of_image(image):
    m = _max(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]), axes=[3, 2, 1])
    m = torch.maximum(m, _max(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]), axes=[3, 2, 1]))
    m = torch.maximum(m, _max(torch.abs(image[:, 1:, :, :] - image[:, :-1, :, :]), axes=[3, 2, 1]))
    return m


class Critique(nn.Module):
    def __init__(self, image_size, dtype, device='cuda'):
        super().__init__()
        self.image_size = image_size
        self.x = torch.arange(image_size[0], dtype=dtype, device=device)
        self.y = torch.arange(image_size[1], dtype=dtype, device=device)
        self.z = torch.arange(3, dtype=dtype, device=device)
        self.torch_inf = torch.Tensor([float('inf')])[0].to(device)

    def forward(self, basepoints, normal_vectors):
        result = -self.torch_inf
        for i, params in enumerate(zip(basepoints, normal_vectors)):
            new_witness = self._color_gradient(*params)
            result = (torch.maximum if i % 2 == 0 else torch.minimum)(result, new_witness)
        return result

    def _color_gradient(self, x0, n):
        d = (
            ((self.x - x0[0]) * n[0])[None, :, None]
            + ((self.y - x0[1]) * n[1])[None, None, :]
            + ((self.z - x0[2]) * n[2])[:, None, None]
        )
        return d


class CritiqueParams(nn.Module):
    def __init__(self, n_critique, dtype, device='cuda'):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(3, 3, 3, padding=1),
            nn.MaxPool2d(2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(3, 3, 3, padding=1),
            nn.MaxPool2d(2),
            nn.LeakyReLU(inplace=True),
            ReshapeLayer((-1, (405 // 4) * (720 // 4) * 3)),
            nn.Linear((405 // 4) * (720 // 4) * 3, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(1024, n_critique * 6),
            nn.LeakyReLU(inplace=True),
        )
        self.one = torch.ones(1, dtype=dtype, device=device)

    @staticmethod
    def soft_normalize(x):
        length = x.norm(dim=1)
        return x * ((torch.exp(-length) - 1) / -length)[:, None]

    def forward(self, image):
        params = self.layers(image[None, :, :, :])[0].view(-1, 6)
        return params[:, :3], CritiqueParams.soft_normalize(params[:, 3:])


class Wasserstein(nn.Module):
    def __init__(self, image_size, n_critique, dtype, device='cuda'):
        super().__init__()
        self.critique = Critique(image_size, dtype, device)
        self.critique_params = CritiqueParams(n_critique, dtype, device)

    def forward(self, image):
        params = self.critique_params(image)
        witness = self.critique(*params)
        return witness


def L1(a, b):
    return torch.abs(a - b).mean()


def L2(a, b):
    return torch.square(a - b).mean()


def C1(a, b):
    return color_difference2(a, b).mean()

