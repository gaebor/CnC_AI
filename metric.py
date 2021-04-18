import torch
from torchvision import transforms


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
    return (torch.abs(a - b) * _rgb_weights[None, :, None, None]).sum(dim=1)


def calculate_lipsitz_constant_of_image(image, color_metric=color_difference2):
    m = _max(color_metric(image[:, :, :, 1:], image[:, :, :, :-1]), axes=[2, 1])
    m = torch.maximum(m, _max(color_metric(image[:, :, 1:, :], image[:, :, :-1, :]), axes=[2, 1]))
    m = torch.maximum(
        m, _max(color_metric(image[:, :, 1:, 1:], image[:, :, :-1, :-1]) / 2 ** 0.5, axes=[2, 1],),
    )
    m = torch.maximum(
        m, _max(color_metric(image[:, :, 1:, :-1], image[:, :, :-1, 1:]) / 2 ** 0.5, axes=[2, 1],),
    )
    return m


_gaussian_blur = transforms.GaussianBlur(5)


def lipsitz_blur(image):
    lip_mask = calculate_lipsitz_constant_of_image(image) > 1

    blurred = image * lip_mask[:, None, None, None] + _gaussian_blur(image) * (
        ~lip_mask[:, None, None, None]
    )

    return blurred / calculate_lipsitz_constant_of_image(blurred)[:, None, None, None]


def wasserstein(a, b):
    f = lipsitz_blur(a.detach() - b.detach())
    return (f * (a - b)).mean()


def L1(a, b):
    return torch.abs(a - b).mean()


def L2(a, b):
    return torch.square(a - b).mean()


def C1(a, b):
    return color_difference2(a, b).mean()

