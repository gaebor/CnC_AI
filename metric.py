import torch


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


def L1(a, b):
    return torch.abs(a - b).mean()


def L2(a, b):
    return torch.square(a - b).mean()


def C1(a, b):
    return color_difference2(a, b).mean()

