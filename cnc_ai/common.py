from typing import List

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy
from torch.nn.utils.rnn import pad_sequence as pad_sequence_torch
import torch

plt.ion()


def plot_images(*sets_of_images):
    for figure_number, images in enumerate(sets_of_images):
        shape_x, shape_y = images[0].shape[0], images[0].shape[1]
        big_image = numpy.zeros((shape_x, len(images) * shape_y), dtype=float)
        for i, image in enumerate(images):
            big_image[: image.shape[0], i * shape_y : i * shape_y + image.shape[1]] = image

        plt.figure(figure_number)
        plt.imshow(big_image, cmap="hot", norm=colors.Normalize(vmin=0, vmax=1))

    plt.pause(1e-6)


def number_of_digits(n):
    return len(str(n))


def retrieve(t):
    return t.detach().to('cpu').numpy()


def get_log_formatter(indices):
    return ', '.join(
        f'{index_name}: {{:0{len(str(max_value))}d}}/{max_value}'
        for index_name, max_value in indices.items()
    )


def dictmap(d, f):
    return {k: f(v) for k, v in d.items()}


def pad_sequence(tensors: List[numpy.ndarray], padding_value: float = 0.0) -> numpy.ndarray:
    padded_tensors = pad_sequence_torch(
        [torch.tensor(t) for t in tensors], batch_first=True, padding_value=padding_value
    ).numpy()
    return padded_tensors


def numpy_to_torch(dtype: numpy.dtype, float_type: torch.dtype) -> torch.dtype:
    if dtype == numpy.dtype('bool'):
        return torch.bool
    if 'float' in str(dtype):
        return float_type
    return torch.long
