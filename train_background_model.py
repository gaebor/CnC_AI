import argparse
import logging

import sys
import json
import zipfile
from io import BytesIO

from PIL import Image
import torch
import numpy
from torchvision import transforms


import matplotlib.pyplot as plt

import model


class WindowedIterator:
    def __init__(self, extractor, window=5):
        self.extractor = extractor
        self.window = window

    def iter(self):
        self.permutation_mask = numpy.zeros((self.window, self.window), dtype='float32')
        self.target_index = 0
        self.extractor_iter = iter(self.extractor)
        return self

    def next(self, target_array):
        next_element = next(self.extractor_iter)
        target_array[self.target_index] = next_element

        self.permutation_mask = numpy.roll(self.permutation_mask, -1, axis=0)
        self.permutation_mask[-1] = 0
        self.permutation_mask[-1, self.target_index] = 1

        self.target_index += 1
        self.target_index %= self.window

        return self.permutation_mask


def training_data_iterator(filename):
    with zipfile.ZipFile(filename, 'r') as f:
        messages = json.loads(f.read('messages.json'))
        frames = {message['frame']: message for message in messages if 'frame' in message}

        for frame_number in sorted(frames.keys()):
            filename = f'{frame_number:010}.bmp'
            file_content = f.read(filename)
            frame = Image.open(BytesIO(file_content))
            yield torch.Tensor(numpy.array(frame, dtype='float32').transpose((2, 0, 1))).to(
                'cuda'
            ) / 255


def main():
    filename = sys.argv[1]
    window = 5

    predictor = model.Predictor(
        model.ImageEmbedding(time_window=window - 1), model.Generator()
    ).to('cuda')
    optimizer = torch.optim.RMSprop(predictor.parameters(), lr=0.01, momentum=0, alpha=0.5)

    batch = torch.Tensor(numpy.zeros((window, 3, 405, 720), dtype='float32')).to('cuda')

    loss_f = torch.nn.MSELoss()

    toimage = transforms.ToPILImage()
    y = WindowedIterator(training_data_iterator(filename), window=window).iter()
    iter_index = 0
    while True:
        try:
            p = y.next(batch)
            ordered_batch = torch.matmul(torch.Tensor(p).to('cuda'), batch.view(window, -1)).view(
                *batch.shape
            )

            optimizer.zero_grad()
            predicted = predictor(ordered_batch[:-1])[0]

            error = loss_f(predicted, ordered_batch[-1])
            print(error)

            error.backward()
            optimizer.step()

            if iter_index % 60 == 0:
                toimage(torch.cat([ordered_batch[-1], predicted], dim=1).to('cpu').detach()).show()
            iter_index += 1
        except StopIteration:
            break


if __name__ == '__main__':
    main()
