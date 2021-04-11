# -*- coding: utf-8 -*-

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


def main(args):
    if args.load:
        predictor = torch.load(args.load)
    else:
        predictor = model.Predictor(
            model.ImageEmbedding(time_window=args.window), model.Generator()
        ).to('cuda')

    optimizer = torch.optim.RMSprop(predictor.parameters(), lr=0.01, momentum=0, alpha=0.5)

    loss_f = torch.nn.MSELoss()

    toimage = transforms.ToPILImage()

    for _ in range(args.epoch):
        for filename in args.folder:
            batch = torch.Tensor(numpy.zeros((args.window + 1, 3, 405, 720), dtype='float32')).to(
                'cuda'
            )
            y = WindowedIterator(training_data_iterator(filename), window=args.window + 1).iter()
            iter_index = 0
            while True:
                try:
                    p = y.next(batch)
                except StopIteration:
                    break
                ordered_batch = torch.matmul(
                    torch.Tensor(p).to('cuda'), batch.view(args.window + 1, -1)
                ).view(*batch.shape)

                optimizer.zero_grad()
                predicted = predictor(ordered_batch[:-1])[0]

                error = loss_f(predicted, ordered_batch[-1])
                logging.info(error.to('cpu').detach().numpy())

                error.backward()
                optimizer.step()

                if iter_index % 60 == 0:
                    toimage(
                        torch.cat([ordered_batch[-1], predicted], dim=1).to('cpu').detach()
                    ).show()
                iter_index += 1
        torch.save(predictor, args.model)


def get_params():
    parser = argparse.ArgumentParser(
        description="""author: Gábor Borbély, contact: borbely@math.bme.hu""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'folder',
        nargs='+',
        help='folder(s) of the training video(s), saved with '
        'https://github.com/gaebor/CnC_Remastered_Collection/blob/ai/grab_websocket.py',
    )
    parser.add_argument(
        '--window',
        type=int,
        default=4,
        help='time windows to consider when calculating the next frame',
    )
    parser.add_argument(
        '--epoch', type=int, default=1, help='number of times to iterate over one video',
    )
    parser.add_argument('--model', default='cnc_background_model', help='name of the saved model')
    parser.add_argument('--load', default='', help='model to load and continue training, if any')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    main(get_params())
