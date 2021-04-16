# -*- coding: utf-8 -*-

import argparse
import logging
import time

import sys
import json
import zipfile
from io import BytesIO

from PIL import Image
import torch
import numpy
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


import matplotlib.pyplot as plt

import model


def main(args):
    if args.load:
        predictor = torch.load(args.load)
    else:
        predictor = model.Predictor(
            model.ImageEmbedding(time_window=args.window), model.Generator()
        ).to('cuda')

    optimizer = torch.optim.RMSprop(predictor.parameters(), lr=args.lr, momentum=0, alpha=0.5)

    loss_f = torch.nn.MSELoss()

    toimage = transforms.ToPILImage()
    previous_time = time.time()
    for _ in range(args.epoch):
        dataset = ImageFolder(args.folder, transform=transforms.ToTensor())
        dataloader = DataLoader(
            dataset, batch_size=args.window + 1, shuffle=False, num_workers=0, pin_memory=True
        )
        for iter_index, (batch, _) in enumerate(dataloader):
            batch = batch.to('cuda', non_blocking=True)
            optimizer.zero_grad()
            predicted = predictor(batch[:-1])[0]

            error = loss_f(predicted, batch[-1])

            current_time = time.time()
            logging.info(
                f"loss: {error.to('cpu').detach().numpy():e}, time: {current_time-previous_time:e}"
            )
            previous_time = current_time

            error.backward()
            optimizer.step()

            if iter_index % 60 == 0:
                toimage(torch.cat([batch[-1], predicted], dim=1).to('cpu').detach()).show()
            iter_index += 1
        torch.save(predictor, args.model)


def get_params():
    parser = argparse.ArgumentParser(
        description="""author: Gábor Borbély, contact: borbely@math.bme.hu""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'folder',
        help='folder of the training videos, saved with '
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
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument(
        '--model', default='cnc_background_model.pt', help='name of the saved model'
    )
    parser.add_argument('--load', default='', help='model to load and continue training, if any')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(get_params())
