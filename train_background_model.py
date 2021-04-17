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
        predictor = model.Predictor(model.ImageEmbedding(), model.Generator()).to('cuda')

    if args.eval:
        predictor.eval()
    else:
        optimizer = torch.optim.RMSprop(predictor.parameters(), lr=args.lr, momentum=0, alpha=0.5)

    loss_f = torch.nn.MSELoss()

    toimage = transforms.ToPILImage()
    previous_time = time.time()
    for _ in range(args.epoch):
        dataset = ImageFolder(args.folder, transform=transforms.ToTensor())
        dataloader = DataLoader(
            dataset, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True
        )
        for iter_index, (batch, _) in enumerate(dataloader):
            batch = batch.to('cuda', non_blocking=True)
            if not args.eval:
                optimizer.zero_grad()
            predicted = predictor(batch)

            error = loss_f(predicted, batch)

            current_time = time.time()
            logging.info(
                "loss: {:e}, fps: {:g}".format(
                    error.to('cpu').detach().numpy(), args.batch / (current_time - previous_time)
                )
            )
            previous_time = current_time
            if not args.eval:
                error.backward()
                optimizer.step()

            if iter_index % 60 == 0:
                toimage(torch.cat([batch[-1], predicted[-1]], dim=1).to('cpu').detach()).show()
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
        '--batch', type=int, default=8, help='batch size',
    )
    parser.add_argument(
        '--epoch', type=int, default=1, help='number of times to iterate over one video',
    )
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument(
        '--model', default='cnc_background_model.pt', help='name of the saved model'
    )
    parser.add_argument('--load', default='', help='model to load and continue training, if any')
    parser.add_argument(
        '--eval',
        default=False,
        action='store_true',
        help='switch to evaulation mode, inference only',
    )
    parser.add_argument(
        '--sample',
        default=10,
        type=int,
        help='sample generated images once every \'sample\' batch',
    )
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(get_params())
