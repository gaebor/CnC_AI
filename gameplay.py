# -*- coding: utf-8 -*-

import argparse
import logging
import time
from glob import glob

import torch
from torch.utils.data import DataLoader

import common
import model


def train(args):
    if args.load:
        ai_player = torch.load(args.load)
    else:
        ai_player = model.GamePlay(hidden_dim=args.hidden, num_layers=args.layers)
    ai_player = ai_player.to(args.device)

    optimizer = torch.optim.SGD(ai_player.parameters(), lr=args.lr)

    previous_time = time.time()
    for epoch_index in range(1, args.epoch + 1):
        matches = glob(f'{args.folder}/*.pt') + glob(f'{args.folder}/*/*.pt')
        for iter_index, match_filename in enumerate(matches, 1):
            match = torch.load(match_filename, map_location=args.device)

            optimizer.zero_grad()
            predicted_cursor, predicted_button = ai_player(
                match['latent_embedding'][:-1], match['cursor'][:-1], match['button'][:-1]
            )
            error = ai_player.loss(
                match['cursor'][1:], match['button'][1:], predicted_cursor, predicted_button
            )
            error.backward()
            optimizer.step()

            current_time = time.time()
            logging.info(
                "epoch: {:0{}d}/{}, match: {:0{}d}/{}, loss: {:e}, fps: {:g}".format(
                    epoch_index,
                    common.number_of_digits(args.epoch),
                    args.epoch,
                    iter_index,
                    common.number_of_digits(len(matches)),
                    len(matches),
                    common.retrieve(error).numpy(),
                    match['button'].shape[0] / (current_time - previous_time),
                )
            )
            previous_time = current_time
        torch.save(ai_player, args.model)


def main(args):
    if args.eval:
        inference(args)
    else:
        train(args)


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
        '--epoch', type=int, default=1, help='number of times to iterate over dataset',
    )
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument(
        '--model',
        default='CnC_TD_gameplay_model.pt',
        help='name of the model (to train or to evaluate)',
    )
    parser.add_argument('--load', default='', help='model to load and continue training')
    parser.add_argument('--device', default='cuda', help='device to compute on')
    parser.add_argument('--hidden', default=1024, type=int, help='hidden layer size in LSTM')
    parser.add_argument('--layers', default=4, type=int, help='number of layers in LSTM')
    parser.add_argument(
        '--eval',
        default=False,
        action='store_true',
        help='switch to evaulation mode, inference only',
    )
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(get_params())
