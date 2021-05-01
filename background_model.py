# -*- coding: utf-8 -*-

import argparse
import logging
import time
from os import listdir, makedirs
from os import path

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import model
import dataset
import common

from discriminator import Critique


def inference(args):
    embedding_f = torch.load(args.model).embedding.to(args.device).eval()

    previous_time = time.time()

    subdirectories = map(lambda x: args.folder + '/' + x, listdir(args.folder))
    subdirectories = sorted(list(filter(path.isdir, subdirectories)))

    for gameoutcome in [True, False, None]:
        makedirs(f"{args.eval}/{gameoutcome}", exist_ok=True)

    for folder_index, subfolder in enumerate(subdirectories, 1):
        recording = dataset.CnCRecording(subfolder)
        subfolder = path.basename(subfolder)

        embeddings, mouse_events = [], []
        dataloader = DataLoader(
            recording,
            batch_size=args.batch,
            shuffle=False,
            pin_memory=True,
            num_workers=args.workers,
        )
        for iter_index, batch in enumerate(dataloader, 1):
            embeddings.append(
                embedding_f(batch['image'].to(args.device, non_blocking=True)).to('cpu')
            )
            mouse_events.append(batch['mouse'])

            current_time = time.time()
            logging.info(
                "recording {:0{}d} of {}: {}, iter: {:0{}d}/{}, fps: {:g}".format(
                    folder_index,
                    common.number_of_digits(len(subdirectories)),
                    len(subdirectories),
                    subfolder,
                    iter_index,
                    common.number_of_digits(len(dataloader)),
                    len(dataloader),
                    batch['mouse'].shape[0] / (current_time - previous_time),
                )
            )
            previous_time = current_time

        mouse_events = torch.cat(mouse_events, dim=0)
        torch.save(
            {
                'latent_embedding': torch.cat(embeddings, dim=0),
                'cursor': mouse_events[:, :2],
                'button': mouse_events[:, -1].long(),
            },
            f"{args.eval}/{recording.winner}/{subfolder}.pt",
        )


def train(args):
    if args.load:
        predictor = torch.load(args.load)
    else:
        predictor = model.Predictor(
            model.ImageEmbedding(), model.Generator(activation=torch.nn.Sigmoid())
        )
    predictor = predictor.to(args.device)

    if args.metric == 'L2':
        loss_f = torch.nn.MSELoss()
    elif args.metric == 'L1':
        loss_f = torch.nn.L1Loss()
    else:
        loss_f = torch.load(args.metric).to(args.device).loss

    optimizer = torch.optim.RMSprop(predictor.parameters(), lr=args.lr, momentum=0, alpha=0.5)

    toimage = transforms.ToPILImage()
    previous_time = time.time()
    for epoch_index in range(1, args.epoch + 1):
        dataset = ImageFolder(args.folder, transform=transforms.ToTensor())
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch,
            shuffle=True,
            pin_memory=True,
            num_workers=args.workers,
        )
        for iter_index, (batch, _) in enumerate(dataloader, 1):
            batch = batch.to(args.device, non_blocking=True)

            optimizer.zero_grad()
            predicted = predictor(batch)
            error = loss_f(predicted, batch)
            error.backward()
            optimizer.step()

            current_time = time.time()
            logging.info(
                "epoch: {:0{}d}/{}, iter: {:0{}d}/{}, loss: {:e}, fps: {:g}".format(
                    epoch_index,
                    common.number_of_digits(args.epoch),
                    args.epoch,
                    iter_index,
                    common.number_of_digits(len(dataloader)),
                    len(dataloader),
                    common.retrieve(error).numpy(),
                    batch.shape[0] / (current_time - previous_time),
                )
            )
            previous_time = current_time

            if iter_index % args.sample == 0:
                toimage(
                    torch.cat([common.retrieve(batch[-1]), common.retrieve(predicted[-1])], dim=1,)
                ).show()
        torch.save(predictor, args.model)


def main(args):
    if args.eval != '':
        with torch.no_grad():
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
        '--batch', type=int, default=8, help='batch size',
    )
    parser.add_argument(
        '--epoch', type=int, default=1, help='number of times to iterate over dataset',
    )
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument(
        '--model',
        default='CnC_TD_background_model.pt',
        help='name of the model (to train or to evaluate)',
    )
    parser.add_argument('--load', default='', help='model to load and continue training')
    parser.add_argument('--device', default='cuda', help='device to compute on')
    parser.add_argument(
        '--eval',
        default='',
        type=str,
        help='if set then switch to inference mode and render video into the given folder',
    )
    parser.add_argument(
        '--sample',
        default=20,
        type=int,
        help='sample the generated images once every \'sample\' batch',
    )
    parser.add_argument(
        '--metric', default='L1', help='\'L1\', \'L2\' or a discriminator model name'
    )
    parser.add_argument('--workers', default=1, type=int, help='number of workers to read images')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(get_params())
