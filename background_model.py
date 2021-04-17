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


def number_of_digits(n):
    return len(str(n))


def inference(args):
    embedding_f = torch.load(args.model).embedding
    embedding_f.eval()

    previous_time = time.time()

    subdirectories = map(lambda x: args.folder + '/' + x, listdir(args.folder))
    subdirectories = sorted(list(filter(path.isdir, subdirectories)))

    makedirs("False", exist_ok=True)
    makedirs("True", exist_ok=True)

    for folder_index, subfolder in enumerate(subdirectories):
        recording = dataset.CnCRecording(subfolder)
        subfolder = path.basename(subfolder)

        embeddings, mouse_events = [], []
        dataloader = DataLoader(recording, batch_size=args.batch, shuffle=False, pin_memory=True)
        for iter_index, batch in enumerate(dataloader):
            embeddings.append(embedding_f(batch['image'].to('cuda', non_blocking=True)).detach())
            mouse_events.append(batch['mouse'])

            current_time = time.time()
            logging.info(
                "recording {:0{}d} of {}: {}, iter: {:0{}d}/{}, fps: {:g}".format(
                    folder_index + 1,
                    number_of_digits(len(subdirectories)),
                    len(subdirectories),
                    subfolder,
                    iter_index + 1,
                    number_of_digits(len(dataloader)),
                    len(dataloader),
                    args.batch / (current_time - previous_time),
                )
            )
            previous_time = current_time

        torch.save(
            torch.cat(
                [torch.cat(embeddings, dim=0).to('cpu'), torch.cat(mouse_events, dim=0)], dim=1
            ),
            f"{recording.winner}/{subfolder}.pt",
        )


def train(args):
    if args.load:
        predictor = torch.load(args.load)
    else:
        predictor = model.Predictor(model.ImageEmbedding(), model.Generator()).to('cuda')

    optimizer = torch.optim.RMSprop(predictor.parameters(), lr=args.lr, momentum=0, alpha=0.5)

    loss_f = torch.nn.MSELoss()

    toimage = transforms.ToPILImage()
    previous_time = time.time()
    for epoch_index in range(args.epoch):
        dataset = ImageFolder(args.folder, transform=transforms.ToTensor())
        dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, pin_memory=True)
        for iter_index, (batch, _) in enumerate(dataloader):
            batch = batch.to('cuda', non_blocking=True)
            optimizer.zero_grad()
            predicted = predictor(batch)

            error = loss_f(predicted, batch)

            current_time = time.time()
            logging.info(
                "epoch: {:0{}d}/{}, iter: {:0{}d}/{}, loss: {:e}, fps: {:g}".format(
                    epoch_index + 1,
                    number_of_digits(args.epoch + 1),
                    args.epoch,
                    iter_index + 1,
                    number_of_digits(len(dataloader)),
                    len(dataloader),
                    error.to('cpu').detach().numpy(),
                    args.batch / (current_time - previous_time),
                )
            )
            previous_time = current_time

            error.backward()
            optimizer.step()

            if iter_index % args.sample == 0:
                toimage(torch.cat([batch[-1], predicted[-1]], dim=1).to('cpu').detach()).show()
            iter_index += 1
        torch.save(predictor, args.model)


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
        '--batch', type=int, default=8, help='batch size',
    )
    parser.add_argument(
        '--epoch', type=int, default=1, help='number of times to iterate over dataset',
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
        default=20,
        type=int,
        help='sample generated images once every \'sample\' batch',
    )
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(get_params())
