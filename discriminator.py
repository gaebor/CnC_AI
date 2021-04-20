# -*- coding: utf-8 -*-

import argparse
import logging
import time

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import model
import common


def soft_inverse_norm(length):
    return (torch.exp(-length) - 1) / -length


def soft_normalize(x):
    return x * soft_inverse_norm(x.norm(dim=1))[:, None]


def calculate_lipsitz_constant_of_image(image):
    m = common.max(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]), axes=[3, 2, 1])
    m = torch.maximum(
        m, common.max(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]), axes=[3, 2, 1])
    )
    m = torch.maximum(
        m, common.max(torch.abs(image[:, 1:, :, :] - image[:, :-1, :, :]), axes=[3, 2, 1])
    )
    return m


class Critique(model.Predictor):
    def __init__(self):
        super().__init__(model.ImageEmbedding(), model.Generator(activation=torch.nn.Identity()))

    def forward(self, image):
        witness = super().forward(image)
        lip = calculate_lipsitz_constant_of_image(witness)
        # return witness / lip[:, None, None, None]
        return witness * soft_inverse_norm(lip)[:, None, None, None]

    def loss(self, image1, image2):
        difference = image1 - image2
        witness = self.forward(difference.detach())
        return (witness * difference).mean()


def plot_derivative(image):
    output = torch.zeros_like(image)
    dx = image[1:, :-1, :-1] - image[:-1, :-1, :-1]
    dy = image[-1:, 1:, :-1] - image[:-1, :-1, :-1]
    dz = image[-1:, :-1, 1:] - image[:-1, :-1, :-1]

    output[:-1, :-1, :-1] = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
    return output


def train(args):
    if args.load:
        discriminator = torch.load(args.load)
    else:
        discriminator = Critique()

    discriminator = discriminator.to(args.device)

    optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=args.lr, momentum=0, alpha=0.5)

    toimage = transforms.ToPILImage()
    previous_time = time.time()
    for epoch_index in range(1, args.epoch + 1):
        dataset = ImageFolder(args.folder, transform=transforms.ToTensor())
        dataloader1 = DataLoader(dataset, batch_size=args.batch, shuffle=True, pin_memory=True)
        dataloader2 = DataLoader(dataset, batch_size=args.batch, shuffle=True, pin_memory=True)
        for iter_index, (images1, images2) in enumerate(zip(dataloader1, dataloader2), 1):
            optimizer.zero_grad()
            delta_image = (images1[0] - images2[0]).to(args.device, non_blocking=True)
            witness = discriminator(delta_image)

            error = -(witness * delta_image).mean()
            error.backward()
            optimizer.step()

            current_time = time.time()
            logging.info(
                "epoch: {:0{}d}/{}, iter: {:0{}d}/{}, loss: {:e}, fps: {:g}".format(
                    epoch_index,
                    common.number_of_digits(args.epoch),
                    args.epoch,
                    iter_index,
                    common.number_of_digits(len(dataloader1)),
                    len(dataloader1),
                    common.retrieve(-error).numpy(),
                    len(images1) / (current_time - previous_time),
                )
            )
            previous_time = current_time

            if iter_index % args.sample == 0:
                toimage(
                    torch.cat(
                        [
                            common.retrieve(delta_image[-1]),
                            common.retrieve(witness[-1]),
                            plot_derivative(common.retrieve(witness[-1])),
                        ],
                        dim=1,
                    )
                ).show()
        torch.save(discriminator, args.model)


def get_params():
    parser = argparse.ArgumentParser(
        description="""author: Gábor Borbély, contact: borbely@math.bme.hu""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('folder', help='folder of the training videos')
    parser.add_argument(
        '--batch', type=int, default=8, help='batch size',
    )
    parser.add_argument(
        '--epoch', type=int, default=1, help='number of times to iterate over dataset',
    )
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
    parser.add_argument(
        '--model', default='wasserstein_discriminator.pt', help='name of the saved model',
    )
    parser.add_argument('--load', default='', help='discriminator to load and continue training')
    parser.add_argument('--device', default='cuda', help='device to compute on')

    parser.add_argument(
        '--sample',
        default=20,
        type=int,
        help='sample the generated images once every \'sample\' batch',
    )
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    train(get_params())
