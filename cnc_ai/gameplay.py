# -*- coding: utf-8 -*-

import argparse
import logging
import time
from glob import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.functional import to_pil_image

import cnc_ai.model
from cnc_ai import dataset
from cnc_ai import common


def inference(args):
    embedding_f = torch.load(args.eval).embedding.to(args.device).eval()
    ai_player = torch.load(args.model).to(args.device).eval()

    recording = dataset.CnCRecording(args.folder)
    dataloader = DataLoader(recording, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)

    hidden_state = None
    for iter_index, batch in enumerate(dataloader, 1):

        embedding = embedding_f(batch['image'].to(args.device, non_blocking=True))

        mouse_cursor = batch['mouse'][:, :2].to(args.device)
        pressed_button = batch['mouse'][:, 2].long().to(args.device)

        predicted_cursor_movement, predicted_button_probs, hidden_state = ai_player(
            embedding, mouse_cursor, pressed_button, hidden_state
        )
        predicted_probs = torch.softmax(predicted_button_probs.to('cpu'), dim=1)[0]
        predicted_button = torch.max(predicted_probs, dim=0)[1].numpy()
        predicted_cursor_movement = predicted_cursor_movement[0].to('cpu').numpy()

        print(iter_index, predicted_cursor_movement, predicted_button)

        if iter_index % args.sample == 0:
            new_cursor_pos = (
                torch.minimum(
                    torch.maximum(
                        batch['mouse'][0, :2] + predicted_cursor_movement, torch.Tensor([0, 0])
                    ),
                    torch.Tensor([719, 404]),
                )
                .int()
                .numpy()
            )
            snapshot = batch['image'][0]
            snapshot[:, new_cursor_pos[1], new_cursor_pos[0]] = predicted_probs
            to_pil_image(snapshot).show()


def train(args):
    if args.load:
        ai_player = torch.load(args.load)
    else:
        ai_player = cnc_ai.model.GamePlay(latent_size=args.hidden)
    ai_player = ai_player.to(args.device)

    optimizer = torch.optim.RMSprop(ai_player.parameters(), lr=args.lr, momentum=0, alpha=0.5)

    previous_time = time.time()
    for epoch_index in range(1, args.epoch + 1):
        matches = glob(f'{args.folder}/*.pt') + glob(f'{args.folder}/*/*.pt')
        for match_index, match_filename in enumerate(matches, 1):
            match = torch.load(match_filename)
            data_iterator = DataLoader(
                TensorDataset(*match.values()),
                batch_size=args.memory,
                pin_memory=True,
                shuffle=False,
            )
            hidden_state = None
            for iter_index, batch in enumerate(data_iterator, 1):
                latent_embedding = batch[0].to(args.device)
                cursor = batch[1].to(args.device)
                button = batch[2].to(args.device)

                optimizer.zero_grad()
                predicted_cursor_movement, predicted_button, hidden_state = ai_player(
                    latent_embedding[:-1], cursor[:-1], button[:-1], hidden_state=hidden_state
                )
                error = cnc_ai.model.cursor_pos_loss(
                    cursor[1:], cursor[:-1] + predicted_cursor_movement
                ) + cnc_ai.model.button_loss(button[1:], predicted_button)
                error.backward()
                optimizer.step()

                # TODO advance hidden state one more to keep continuity between batches

                current_time = time.time()
                logging.info(
                    "epoch: {:0{}d}/{}, match: {:0{}d}/{}, iter: {:0{}d}/{}, loss: {:e}, fps: {:g}".format(
                        epoch_index,
                        common.number_of_digits(args.epoch),
                        args.epoch,
                        match_index,
                        common.number_of_digits(len(matches)),
                        len(matches),
                        iter_index,
                        common.number_of_digits(len(data_iterator)),
                        len(data_iterator),
                        common.retrieve(error).numpy(),
                        button.shape[0] / np.array(current_time - previous_time),
                    )
                )
                previous_time = current_time
        torch.save(ai_player, args.model)


def main(args):
    if args.eval:
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
        '--epoch', type=int, default=1, help='number of times to iterate over dataset',
    )
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument(
        '--model',
        default='CnC_TD_gameplay_model.pt',
        help='name of the model (to train or to evaluate)',
    )
    parser.add_argument('--load', default='', help='model to load and continue training')
    parser.add_argument('--device', default='cuda', help='device to compute on')
    parser.add_argument('--hidden', default=1024, type=int, help='size of latent image embedding')
    parser.add_argument(
        '--sample', default=10, type=int, help='samples gameplay during infernece mode'
    )
    parser.add_argument(
        '--memory', default=1024, type=int, help='maximum time window to backpropagate to'
    )
    parser.add_argument(
        '--eval',
        default='',
        type=str,
        help='if set then switch to inferece mode, give the trained background model here.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(get_params())
