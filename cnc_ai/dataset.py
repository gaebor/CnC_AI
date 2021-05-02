import json

from PIL import Image

import torch
from torchvision import transforms


class CnCRecording(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.player = None
        self.winner = None
        self.frames = {}

        self.transform = transforms.ToTensor()

        with open(self.root_dir + "/messages.json", 'rt') as f:
            messages = json.load(f)

        for message in messages:
            if 'player' in message and self.player is None:
                if 'Color' in message:
                    self.player = message['player']
                    self.player_color = message['Color']
                    self.player_house = message['House']
                else:
                    return
            elif 'frame' in message:
                mouse = message['mouse']
                self.frames[message['frame']] = torch.Tensor(
                    [mouse['x'], mouse['y'], mouse['button']]
                )
            elif 'winner' in message:
                if message['player'] == self.player:
                    self.winner = message['winner']

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        if idx not in self.frames:
            raise IndexError(f'Index {repr(idx)} is out of range!')

        image = self.transform(Image.open(f'{self.root_dir}/{idx:010d}.bmp'))
        return {'image': image, 'mouse': self.frames[idx]}
