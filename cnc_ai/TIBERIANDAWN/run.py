import argparse
from os import spawnl, P_NOWAIT, mkdir
import ctypes
from random import choice
from itertools import chain
from datetime import datetime
import pickle

from tqdm import trange
import numpy
import torch

import tornado.web
import tornado.websocket
import tornado.ioloop


from cnc_ai.nn import load, save, interflatten
from cnc_ai.common import dictmap
from cnc_ai.TIBERIANDAWN import cnc_structs
from cnc_ai.TIBERIANDAWN.model import TD_GamePlay
from cnc_ai.TIBERIANDAWN.bridge import (
    pad_game_states,
    render_action,
    encode_list,
    render_add_player_command,
)


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--port', default=8888, type=int, help=' ')
    parser.add_argument(
        '-n', '--n', default=2, type=int, help='number of games to play simultaneously'
    )
    parser.add_argument(
        '-l',
        '--limit',
        '--end-limit',
        '--end_limit',
        dest='end_limit',
        default=10_000,
        type=int,
        help='stop game after this many iterations if still going',
    )
    parser.add_argument('--exe', default='TIBERIANDAWN_wrapper.exe', help='absolute path')
    parser.add_argument(
        '--dir',
        default=r'C:\Program Files (x86)\Steam\steamapps\common\CnCRemastered',
        help='absolute path',
    )
    parser.add_argument('-d', '--device', default='cpu', help='pytorch device')
    parser.add_argument(
        '--load',
        default='',
        help='loads model from pytorch model file '
        'rather than initializing a new one on the beginning',
    )
    parser.add_argument(
        '--save', default='', help='save model to pytorch model file after the game(s)'
    )
    return parser.parse_args()


class GameHandler(tornado.websocket.WebSocketHandler):
    games = []
    players = []
    chdir = '.'
    end_limit = 10_000
    nn = None
    device = 'cpu'
    n_games = 0
    ended_games = []

    def on_message(self, message):
        if len(message) == 1:
            self.loser_mask = ctypes.c_ubyte.from_buffer_copy(message).value
            self.close()
        else:
            # recieved the current game state
            offset = 0
            self.game_states.append([])
            for _ in range(len(self.players)):
                self.game_states[-1].append(cnc_structs.convert_to_np(message[offset:]))
                offset += cnc_structs.get_game_state_size(message[offset:])

            if len(self.game_states) >= GameHandler.end_limit:
                self.close()
                return

            # sync games
            if (GameHandler.n_games == len(GameHandler.games)) and all(
                len(game.game_states) == len(self.game_states) for game in GameHandler.games
            ):
                GameHandler.tqdm.update()
                game_state_tensor = GameHandler.last_game_states_to_tensor()
                action_parameters = GameHandler.nn(**game_state_tensor)
                actions = interflatten(GameHandler.nn.actions.sample, *action_parameters)
                for game, per_game_actions in zip(
                    GameHandler.games,
                    GameHandler.split_per_games(zip(*GameHandler.extract_actions(*actions))),
                ):
                    game.game_actions.append(per_game_actions)
                    message = b''
                    for player, action in zip(game.players, per_game_actions):
                        message += render_action(player.GlyphxPlayerID, *action)
                    game.write_message(message, binary=True)

    def open(self):
        GameHandler.games.append(self)
        self.loser_mask = 0
        self.game_states = []
        self.game_actions = []
        self.set_nodelay(True)
        self.init_game()

    def on_close(self):
        GameHandler.ended_games.append(self)
        self.end_game()
        for game in GameHandler.games:
            if game not in GameHandler.ended_games:
                game.close()

        GameHandler.destroy_if_all_stopped()

    def add_players(self):
        colors = set(range(6))
        self.players = []
        for player in GameHandler.players:
            player = cnc_structs.CNCPlayerInfoStruct.from_buffer_copy(bytes(player))
            if player.ColorIndex < 0 or player.ColorIndex >= 6:
                player.ColorIndex = choice(list(colors))
                colors -= {player.ColorIndex}
            if player.House not in [0, 1]:
                player.House = choice([0, 1])

            self.write_message(render_add_player_command(player), binary=True)
            self.players.append(player)

    def print_what_player_sees(self, player):
        game_state = self.game_states[-1][player]
        print(cnc_structs.render_game_state_terminal(game_state))

    def assess_players_performance(self):
        scores = []
        if len(self.game_states[-1]) > 0:
            for player, game_state in zip(self.players, self.game_states[-1]):
                scores.append(cnc_structs.score(game_state, player.ColorIndex))
            loser_mask = sum(1 << i for i in range(len(self.players)) if scores[i] < max(scores))
        else:
            loser_mask = 0
        return loser_mask

    def end_game(self):
        if self.loser_mask == 0:
            self.loser_mask = self.assess_players_performance()
        if self.loser_mask > 0:
            self.save_gameplay()
            for i in range(len(self.players)):
                self.print_what_player_sees(i)

    def init_game(self):
        # change to the directory where CnC is installed
        buffer = bytes(ctypes.c_uint32(0)) + GameHandler.chdir.encode('utf8') + b'\0'
        self.write_message(buffer, binary=True)

        # init dll
        buffer = bytes(ctypes.c_uint32(1))
        buffer += b'TiberianDawn.dll\0'
        buffer += b'-CDDATA\\CNCDATA\\TIBERIAN_DAWN\\CD1\0'
        self.write_message(buffer, binary=True)

        # communicate asset names
        buffer = bytes(ctypes.c_uint32(9))
        buffer += encode_list(cnc_structs.static_tile_names)
        self.write_message(buffer, binary=True)

        buffer = bytes(ctypes.c_uint32(10))
        buffer += encode_list(cnc_structs.dynamic_object_names)
        self.write_message(buffer, binary=True)

        # add players
        self.add_players()

        # start game
        buffer = bytes(ctypes.c_uint32(3))
        buffer += bytes(
            cnc_structs.StartGameArgs(
                cnc_structs.CNCMultiplayerOptionsStruct(
                    MPlayerCount=2,
                    MPlayerBases=1,
                    MPlayerCredits=5000,
                    MPlayerTiberium=1,
                    MPlayerGoodies=1,
                    MPlayerGhosts=0,
                    MPlayerSolo=1,
                    MPlayerUnitCount=1,
                    IsMCVDeploy=False,
                    SpawnVisceroids=False,
                    EnableSuperweapons=True,
                    MPlayerShadowRegrow=False,
                    MPlayerAftermathUnits=True,
                    CaptureTheFlag=False,
                    DestroyStructures=False,
                    ModernBalance=True,
                ),
                50,
                7,
                2,
            )
        )
        self.write_message(buffer, binary=True)

    @classmethod
    def destroy_if_all_stopped(cls):
        if set(cls.games) == set(cls.ended_games):
            tornado.ioloop.IOLoop.current().stop()

    def save_gameplay(self):
        folder = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%S.%fs") + '_' + str(id(self))
        mkdir(folder)
        with open(folder + '/players.pkl', 'wb') as f:
            pickle.dump(self.players, f)
        with open(folder + '/loser_mask.txt', 'wt') as f:
            print(self.loser_mask, file=f)
        with open(folder + '/game_states.npy', 'wb') as f:
            numpy.save(f, self.game_states)
        with open(folder + '/game_actions.pkl', 'wb') as f:
            pickle.dump(self.game_actions, f)

    @classmethod
    def all_game_states_to_tensor(cls):
        length = min(len(game.game_actions) for game in cls.games)
        n_players = sum(len(game.players) for game in cls.games)
        tensors = [
            game_state
            for i in range(length)
            for game in cls.games
            for game_state in game.game_states[i]
        ]
        padded_tensors = pad_game_states(tensors)
        tensors_in_device = dictmap(
            padded_tensors, lambda t: t.to(cls.device).reshape(length, n_players, *t.shape[1:])
        )
        return tensors_in_device

    @classmethod
    def all_game_actions_to_tensor(cls):
        length = min(len(game.game_actions) for game in cls.games)
        actions = numpy.concatenate([game.game_actions[:length] for game in cls.games], axis=1)
        buttons = torch.tensor(actions[:, :, 0], dtype=torch.int64, device=cls.device)
        mouse_x = torch.tensor(actions[:, :, 1], dtype=torch.float32, device=cls.device)
        mouse_y = torch.tensor(actions[:, :, 2], dtype=torch.float32, device=cls.device)
        return buttons, mouse_x, mouse_y

    @classmethod
    def last_game_states_to_tensor(cls):
        tensors = list(chain(*(game.game_states[-1] for game in cls.games)))
        padded_tensors = pad_game_states(tensors)
        tensors_in_device = dictmap(
            padded_tensors, lambda t: t.to(cls.device).reshape(1, *t.shape)
        )
        return tensors_in_device

    @classmethod
    def split_per_games(cls, l):
        result = []
        l = iter(l)
        for game in cls.games:
            result.append([next(l) for _ in range(len(game.players))])
        return result

    @classmethod
    def get_rewards(cls):
        rewards = torch.tensor(
            [
                0 if game.loser_mask == 0 else (-1 if game.loser_mask & (1 << player) else 1)
                for game in cls.games
                for player in range(len(game.players))
            ],
            dtype=torch.float32,
            device=cls.device,
        )
        return rewards

    @classmethod
    def train(cls):
        game_state_tensor = cls.all_game_states_to_tensor()
        action_parameters = cls.nn(**game_state_tensor)
        actions = cls.all_game_actions_to_tensor()
        log_prob = interflatten(GameHandler.nn.actions.evaluate, *action_parameters, *actions).sum(
            axis=0
        )
        objective = log_prob.dot(cls.get_rewards())

    @staticmethod
    def extract_actions(*actions):
        return tuple(action[-1].cpu().numpy() for action in actions)


def main():
    args = get_args()

    GameHandler.device = args.device
    GameHandler.n_games = args.n
    GameHandler.nn = (
        (TD_GamePlay() if args.load == '' else load(TD_GamePlay, args.load))
        .to(GameHandler.device)
        .train(False)
    )

    GameHandler.chdir = args.dir
    GameHandler.end_limit = args.end_limit
    GameHandler.players.append(
        cnc_structs.CNCPlayerInfoStruct(
            GlyphxPlayerID=314159265,
            Name=b"ai1",
            House=127,
            Team=0,
            AllyFlags=0,
            ColorIndex=127,
            IsAI=False,
            StartLocationIndex=127,
        )
    )
    GameHandler.players.append(
        cnc_structs.CNCPlayerInfoStruct(
            GlyphxPlayerID=271828182,
            Name=b"ai2",
            House=127,
            Team=1,
            AllyFlags=0,
            ColorIndex=127,
            IsAI=False,
            StartLocationIndex=127,
        )
    )

    application = tornado.web.Application([(r"/", GameHandler)])
    application.listen(args.port)
    for _ in range(args.n):
        spawnl(P_NOWAIT, args.exe, args.exe, str(args.port))
    GameHandler.tqdm = trange(args.end_limit)
    tornado.ioloop.IOLoop.current().start()
    GameHandler.tqdm.close()
    GameHandler.train()
    if args.save != '':
        GameHandler.nn.reset()
        save(GameHandler.nn, args.save)


if __name__ == '__main__':
    main()
