import argparse
from subprocess import Popen
from os import mkdir
import ctypes
from random import choice
from itertools import chain
from datetime import datetime
import pickle

from tqdm import trange
import numpy

import tornado.web
import tornado.websocket
import tornado.ioloop

from cnc_ai.common import dictmap

from cnc_ai.TIBERIANDAWN import cnc_structs
from cnc_ai.TIBERIANDAWN.agent import NNAgent, SimpleAgent, mix_actions
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
    parser.add_argument(
        '--dll',
        default='TiberianDawn.dll',
        help='path to the game DLL, absolute or relative from `dir`. '
        'This DLL is released with the game itself but it was opensourced too: '
        'https://github.com/electronicarts/CnC_Remastered_Collection',
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
    parser.add_argument(
        '-D',
        dest='spawn',
        default=True,
        action='store_false',
        help="don't spawn subprocesses to run the DLL (set by '--exe'). "
        "this is good for debugging when you start the dll wrapper separately.",
    )
    parser.add_argument(
        '-P',
        '--print',
        default=False,
        action='store_true',
        help="print out what was the terminal state of the game(s)",
    )
    parser.add_argument(
        '-T', '--train', default=False, action='store_true', help="train at the end of the game"
    )
    return parser.parse_args()


simple_agent = SimpleAgent()


class GameHandler(tornado.websocket.WebSocketHandler):
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
                nn_actions = GameHandler.agent(**game_state_tensor)
                simple_actions = simple_agent(**game_state_tensor)
                actions = mix_actions(
                    simple_actions,
                    [action[-1] for action in nn_actions],
                    (numpy.arange(len(simple_actions[0])) % 2).astype(bool),
                )
                for game, per_game_actions in zip(
                    GameHandler.games, zip(*map(GameHandler.split_per_games, actions))
                ):
                    game.game_actions.append(per_game_actions)
                    message = b''
                    for player, action in enumerate(zip(*per_game_actions)):
                        message += render_action(player, *action)
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

    def init_game(self):
        self.folder = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%S.%fs") + '_' + str(id(self))

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
                    MPlayerGoodies=0,
                    MPlayerGhosts=0,
                    MPlayerSolo=1,
                    MPlayerUnitCount=0,
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
        mkdir(self.folder)
        with open(self.folder + '/players.pkl', 'wb') as f:
            pickle.dump(self.players, f)
        with open(self.folder + '/loser_mask.txt', 'wt') as f:
            print(self.loser_mask, file=f)
        with open(self.folder + '/game_states.npy', 'wb') as f:
            numpy.save(f, self.game_states)
        with open(self.folder + '/game_actions.pkl', 'wb') as f:
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
            padded_tensors, lambda t: t.reshape(length, n_players, *t.shape[1:])
        )
        return tensors_in_device

    @classmethod
    def all_game_actions_to_tensor(cls):
        length = min(len(game.game_actions) for game in cls.games)
        actions = numpy.concatenate([game.game_actions[:length] for game in cls.games], axis=2)
        buttons = actions[:, 0, :].astype('int64')
        mouse_x = actions[:, 1, :]
        mouse_y = actions[:, 2, :]
        return buttons, mouse_x, mouse_y

    @classmethod
    def last_game_states_to_tensor(cls):
        tensors = list(chain(*(game.game_states[-1] for game in cls.games)))
        padded_tensors = pad_game_states(tensors)
        tensors_in_device = dictmap(padded_tensors, lambda t: t.reshape(1, *t.shape))
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
        rewards = numpy.array(
            [
                0 if game.loser_mask == 0 else (-1 if game.loser_mask & (1 << player) else 1)
                for game in cls.games
                for player in range(len(game.players))
            ]
        )
        return rewards

    @classmethod
    def train(cls):
        game_state_tensor = cls.all_game_states_to_tensor()
        rewards = cls.get_rewards()
        actions = cls.all_game_actions_to_tensor()
        cls.agent.learn(game_state_tensor, actions, rewards)

    @classmethod
    def configure(cls, agent, n_games=2, end_limit=10_000, players=()):
        cls.n_games = n_games
        cls.agent = agent
        cls.end_limit = end_limit
        cls.tqdm = trange(end_limit)
        cls.players = list(players)
        cls.games = []
        cls.ended_games = []


def main():
    args = get_args()

    if args.load:
        agent = NNAgent.load(args.load)
    else:
        agent = NNAgent()  # hyperparameters can come here
    agent.to(args.device)
    agent.init_optimizer()  # hyperparameters can come here

    GameHandler.configure(
        agent,
        args.n,
        args.end_limit,
        [
            cnc_structs.CNCPlayerInfoStruct(
                GlyphxPlayerID=314159265,
                Name=b"ai1",
                House=127,
                Team=0,
                AllyFlags=0,
                ColorIndex=127,
                IsAI=False,
                StartLocationIndex=127,
            ),
            cnc_structs.CNCPlayerInfoStruct(
                GlyphxPlayerID=271828182,
                Name=b"ai2",
                House=127,
                Team=1,
                AllyFlags=0,
                ColorIndex=127,
                IsAI=False,
                StartLocationIndex=127,
            ),
        ],
    )

    application = tornado.web.Application([(r"/", GameHandler)])
    application.listen(args.port)
    if args.spawn:
        for i in range(args.n):
            tornado.ioloop.IOLoop.current().call_later(
                0.1 * i,
                lambda: Popen([args.exe, str(args.port), args.dir, args.dll]),
            )
    tornado.ioloop.IOLoop.current().start()
    GameHandler.tqdm.close()

    for game in GameHandler.games:
        print(game.folder, game.loser_mask)
        if args.print:
            for i in range(len(game.players)):
                print(i)
                game.print_what_player_sees(i)
    if args.train:
        GameHandler.train()
    if args.save:
        agent.save(args.save)


if __name__ == '__main__':
    main()
