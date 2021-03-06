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
from cnc_ai.TIBERIANDAWN.bridge import pad_game_states, encode_list


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
    parser.add_argument(
        '-T',
        '--time-window',
        default=200,
        type=int,
        help='limits the maximum depth in time to backpropagate to',
    )
    parser.add_argument(
        '--exe', default='TIBERIANDAWN_wrapper.exe', metavar='absolute path', help=' '
    )
    parser.add_argument(
        '--dir',
        default=r'C:\Program Files (x86)\Steam\steamapps\common\CnCRemastered',
        metavar='absolute path',
        help='The game installation directory. '
        'Buy the game on Steam: https://store.steampowered.com/app/1213210',
    )
    parser.add_argument(
        '--dll',
        default='TiberianDawn.dll',
        help='Path to the game DLL, absolute or relative from `dir`. '
        'This DLL is released with the game itself but it was opensourced too: '
        'https://github.com/electronicarts/CnC_Remastered_Collection',
    )
    parser.add_argument('-d', '--device', default='cpu', help='pytorch device')
    parser.add_argument(
        '--load',
        default='',
        help='Load model from pytorch model file '
        'rather than initializing a new one on the beginning.',
    )
    parser.add_argument(
        '--save', default='', help='Save model to pytorch model file after the game(s).'
    )
    parser.add_argument(
        '-D',
        dest='spawn',
        default=True,
        action='store_false',
        help="Don't spawn subprocesses to run the DLL (set by '--exe'). "
        "This is good for debugging when you start the dll wrapper separately.",
    )
    parser.add_argument(
        '-P',
        '--print',
        default=False,
        action='store_true',
        help="Print out what was the terminal state of the game(s).",
    )
    parser.add_argument('--train', default=0, type=int, help="Train at the end of the game.")
    parser.add_argument(
        '-r',
        '--record',
        default=False,
        action='store_true',
        help="Save recording of the games for later inspection or training. "
        "If you train at the end of the games than save recording may not be necessary.",
    )
    parser.add_argument(
        '-a',
        '--agents',
        dest='agents',
        default='AIvAI',
        choices=['AIvAI', 'AIvNN', 'NNvNN'],
        help=" ",
    )
    return parser.parse_args()


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

                if 'NN' in GameHandler.agents:
                    nn_actions = GameHandler.nn_agent(**game_state_tensor)
                    nn_actions = [action[-1] for action in nn_actions]
                if 'AI' in GameHandler.agents:
                    simple_actions = GameHandler.simple_agent(**game_state_tensor)

                if GameHandler.agents == 'NNvNN':
                    actions = nn_actions
                elif GameHandler.agents == 'AIvAI':
                    actions = simple_actions
                else:
                    actions = mix_actions(
                        simple_actions,
                        nn_actions,
                        (numpy.arange(len(simple_actions[0])) % 2).astype(bool),
                    )
                # print(decode_action(actions[0], game_state_tensor['SidebarAssetName'][0]))
                for game, per_game_actions in zip(
                    GameHandler.games, zip(*map(GameHandler.split_per_games, actions))
                ):
                    game.game_actions.append(per_game_actions)
                    message = b''
                    for player, action in enumerate(zip(*per_game_actions)):
                        message += cnc_structs.ActionRequestArgs(player, *action).render_message()
                    game.write_message(message, binary=True)

    def open(self):
        GameHandler.games.append(self)
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

            self.write_message(player.render_message(), binary=True)
            self.players.append(player)

    def print_what_player_sees(self, player):
        game_state = self.game_states[-1][player]
        print(cnc_structs.render_game_state_terminal(game_state))

    def assess_players_performance(self):
        scores = self.compute_scores()
        loser_mask = sum(1 << i for i in range(len(self.players)) if scores[i] < max(scores))
        return loser_mask

    def compute_scores(self):
        scores = [0] * len(self.players)
        if len(self.game_states[-1]) > 0:
            for i, (player, game_state) in enumerate(zip(self.players, self.game_states[-1])):
                scores[i] = cnc_structs.score(game_state, player.ColorIndex)
        return scores

    def end_game(self):
        if self.loser_mask == 0:
            self.loser_mask = self.assess_players_performance()

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

        self.loser_mask = 0
        self.game_states = []
        n_players = len(self.players)
        # Actions are shifted by one
        # This is actually the previous action (None in the first turn)
        self.game_actions = [
            ([0] * n_players, [0] * n_players, [744.0] * n_players, [744.0] * n_players)
        ]

        # start game
        buffer = self.start_game_args.render_message()
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
    def all_game_states_and_actions(cls):
        length = min(len(game.game_actions) for game in cls.games) - 1
        n_players = sum(len(game.players) for game in cls.games)
        tensors = [
            game_state
            for i in range(length)
            for game in cls.games
            for game_state in game.game_states[i]
        ]
        padded_tensors = dictmap(
            pad_game_states(tensors), lambda t: t.reshape(length, n_players, *t.shape[1:])
        )
        all_actions = numpy.concatenate(
            [game.game_actions[: length + 1] for game in cls.games], axis=2
        )
        padded_tensors.update(
            previous_action_item=numpy.take_along_axis(
                padded_tensors['SidebarAssetName'],
                all_actions[:-1, 0, :].astype('int64')[:, :, None],
                axis=2,
            )[:, :, 0].astype('int64'),
            previous_action_type=all_actions[:-1, 1, :].astype('int64'),
            previous_mouse_x=all_actions[:-1, 2, :].astype('float32'),
            previous_mouse_y=all_actions[:-1, 3, :].astype('float32'),
        )
        actions = (
            all_actions[1:, 0, :].astype('int64'),
            all_actions[1:, 1, :].astype('int64'),
            all_actions[1:, 2, :].astype('float32'),
            all_actions[1:, 3, :].astype('float32'),
        )
        return padded_tensors, actions

    @classmethod
    def last_game_states_to_tensor(cls):
        tensors = list(chain(*(game.game_states[-1] for game in cls.games)))
        padded_tensors = pad_game_states(tensors)
        previous_actions = numpy.concatenate([game.game_actions[-1] for game in cls.games], axis=1)
        padded_tensors.update(
            previous_action_item=numpy.take_along_axis(
                padded_tensors['SidebarAssetName'],
                previous_actions[0, :].astype('int64')[:, None],
                axis=1,
            )[:, 0].astype('int64'),
            previous_action_type=previous_actions[1, :].astype('int64'),
            previous_mouse_x=previous_actions[2, :].astype('float32'),
            previous_mouse_y=previous_actions[3, :].astype('float32'),
        )
        tensors_with_one_time_dimension = dictmap(padded_tensors, lambda t: t.reshape(1, *t.shape))
        return tensors_with_one_time_dimension

    @classmethod
    def split_per_games(cls, l):
        result = []
        l = iter(l)
        for game in cls.games:
            result.append([next(l) for _ in range(len(game.players))])
        return result

    def get_rewards(self):
        rewards = [
            0 if self.loser_mask == 0 else (-1 if self.loser_mask & (1 << player) else 1)
            for player in range(len(self.players))
        ]
        if GameHandler.agents == 'AIvAI':
            rewards = numpy.ones_like(rewards)
        return rewards

    @classmethod
    def train(cls, n=1, time_window=200):
        game_state_tensor, actions = cls.all_game_states_and_actions()
        rewards = numpy.concatenate([game.get_rewards() for game in cls.games])
        cls.nn_agent.init_optimizer()  # hyperparameters can come here
        cls.nn_agent.learn(game_state_tensor, actions, rewards, n=n, time_window=time_window)

    @classmethod
    def configure(
        cls, nn_agent, players, start_game_args, n_games=2, end_limit=10_000, agents='NNvNN'
    ):
        cls.n_games = n_games
        cls.nn_agent = nn_agent
        cls.end_limit = end_limit
        cls.tqdm = trange(end_limit)
        cls.players = list(players)
        cls.start_game_args = start_game_args
        cls.games = []
        cls.ended_games = []
        cls.simple_agent = SimpleAgent()
        cls.agents = agents


def main():
    args = get_args()

    if args.load:
        agent = NNAgent.load(args.load)
    else:
        agent = NNAgent()  # hyperparameters can come here
    agent.to(args.device)

    GameHandler.configure(
        agent,
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
        start_game_args=cnc_structs.StartGameArgs(
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
        ),
        n_games=args.n,
        end_limit=args.end_limit,
        agents=args.agents,
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
        print(game.folder, game.compute_scores(), game.get_rewards())
        if args.print:
            for i in range(len(game.players)):
                print(i)
                game.print_what_player_sees(i)
    if args.record:
        for game in GameHandler.games:
            game.save_gameplay()
    if args.train > 0:
        GameHandler.train(args.train, args.time_window)
    if args.save:
        agent.save(args.save)


if __name__ == '__main__':
    main()
