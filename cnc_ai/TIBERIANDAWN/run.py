import argparse
from os import spawnl, P_NOWAIT, mkdir
import ctypes
from random import choice
from itertools import chain
from datetime import datetime
import pickle

from torch import no_grad
import numpy

import tornado.web
import tornado.websocket
import tornado.ioloop

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
            self.per_player_game_state = []
            offset = 0
            for _ in range(len(self.players)):
                self.per_player_game_state.append(cnc_structs.convert_to_np(message[offset:]))
                offset += cnc_structs.get_game_state_size(message[offset:])

            self.n_messages += 1

            if self.n_messages >= GameHandler.end_limit:
                self.close()
                return

            # sync games
            if (GameHandler.n_games == len(GameHandler.games)) and all(
                game.n_messages == self.n_messages for game in GameHandler.games
            ):
                # have to keep track of internal game state
                game_state_tensor = pad_game_states(
                    GameHandler.chain_game_states(), GameHandler.device
                )
                action_parameters = GameHandler.nn(**game_state_tensor)
                actions = GameHandler.nn.actions.sample(*action_parameters)
                log_prob = GameHandler.nn.actions.evaluate(*action_parameters, *actions)
                for game, per_game_actions in zip(
                    GameHandler.games, GameHandler.split_per_games(zip(*actions))
                ):
                    numpy.save(game.messages, game.per_player_game_state)
                    numpy.save(game.messages, per_game_actions)
                    message = b''
                    for player, action in zip(game.players, per_game_actions):
                        message += render_action(player.GlyphxPlayerID, *action)
                    game.write_message(message, binary=True)

    def open(self):
        GameHandler.games.append(self)
        self.loser_mask = 0
        self.n_messages = 0
        self.set_nodelay(True)
        self.init_game()
        self.start_recording()

    def on_close(self):
        GameHandler.ended_games.append(self)
        self.messages.close()
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
        game_state = self.per_player_game_state[player]
        print(cnc_structs.render_game_state_terminal(game_state))

    def assess_players_performance(self):
        scores = []
        if len(self.per_player_game_state) > 0:
            for player, game_state in zip(self.players, self.per_player_game_state):
                scores.append(cnc_structs.score(game_state, player.ColorIndex))
            loser_mask = sum(1 << i for i in range(len(self.players)) if scores[i] < max(scores))
        else:
            print(self.messages, self)
            loser_mask = 0
        return loser_mask

    def end_game(self):
        if self.loser_mask == 0:
            self.loser_mask = self.assess_players_performance()
        with open(self.folder + '/loser_mask.txt', 'wt') as f:
            print(self.loser_mask, file=f)
        print(f'game: {id(self)}, length: {self.n_messages}, loser_mask: {self.loser_mask}')
        if self.loser_mask > 0:
            for i in range(len(self.players)):
                print(f"player {i}:")
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
            cls.games = []
            cls.ended_games = []
            tornado.ioloop.IOLoop.current().stop()

    def start_recording(self):
        self.folder = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%S.%fs") + '_' + str(id(self))
        mkdir(self.folder)
        with open(self.folder + '/players.pkl', 'wb') as f:
            pickle.dump(self.players, f)
        self.messages = open(self.folder + '/messages.npy', 'wb')

    @classmethod
    def chain_game_states(cls):
        return list(chain(*(game.per_player_game_state for game in cls.games)))

    @classmethod
    def split_per_games(cls, l):
        result = []
        l = iter(l)
        for game in cls.games:
            result.append([next(l) for _ in range(len(game.players))])
        return result


def main():
    args = get_args()

    GameHandler.device = args.device
    GameHandler.n_games = args.n
    GameHandler.nn = TD_GamePlay().to(GameHandler.device)
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
    tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    with no_grad():
        main()
