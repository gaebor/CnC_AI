import argparse
from os import spawnl, P_NOWAIT
import ctypes
from random import choice
from itertools import chain

from torch import no_grad
import numpy

import tornado.web
import tornado.websocket
import tornado.ioloop

from cnc_ai.TIBERIANDAWN import cnc_structs
from cnc_ai.TIBERIANDAWN.model import pad_game_states, TD_GamePlay


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
    nn = TD_GamePlay()
    device = 'cpu'
    n_games = 0
    ended_games = []

    def on_message(self, message):
        if len(message) == 1:
            self.loser_mask = ctypes.c_ubyte.from_buffer_copy(message).value
            self.close()
        else:
            # recieved the current game state
            per_player_game_state = []
            offset = 0
            for _ in range(len(self.players)):
                per_player_game_state.append(cnc_structs.convert_to_np(message[offset:]))
                offset += cnc_structs.get_game_state_size(message[offset:])

            self.messages.append(per_player_game_state)

            if len(self.messages) >= GameHandler.end_limit:
                self.close()
                return

            # sync games
            if (GameHandler.n_games == len(GameHandler.games)) and all(
                len(game.messages) == len(self.messages) for game in GameHandler.games
            ):
                # have to keep track of internal game state
                dynamic_mask, sidebar_mask, game_state_tensor = pad_game_states(
                    list(chain(*(game.messages[-1] for game in GameHandler.games))),
                    GameHandler.device,
                )
                actions = [
                    action.cpu().numpy()
                    for action in GameHandler.nn(dynamic_mask, sidebar_mask, **game_state_tensor)
                ]
                i = 0
                for game in GameHandler.games:
                    message = render_actions(i, len(game.players), *actions)
                    game.write_message(message, binary=True)
                    i += len(game.players)

    def open(self):
        GameHandler.games.append(self)
        self.loser_mask = 0
        self.messages = []
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

            buffer = bytes(ctypes.c_uint32(2))
            buffer += bytes(player)
            self.write_message(buffer, binary=True)
            self.players.append(player)

    def print_what_player_sees(self, player):
        game_state = self.messages[-1][player]
        print(cnc_structs.render_game_state_terminal(game_state))

    def assess_players_performance(self):
        scores = []
        if len(self.messages) > 0:
            for player, game_state in zip(self.players, self.messages[-1]):
                scores.append(cnc_structs.score(game_state, player.ColorIndex))
            loser_mask = sum(1 << i for i in range(len(self.players)) if scores[i] < max(scores))
        else:
            print(self.messages, self)
            loser_mask = 0
        return loser_mask

    def end_game(self):
        if self.loser_mask == 0:
            self.loser_mask = self.assess_players_performance()
        print(f'game: {id(self)}, length: {len(self.messages)}, loser_mask: {self.loser_mask}')
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


def render_actions(
    offset, n_players, main_action, sidebar_action, input_request_type, mouse_position
):
    buffer = b''
    for i, player_id in zip(range(offset, offset + n_players), range(n_players)):
        action_type = main_action[i].argmax()
        if action_type == 0:
            buffer += bytes(ctypes.c_uint32(7))  # NOUGHTREQUEST
            buffer += bytes(cnc_structs.NoughtRequestArgs(player_id=player_id))
        if action_type == 1:
            if sidebar_action.shape[1] > 0 and numpy.isnfinite(sidebar_action[i, 0]):
                possible_actions = sidebar_action[i]
                best_sidebar_element, best_action_type = numpy.unravel_index(
                    possible_actions.argmax(), possible_actions.shape
                )
                buffer += bytes(ctypes.c_uint32(6))  # SIDEBARREQUEST
                buffer += bytes(
                    cnc_structs.SidebarRequestArgs(
                        player_id=player_id,
                        requestType=best_action_type,
                        assetNameIndex=best_sidebar_element,
                    )
                )
        elif action_type == 2:
            buffer += bytes(ctypes.c_uint32(5))  # INPUTREQUEST
            request_type = input_request_type[i].argmax()
            buffer += bytes(
                cnc_structs.InputRequestArgs(
                    player_id=player_id,
                    requestType=request_type,
                    x1=mouse_position[i, request_type, 0],
                    y1=mouse_position[i, request_type, 1],
                )
            )
    return buffer


def encode_list(list_of_strings):
    return b''.join(map(lambda s: str.encode(s, encoding='ascii') + b'\0', list_of_strings))


def main():
    args = get_args()

    GameHandler.device = args.device
    GameHandler.n_games = args.n
    GameHandler.nn = GameHandler.nn.to(GameHandler.device)
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
