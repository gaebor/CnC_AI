import argparse
from os import spawnl, P_NOWAIT
import ctypes
from random import choice

import tornado.web
import tornado.websocket
import tornado.ioloop

import cnc_structs
from vectorization import convert_to_np


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--port', default=8888, type=int, help=' ')
    parser.add_argument(
        '-n', '--n', default=4, type=int, help='number of games to play simultaneously'
    )
    parser.add_argument('--exe', default='TIBERIANDAWN_wrapper.exe', help='absolute path')
    parser.add_argument(
        '--dir',
        default=r'C:\Program Files (x86)\Steam\steamapps\common\CnCRemastered',
        help='absolute path',
    )
    return parser.parse_args()


class GameHandler(tornado.websocket.WebSocketHandler):
    games = []
    ended_games = set()
    players = []
    chdir = '.'

    def on_message(self, message):
        if message == b'READY\0':
            # change to the directory where CnC is installed
            buffer = bytes(ctypes.c_uint32(0)) + GameHandler.chdir.encode('utf8') + b'\0'
            self.write_message(buffer, binary=True)

            # init dll
            buffer = bytes(ctypes.c_uint32(1))
            buffer += b'TiberianDawn.dll\0'
            buffer += b'-CDDATA\\CNCDATA\\TIBERIAN_DAWN\\CD1\0'
            self.write_message(buffer, binary=True)

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
                        SpawnVisceroids=True,
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
        elif len(message) == 1:
            loser_mask = ctypes.c_ubyte.from_buffer_copy(message).value
            self.end_game(loser_mask)
        else:
            # recieved the current game state
            self.messages.append(message)
            if len(self.messages) > 10_000:
                self.close()
                self.end_game(self.assess_players_performance())
                return

            # calculate reactions per player
            buffer = b''
            for i in range(len(self.players)):
                buffer += bytes(ctypes.c_uint32(7))  # nought
                buffer += bytes(cnc_structs.NoughtRequestArgs(player_id=i))
            # send responses
            self.write_message(buffer, binary=True)

    def open(self):
        GameHandler.games.append(self)
        self.messages = []
        self.set_nodelay(True)

    def on_close(self):
        GameHandler.ended_games.add(self)
        if set(GameHandler.games) == GameHandler.ended_games:
            GameHandler.games = []
            GameHandler.ended_games = set()
            tornado.ioloop.IOLoop.current().stop()

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

    def print_what_player_sees(self, winner_player):
        game_state = self.messages[-1]
        message_offset = 0
        for _ in range(winner_player):
            message_offset += cnc_structs.get_game_state_size(game_state[message_offset:])
        print(cnc_structs.render_game_state_terminal(game_state[message_offset:]))

    def assess_players_performance(self):
        offset = 0
        scores = []
        game_state = self.messages[-1]
        for player in self.players:
            scores.append(cnc_structs.score(convert_to_np(game_state[offset:]), player.ColorIndex))
            offset += cnc_structs.get_game_state_size(game_state[offset:])
        loser_mask = sum(1 << i for i in range(len(self.players)) if scores[i] < max(scores))
        return loser_mask

    def end_game(self, loser_mask):
        print(self)
        winner_player = [((1 << i) & loser_mask) > 0 for i in range(len(self.players))].index(
            False
        )
        self.print_what_player_sees(winner_player)


def main():
    args = get_args()
    application = tornado.web.Application([(r"/", GameHandler)])

    GameHandler.players.append(
        cnc_structs.CNCPlayerInfoStruct(
            GlyphxPlayerID=314159265,
            Name=b"gaebor",
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
            Name=b"ai1",
            House=127,
            Team=1,
            AllyFlags=0,
            ColorIndex=127,
            IsAI=True,
            StartLocationIndex=127,
        )
    )
    GameHandler.chdir = args.dir
    application.listen(args.port)
    for _ in range(args.n):
        spawnl(P_NOWAIT, args.exe, args.exe, str(args.port))
    tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    main()
