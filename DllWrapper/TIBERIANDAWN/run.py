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
            self.messages.append(message)
            loser_mask = ctypes.c_ubyte.from_buffer_copy(message).value
            print(
                f"Game {GameHandler.games.index(self)} has ended in {len(self.messages)} steps.",
                *(
                    f"Player {i} lost."
                    for i in range(len(GameHandler.players))
                    if ((1 << i) & loser_mask)
                ),
            )
        else:
            # recieved the current game state
            self.messages.append(message)
            if len(self.messages) > 10_000:
                self.close()
                print(f"Game {GameHandler.games.index(self)} was stopped.")
                return

            convert_to_np(message)
            convert_to_np(message[cnc_structs.get_game_state_size(message) :])

            # calculate reactions per player
            buffer = b''
            for i in range(len(GameHandler.players)):
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
        for player in GameHandler.players:
            player = cnc_structs.CNCPlayerInfoStruct.from_buffer_copy(bytes(player))
            if player.ColorIndex < 0:
                player.ColorIndex = choice(list(colors))
                colors -= {player.ColorIndex}
            if player.House not in [0, 1]:
                player.House = choice([0, 1])

            buffer = bytes(ctypes.c_uint32(2))
            buffer += bytes(player)
            self.write_message(buffer, binary=True)


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
            ColorIndex=-1,
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
            ColorIndex=-1,
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
