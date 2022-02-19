import sys
import ctypes

import tornado.web
import tornado.websocket
import tornado.ioloop

import cnc_structs


class GameHandler(tornado.websocket.WebSocketHandler):
    games = []

    def on_message(self, message):
        if message == b'READY\0':
            if len(sys.argv) > 3:
                # change directory
                buffer = bytes(ctypes.c_uint32(0)) + sys.argv[3].encode('utf8') + b'\0'
                self.write_message(buffer, binary=True)

            # init dll
            buffer = bytes(ctypes.c_uint32(1))
            buffer += sys.argv[1].encode('utf8') + b'\0'  # dll name
            buffer += sys.argv[2].encode('ascii') + b'\0'  # content directory argument
            self.write_message(buffer, binary=True)

            # add players
            buffer = bytes(ctypes.c_uint32(2))
            buffer += bytes(
                cnc_structs.CNCPlayerInfoStruct(
                    GlyphxPlayerID=314159265,
                    Name=b"gaebor",
                    House=0,
                    Team=0,
                    AllyFlags=0,
                    ColorIndex=0,
                    IsAI=False,
                    StartLocationIndex=127,
                )
            )
            self.write_message(buffer, binary=True)
            buffer = bytes(ctypes.c_uint32(2))
            buffer += bytes(
                cnc_structs.CNCPlayerInfoStruct(
                    GlyphxPlayerID=271828182,
                    Name=b"ai1",
                    House=1,
                    Team=1,
                    AllyFlags=0,
                    ColorIndex=2,
                    IsAI=True,
                    StartLocationIndex=127,
                )
            )
            self.write_message(buffer, binary=True)

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
            print(
                "Game has ended.",
                *(f"Player{i} lost." for i in range(2) if ((1 << i) & loser_mask)),
            )
        else:
            print(len(message))
            buffer = b''
            for i in range(2):  # players
                buffer += bytes(ctypes.c_uint32(7))  # nought
                buffer += bytes(cnc_structs.NoughtRequestArgs(player_id=i))
            self.write_message(buffer, binary=True)

    def open(self):
        self.games.append(self)
        self.set_nodelay(True)

    def on_close(self):
        print('Bye', self.games.index(self))


def main():
    port = 8889
    application = tornado.web.Application([(r"/", GameHandler)])
    application.listen(port)
    tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    main()
