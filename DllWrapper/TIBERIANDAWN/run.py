import sys
import ctypes

import tornado.web
import tornado.websocket
import tornado.ioloop

import cnc_structs


class MainHandler(tornado.websocket.WebSocketHandler):
    def on_message(self, message):
        print(message)

    def open(self):
        self.set_nodelay(True)
        buffer = b''

        if len(sys.argv) > 3:
            buffer += bytes(ctypes.c_uint32(1))
            buffer += sys.argv[3].encode('utf8') + b'\0'

        buffer += bytes(ctypes.c_uint32(2))
        buffer += sys.argv[1].encode('utf8') + b'\0'
        buffer += sys.argv[2].encode('ascii') + b'\0'

        buffer += bytes(ctypes.c_uint32(3))
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
        buffer += bytes(ctypes.c_uint32(3))
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
        buffer += bytes(ctypes.c_uint32(4))
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

    def on_close(self):
        print('Bye!')


def main():
    port = 8889
    application = tornado.web.Application([(r"/", MainHandler)])
    application.listen(port)
    tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    main()
