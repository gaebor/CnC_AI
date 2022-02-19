import sys
import os
import ctypes

import tornado.web
import tornado.websocket
import tornado.ioloop

import cnc_structs


class GameHandler(tornado.websocket.WebSocketHandler):
    games = set()
    ended_games = set()
    players = []

    def on_message(self, message):
        if message == b'READY\0':
            if len(sys.argv) > 4:
                # change directory
                buffer = bytes(ctypes.c_uint32(0)) + sys.argv[4].encode('utf8') + b'\0'
                self.write_message(buffer, binary=True)

            # init dll
            buffer = bytes(ctypes.c_uint32(1))
            buffer += sys.argv[2].encode('utf8') + b'\0'  # dll name
            buffer += sys.argv[3].encode('ascii') + b'\0'  # content directory argument
            self.write_message(buffer, binary=True)

            # add players
            for p in self.players:
                buffer = bytes(ctypes.c_uint32(2))
                buffer += bytes(p)
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
            self.messages.append(message)
            loser_mask = ctypes.c_ubyte.from_buffer_copy(message).value
            print(
                f"Game has ended in {len(self.messages)} steps.",
                *(
                    f"Player {i} lost."
                    for i in range(len(self.players))
                    if ((1 << i) & loser_mask)
                ),
            )
        else:
            # recieved the current game state
            self.messages.append(message)
            buffer = b''
            # calculate reactions per player
            for i in range(2):
                buffer += bytes(ctypes.c_uint32(7))  # nought
                buffer += bytes(cnc_structs.NoughtRequestArgs(player_id=i))
            # send responses
            self.write_message(buffer, binary=True)

    def open(self):
        self.games.add(self)
        self.messages = []
        self.set_nodelay(True)

    def on_close(self):
        self.ended_games.add(self)
        if self.games == self.ended_games:
            tornado.ioloop.IOLoop.current().stop()


def main():
    port = 8889
    application = tornado.web.Application([(r"/", GameHandler)])

    GameHandler.players = [
        cnc_structs.CNCPlayerInfoStruct(
            GlyphxPlayerID=314159265,
            Name=b"gaebor",
            House=0,
            Team=0,
            AllyFlags=0,
            ColorIndex=0,
            IsAI=False,
            StartLocationIndex=127,
        ),
        cnc_structs.CNCPlayerInfoStruct(
            GlyphxPlayerID=271828182,
            Name=b"ai1",
            House=1,
            Team=1,
            AllyFlags=0,
            ColorIndex=2,
            IsAI=True,
            StartLocationIndex=127,
        ),
    ]
    application.listen(port)
    os.spawnl(os.P_NOWAIT, sys.argv[1], sys.argv[1], str(port))
    tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    main()

# python run.py C:\Users\gaebor\Documents\CnC_AI\DllWrapper\bin\Release\TIBERIANDAWN_wrapper.exe TiberianDawn.dll -CDDATA\CNCDATA\TIBERIAN_DAWN\CD1 Z:\Játék\BaeborSteam\steamapps\common\CnCRemastered
