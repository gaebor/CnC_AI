import sys

import ctypes
from multiprocessing import Pipe

import cnc_structs


class COORD(ctypes.Structure):
    _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]


Stdhandle = ctypes.windll.kernel32.GetStdHandle(-11)

# from https://rosettacode.org/wiki/Terminal_control/Cursor_positioning#Python
def move_cursor(r, c):
    ctypes.windll.kernel32.SetConsoleCursorPosition(Stdhandle, COORD(c, r))


def TD_process(
    connection,
    wrapper_dll,
    game_dll,
    players,
    multiplayer_options,
    scenario_index,
    build_level=7,
    content_dir=b'-CDDATA\\CNCDATA\\TIBERIAN_DAWN\\CD1',
    working_dir=None,
):
    TD = ctypes.WinDLL(wrapper_dll)
    TD.Init.restype = ctypes.c_bool
    TD.ChDir.restype = ctypes.c_bool
    TD.StartGame.restype = ctypes.c_bool
    TD.Advance.restype = ctypes.c_bool
    TD.GetGameResult.restype = ctypes.c_ubyte

    if working_dir is not None:
        if False == TD.ChDir(ctypes.c_char_p(working_dir.encode('utf8'))):
            return

    if False == TD.Init(ctypes.c_char_p(game_dll.encode('utf8')), ctypes.c_char_p(content_dir)):
        return

    for player in players:
        if False == TD.AddPlayer(ctypes.byref(player)):
            return

    if False == TD.StartGame(
        ctypes.byref(multiplayer_options),
        ctypes.c_int(scenario_index),
        ctypes.c_int(build_level),
        ctypes.c_int(2),
    ):
        return

    players_buffer = (cnc_structs.PlayerVectorRepresentationView * len(players))()

    i = 0
    while TD.Advance():
        # if False == TD.GetCommonVectorRepresentation(ctypes.byref(buffer)):
        #     return
        if False == TD.GetPlayersVectorRepresentation(players_buffer):
            return
        if i % 10 == 0:
            if i > 0:
                move_cursor(0, 0)
            print(
                cnc_structs.print_game_state(
                    cnc_structs.VectorRepresentationView(
                        map=players_buffer[1].map,
                        dynamic_objects_count=players_buffer[1].dynamic_objects_count,
                        dynamic_objects=players_buffer[1].dynamic_objects,
                    )
                ),
                end='',
            )

        if i >= 5000:
            break
        i += 1
        continue
        # returns a list of per-player representation of the game from the player's POV
        vectorized_game_states = TD.get_what_players_see()
        connection.send({'state': vectorized_game_states})
        # caller computes the appropriate reactions
        actions = connection.recv()  # recieves actions of all players, possibly None
        if actions is None:
            # caller decided to stop game
            # return who was winning
            connection.send(TD.GetGameResult())
            return
        else:
            for player_id, action in enumerate(actions):
                if action is not None:
                    pass
                    # perform player's action
    print()
    print(TD.GetGameResult())


def main():
    parent_connection, child_connection = Pipe()
    TD_process(
        child_connection,
        sys.argv[1],
        sys.argv[2],
        [
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
        ],
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
        working_dir=None if len(sys.argv) <= 3 else sys.argv[3],
    )


if __name__ == '__main__':
    main()
