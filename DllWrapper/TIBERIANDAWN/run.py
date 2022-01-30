import sys

import ctypes
from multiprocessing import Pipe

import cnc_structs


def TD_process(
    connection,
    wrapper_dll,
    game_dll,
    players,
    multiplayer_options,
    scenario_index,
    build_level=7,
    content_dir=b'-CDDATA\\CNCDATA\\TIBERIAN_DAWN\\CD1',
):
    TD = ctypes.WinDLL(wrapper_dll)
    TD.Init.restype = ctypes.c_bool
    TD.ChDir.restype = ctypes.c_bool
    TD.StartGame.restype = ctypes.c_bool
    TD.Advance.restype = ctypes.c_bool
    TD.GetGameResult.restype = ctypes.c_ubyte

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

    while TD.Advance():
        x = TD.GetGameResult()
        continue
        # returns a list of per-player representation of the game from the player's POV
        vectorized_game_states = TD.get_what_players_see()
        connection.send({'state': vectorized_game_states})
        # caller computes the appropriate reactions
        actions = connection.recv()  # recieves actions of all players, possibly None
        if actions is None:
            # caller decided to stop game
            # return who was winning
            connection.send({'stop': TD.GetGameResult()})
    connection.send({'finish': TD.GetGameResult()})
    TD.FreeDll()


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
)
