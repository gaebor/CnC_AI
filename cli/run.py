import ctypes
import argparse
import os
from random import getrandbits, sample


import cnc_structs
from gameplay import TDGameplay
from decoders import layers_term


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dll_path')
    parser.add_argument(
        '--dir',
        default="C:\\Program Files (x86)\\Steam\\steamapps\\common\\CnCRemastered",
        help='Path to the CnC installation directory.',
    )

    return parser.parse_args()


def main(args):
    os.chdir(args.dir)

    TD = TDGameplay(args.dll_path, b'DATA\\CNCDATA\\TIBERIAN_DAWN\\CD1')

    houses = getrandbits(6)
    houses = tuple((houses >> i) % 2 for i in range(6))
    colors = sample([0, 1, 2, 3, 4, 5], k=6)

    TD.add_player(
        cnc_structs.CNCPlayerInfoStruct(
            GlyphxPlayerID=314159265,
            Name=b"gaebor",
            House=houses[0],
            Team=0,
            AllyFlags=0,
            ColorIndex=colors[0],
            IsAI=False,
            StartLocationIndex=127,
        )
    )
    TD.add_player(
        cnc_structs.CNCPlayerInfoStruct(
            GlyphxPlayerID=271828182,
            Name=b"ai1",
            House=houses[1],
            Team=1,
            AllyFlags=0,
            ColorIndex=colors[1],
            IsAI=True,
            StartLocationIndex=127,
        )
    )

    multiplayer_options = cnc_structs.CNCMultiplayerOptionsStruct(
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
    )

    TD.start_game(multiplayer_options, ctypes.c_int(50))

    static_map = TD.get_game_state('GAME_STATE_STATIC_MAP', 0)

    units = None
    selected = None

    frame = 1
    while TD.advance():
        print(frame, end='\r')
        if units is None:
            units = TD.get_units(0)
        elif selected is None:
            selected = units[0]
            TD.register_request(
                0, 'INPUT_REQUEST_SELECT_AT_POSITION', selected.PositionX, selected.PositionY
            )
        elif selected is not None:
            TD.register_request(
                0, 'INPUT_REQUEST_COMMAND_AT_POSITION', selected.PositionX, selected.PositionY
            )

        if frame % 1000 == 0:
            dynamic_map = TD.get_game_state('GAME_STATE_DYNAMIC_MAP', 0)
            layers = TD.get_game_state('GAME_STATE_LAYERS', 0)
            print(layers_term(layers, dynamic_map, static_map))
            # occupiers = TD.get_game_state('GAME_STATE_OCCUPIER', 0)
            sidebar = TD.get_game_state('GAME_STATE_SIDEBAR', 0)
            # print(sidebar)
            # placement = TD.get_game_state('GAME_STATE_PLACEMENT', 0)
            # print(placement)
            # TD.show_image()
        frame += 1
    print()
    TD.retrieve_players_info()
    for player in TD.players:
        print(player)


if __name__ == '__main__':
    main(get_args())
