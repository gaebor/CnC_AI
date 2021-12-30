import ctypes
import argparse
import os
from random import getrandbits, sample

from cnc_structs import *


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dll_path')
    parser.add_argument(
        '--dir',
        default="C:\\Program Files (x86)\\Steam\\steamapps\\common\\CnCRemastered",
        help='Path to the CnC installation directory.',
    )

    return parser.parse_args()


class TDGameplay:
    def __init__(self, dll_path, content_directory):
        self.dll = ctypes.WinDLL(dll_path)
        self.difficulties = self.get_diff()
        self.dll.CNC_Init(ctypes.c_char_p(b"-CD\"" + content_directory + b"\""), None)
        self.dll.CNC_Config(ctypes.byref(self.difficulties))

        self.Advance_Instance = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint64)(
            ('CNC_Advance_Instance', self.dll)
        )
        self.dll.CNC_Set_Multiplayer_Data.restype = ctypes.c_bool
        self.dll.CNC_Start_Instance.restype = ctypes.c_bool
        self.dll.CNC_Start_Instance_Variation.restype = ctypes.c_bool
        self.dll.CNC_Start_Custom_Instance.restype = ctypes.c_bool
        self.dll.CNC_Get_Start_Game_Info.restype = ctypes.c_bool
        self.dll.CNC_Get_Palette.restype = ctypes.c_bool
        self.dll.CNC_Get_Visible_Page.restype = ctypes.c_bool
        self.dll.CNC_Get_Game_State.restype = ctypes.c_bool
        self.game_state_buffer = (ctypes.c_uint8 * 1024)()

    def get_game_state(self, player_id):
        while not self.dll.CNC_Get_Game_State(
            ctypes.c_int(8),
            ctypes.c_ulonglong(player_id),
            self.game_state_buffer,
            ctypes.c_uint(len(self.game_state_buffer)),
        ):
            self.game_state_buffer = (ctypes.c_uint8 * (len(self.game_state_buffer) * 2))()
        return CNCPlayerInfoStruct.from_buffer(self.game_state_buffer)

    def __del__(self):
        ctypes.windll.kernel32.FreeLibrary(self.dll._handle)

    @staticmethod
    def get_diff():
        diff = CNCRulesDataStruct()
        diff.Difficulties[0] = CNCDifficultyDataStruct(
            1.2, 1.2, 1.2, 0.3, 0.8, 0.8, 0.6, 0.001, 0.001, False, True, True
        )
        diff.Difficulties[1] = CNCDifficultyDataStruct(
            1, 1, 1, 1, 1, 1, 1, 0.02, 0.03, True, True, True
        )
        diff.Difficulties[2] = CNCDifficultyDataStruct(
            0.9, 0.9, 0.9, 1.05, 1.05, 1, 1, 0.05, 0.1, True, True, True
        )
        return diff


def main(args):
    os.chdir(args.dir)

    content_directory = b'DATA\\CNCDATA\\TIBERIAN_DAWN\\CD1'
    TD = TDGameplay(args.dll_path, content_directory)

    houses = getrandbits(6)
    houses = tuple((houses >> i) % 2 for i in range(6))
    colors = sample([0, 1, 2, 3, 4, 5], k=6)

    scenario_index = ctypes.c_int(50)

    players = [
        CNCPlayerInfoStruct(
            GlyphxPlayerID=76561199154512029,
            Name=b"gaebor",
            House=houses[0],
            Team=0,
            AllyFlags=0,
            ColorIndex=colors[0],
            IsAI=False,
            StartLocationIndex=127,
        ),
        CNCPlayerInfoStruct(
            GlyphxPlayerID=1055504538,
            Name=b"ai1",
            House=houses[1],
            Team=1,
            AllyFlags=0,
            ColorIndex=colors[1],
            IsAI=True,
            StartLocationIndex=127,
        ),
        # CNCPlayerInfoStruct(
        #     GlyphxPlayerID=76561199154512028,
        #     Name=b"other",
        #     House=houses[2],
        #     Team=2,
        #     AllyFlags=0,
        #     ColorIndex=colors[2],
        #     IsAI=False,
        #     StartLocationIndex=127,
        # ),
    ]
    players = (CNCPlayerInfoStruct * len(players))(*players)

    multiplayer_options = CNCMultiplayerOptionsStruct(
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
    if False == TD.dll.CNC_Set_Multiplayer_Data(
        scenario_index,
        ctypes.byref(multiplayer_options),
        ctypes.c_int(len(players)),
        players,
        ctypes.c_int(6),  # max number of players
    ):
        raise ValueError('CNC_Set_Multiplayer_Data')

    if False == TD.dll.CNC_Start_Instance_Variation(
        scenario_index,
        ctypes.c_int(-1),  # scenario_variation
        ctypes.c_int(0),  # scenario_direction
        ctypes.c_int(7),  # build_level
        ctypes.c_char_p(b"MULTI"),  # faction
        ctypes.c_char_p(b"GAME_GLYPHX_MULTIPLAYER"),  # game_type
        ctypes.c_char_p(content_directory),
        ctypes.c_int(-1),  # sabotaged_structure
        ctypes.c_char_p(b""),  # override_map_name
    ):
        raise ValueError('CNC_Start_Instance_Variation')

    TD.dll.CNC_Set_Difficulty(ctypes.c_int(0))

    for player in players:
        StartLocationIndex = ctypes.c_int()
        if TD.dll.CNC_Get_Start_Game_Info(
            ctypes.c_uint64(player.GlyphxPlayerID), ctypes.byref(StartLocationIndex)
        ):
            player.StartLocationIndex = StartLocationIndex.value
        else:
            raise ValueError('CNC_Get_Start_Game_Info')

    TD.dll.CNC_Handle_Game_Request(ctypes.c_int(1))  # INPUT_GAME_LOADING_DONE

    width, height = ctypes.c_uint(0), ctypes.c_uint(0)
    image_buffer = (ctypes.c_uint8 * ((64 * 24) * (64 * 24)))()
    palette = (ctypes.c_uint8 * (256 * 3))()
    if TD.dll.CNC_Get_Palette(palette):
        for i in range(len(palette)):
            palette[i] *= 4
    else:
        raise ValueError('CNC_Get_Palette')

    frame = 1
    while TD.Advance_Instance(ctypes.c_uint64(0)):
        print(frame, end='\r')
        if frame % 1000 == 0:
            if TD.dll.CNC_Get_Visible_Page(
                image_buffer, ctypes.byref(width), ctypes.byref(height)
            ):
                pass
                # img = Image.frombuffer('P', (width.value, height.value), image_buffer)
                # img.putpalette(palette)
                # img.show()
        frame += 1
    print()
    for player in players:
        print(TD.get_game_state(player.GlyphxPlayerID))


if __name__ == '__main__':
    main(get_args())
