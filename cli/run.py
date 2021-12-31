import ctypes
import argparse
import os
from random import getrandbits, sample

import cnc_structs
import decoders


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
        self.players = []
        self.content_directory = content_directory
        self.dll = ctypes.WinDLL(dll_path)
        self.difficulties = cnc_structs.get_diff()
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
        self.CNC_Get_Game_State = ctypes.WINFUNCTYPE(
            ctypes.c_bool,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_uint,
        )(('CNC_Get_Game_State', self.dll))
        self.game_state_buffer = (ctypes.c_uint8 * (4 * 1024 ** 2))()

    def add_player(self, playerinfo: cnc_structs.CNCPlayerInfoStruct):
        self.players.append(playerinfo)

    def start_game(
        self, multiplayer_options: cnc_structs.CNCMultiplayerOptionsStruct, scenario_index
    ):
        self.players = (cnc_structs.CNCPlayerInfoStruct * len(self.players))(*self.players)
        if False == self.dll.CNC_Set_Multiplayer_Data(
            scenario_index,
            ctypes.byref(multiplayer_options),
            ctypes.c_int(len(self.players)),
            self.players,
            ctypes.c_int(6),  # max number of players
        ):
            raise ValueError('CNC_Set_Multiplayer_Data')

        if False == self.dll.CNC_Start_Instance_Variation(
            scenario_index,
            ctypes.c_int(-1),  # scenario_variation
            ctypes.c_int(0),  # scenario_direction
            ctypes.c_int(7),  # build_level
            ctypes.c_char_p(b"MULTI"),  # faction
            ctypes.c_char_p(b"GAME_GLYPHX_MULTIPLAYER"),  # game_type
            ctypes.c_char_p(self.content_directory),
            ctypes.c_int(-1),  # sabotaged_structure
            ctypes.c_char_p(b""),  # override_map_name
        ):
            raise ValueError('CNC_Start_Instance_Variation')

        self.dll.CNC_Set_Difficulty(ctypes.c_int(0))

        for player in self.players:
            StartLocationIndex = ctypes.c_int()
            if self.dll.CNC_Get_Start_Game_Info(
                ctypes.c_uint64(player.GlyphxPlayerID), ctypes.byref(StartLocationIndex)
            ):
                player.StartLocationIndex = StartLocationIndex.value
            else:
                raise ValueError('CNC_Get_Start_Game_Info')

        self.dll.CNC_Handle_Game_Request(ctypes.c_int(1))  # INPUT_GAME_LOADING_DONE
        self.retrieve_players_info()

    def retrieve_players_info(self):
        # overrides House info GOOD/BAD becomes MULTI1-6
        for player in self.players:
            if not self.CNC_Get_Game_State(
                cnc_structs.GameStateRequestEnum['GAME_STATE_PLAYER_INFO'][0],
                player.GlyphxPlayerID,
                ctypes.cast(ctypes.pointer(player), ctypes.POINTER(ctypes.c_ubyte)),
                ctypes.sizeof(cnc_structs.CNCPlayerInfoStruct)
                + 33,  # A little extra for no reason
            ):
                raise ValueError('CNC_Get_Game_State')

    def get_game_state(self, state_request, player_id):
        request_type, result_type = cnc_structs.GameStateRequestEnum[state_request]
        if self.CNC_Get_Game_State(
            request_type,
            player_id,
            self.game_state_buffer,
            len(self.game_state_buffer),
        ):
            return result_type(self.game_state_buffer)

    def __del__(self):
        ctypes.windll.kernel32.FreeLibrary(self.dll._handle)


def main(args):
    os.chdir(args.dir)

    TD = TDGameplay(args.dll_path, b'DATA\\CNCDATA\\TIBERIAN_DAWN\\CD1')

    houses = getrandbits(6)
    houses = tuple((houses >> i) % 2 for i in range(6))
    colors = sample([0, 1, 2, 3, 4, 5], k=6)

    TD.add_player(
        cnc_structs.CNCPlayerInfoStruct(
            GlyphxPlayerID=1055504538,
            Name=b"ai1",
            House=houses[1],
            Team=1,
            AllyFlags=0,
            ColorIndex=colors[1],
            IsAI=True,
            StartLocationIndex=127,
        )
    )
    TD.add_player(
        cnc_structs.CNCPlayerInfoStruct(
            GlyphxPlayerID=76561199154512029,
            Name=b"gaebor",
            House=houses[0],
            Team=0,
            AllyFlags=0,
            ColorIndex=colors[0],
            IsAI=False,
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

    width, height = ctypes.c_uint(0), ctypes.c_uint(0)
    image_buffer = (ctypes.c_uint8 * ((64 * 24) * (64 * 24)))()
    palette = (ctypes.c_uint8 * (256 * 3))()
    if TD.dll.CNC_Get_Palette(palette):
        for i in range(len(palette)):
            palette[i] *= 4
    else:
        raise ValueError('CNC_Get_Palette')

    map = TD.get_game_state('GAME_STATE_STATIC_MAP', 0)
    frame = 1
    while TD.Advance_Instance(ctypes.c_uint64(0)):
        print(frame, end='\r')
        if frame % 1000 == 0:
            dynamic_map = TD.get_game_state('GAME_STATE_DYNAMIC_MAP', 0)
            layers = TD.get_game_state('GAME_STATE_LAYERS', 0)
            TD.get_game_state('GAME_STATE_OCCUPIER', 0)

            # for player in players:
            #     TD.get_game_state('GAME_STATE_SIDEBAR', player.GlyphxPlayerID)
            #     TD.get_game_state('GAME_STATE_SHROUD', player.GlyphxPlayerID)
            #     TD.get_game_state('GAME_STATE_PLACEMENT', player.GlyphxPlayerID)

            if TD.dll.CNC_Get_Visible_Page(
                image_buffer, ctypes.byref(width), ctypes.byref(height)
            ):
                pass
                # img = Image.frombuffer('P', (width.value, height.value), image_buffer)
                # img.putpalette(palette)
                # img.show()
        frame += 1
    print()
    TD.retrieve_players_info()
    for player in TD.players:
        print(player)


if __name__ == '__main__':
    main(get_args())
