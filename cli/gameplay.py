import ctypes

from PIL import Image

import cnc_structs
import decoders


class TDGameplay:
    def __init__(self, dll_path, content_directory):
        self.players = []
        self.content_directory = content_directory
        self.dll = ctypes.WinDLL(dll_path)
        self.dll.CNC_Init(ctypes.c_char_p(b"-CD\"" + content_directory + b"\""), None)
        self.dll.CNC_Config(ctypes.byref(cnc_structs.get_diff()))

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
        self.dll.CNC_Clear_Object_Selection.restype = ctypes.c_bool
        self.dll.CNC_Select_Object.restype = ctypes.c_bool
        self.CNC_Get_Game_State = ctypes.WINFUNCTYPE(
            ctypes.c_bool,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_uint,
        )(('CNC_Get_Game_State', self.dll))
        self.game_state_buffer = (ctypes.c_uint8 * (4 * 1024 ** 2))()
        self.image_buffer = (ctypes.c_uint8 * ((64 * 24) * (64 * 24)))()

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

        self.init_palette()

    def retrieve_players_info(self):
        # overrides House info: GOOD/BAD becomes MULTI1-6
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

    def init_palette(self):
        self.palette = (ctypes.c_uint8 * (256 * 3))()
        if self.dll.CNC_Get_Palette(self.palette):
            for i in range(len(self.palette)):
                self.palette[i] *= 4
        else:
            raise ValueError('CNC_Get_Palette')

    def show_image(self):
        width, height = ctypes.c_uint(0), ctypes.c_uint(0)
        if self.dll.CNC_Get_Visible_Page(
            self.image_buffer, ctypes.byref(width), ctypes.byref(height)
        ):
            img = Image.frombuffer('P', (width.value, height.value), self.image_buffer)
            img.putpalette(self.palette)
            img.show()

    def deploy_at_start(self, player_index):
        player = self.players[player_index]
        unit = decoders.players_units(self.get_game_state('GAME_STATE_LAYERS', 0), player.House)[0]
        self.dll.CNC_Handle_Input(
            ctypes.c_int(8),  # INPUT_REQUEST_SELECT_AT_POSITION
            ctypes.c_ubyte(0),  # special_key_flags
            ctypes.c_uint64(player.GlyphxPlayerID),
            ctypes.c_int(unit.PositionX),
            ctypes.c_int(unit.PositionY),
            ctypes.c_int(0),
            ctypes.c_int(0),
        )
        self.dll.CNC_Handle_Input(
            ctypes.c_int(9),  # INPUT_REQUEST_COMMAND_AT_POSITION
            ctypes.c_ubyte(0),  # special_key_flags
            ctypes.c_uint64(player.GlyphxPlayerID),
            ctypes.c_int(unit.PositionX),
            ctypes.c_int(unit.PositionY),
            ctypes.c_int(0),
            ctypes.c_int(0),
        )

    def __del__(self):
        ctypes.windll.kernel32.FreeLibrary(self.dll._handle)
