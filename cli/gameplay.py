import ctypes

from PIL import Image

import cnc_structs
import decoders
import input_requests


class TDGameplay:
    def __init__(self, dll_path, content_directory):
        self.players = []
        self.actions = {}
        self.content_directory = content_directory
        self.dll = ctypes.WinDLL(dll_path)
        self.dll.CNC_Init(ctypes.c_char_p(b"-CD\"" + content_directory + b"\""), None)
        self.dll.CNC_Config(ctypes.byref(cnc_structs.get_diff()))

        self.dll.CNC_Advance_Instance.restype = ctypes.c_bool
        self.dll.CNC_Set_Multiplayer_Data.restype = ctypes.c_bool
        self.dll.CNC_Start_Instance.restype = ctypes.c_bool
        self.dll.CNC_Start_Instance_Variation.restype = ctypes.c_bool
        self.dll.CNC_Start_Custom_Instance.restype = ctypes.c_bool
        self.dll.CNC_Get_Start_Game_Info.restype = ctypes.c_bool
        self.dll.CNC_Get_Palette.restype = ctypes.c_bool
        self.dll.CNC_Get_Visible_Page.restype = ctypes.c_bool
        self.dll.CNC_Clear_Object_Selection.restype = ctypes.c_bool
        self.dll.CNC_Select_Object.restype = ctypes.c_bool
        self.dll.CNC_Get_Game_State.restype = ctypes.c_bool
        self.dll.CNC_Save_Load.restype = ctypes.c_bool

        self.game_state_buffer = (ctypes.c_uint8 * (4 * 1024 ** 2))()
        self.image_buffer = (ctypes.c_uint8 * ((64 * 24) ** 2))()

    def add_player(self, playerinfo: cnc_structs.CNCPlayerInfoStruct):
        self.players.append(playerinfo)

    def start_game(
        self,
        multiplayer_options: cnc_structs.CNCMultiplayerOptionsStruct,
        scenario_index,
        difficulty=0,
    ):
        self.players = (cnc_structs.CNCPlayerInfoStruct * len(self.players))(*self.players)
        if False == self.dll.CNC_Set_Multiplayer_Data(
            ctypes.c_int(scenario_index),
            ctypes.byref(multiplayer_options),
            ctypes.c_int(len(self.players)),
            self.players,
            ctypes.c_int(6),  # max number of players
        ):
            raise ValueError('CNC_Set_Multiplayer_Data')

        if False == self.dll.CNC_Start_Instance_Variation(
            ctypes.c_int(scenario_index),
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

        self.dll.CNC_Set_Difficulty(ctypes.c_int(difficulty))

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
        self.static_map = self.get_game_state('GAME_STATE_STATIC_MAP', 0)

    def load(self, filename):
        if False == self.dll.CNC_Save_Load(
            ctypes.c_bool(False),
            ctypes.c_char_p(filename.encode('ascii')),
            ctypes.c_char_p(b'GAME_GLYPHX_MULTIPLAYER'),
        ):
            raise ValueError('CNC_Save_Load')

        self.players = (cnc_structs.CNCPlayerInfoStruct * len(self.players))(*self.players)

        self.dll.CNC_Handle_Game_Request(ctypes.c_int(1))  # INPUT_GAME_LOADING_DONE
        self.retrieve_players_info()

        self.init_palette()
        self.static_map = self.get_game_state('GAME_STATE_STATIC_MAP', 0)

    def save(self, filename):
        if False == self.dll.CNC_Save_Load(
            ctypes.c_bool(True),
            ctypes.c_char_p(filename.encode('ascii')),
            ctypes.c_char_p(b'GAME_GLYPHX_MULTIPLAYER'),
        ):
            raise ValueError('CNC_Save_Load')
        else:
            return

    def retrieve_players_info(self):
        # overrides House info: GOOD/BAD becomes MULTI1-6
        for player in self.players:
            if not self.dll.CNC_Get_Game_State(
                ctypes.c_int(8),  # GAME_STATE_PLAYER_INFO
                ctypes.c_uint64(player.GlyphxPlayerID),
                ctypes.pointer(player),
                ctypes.sizeof(cnc_structs.CNCPlayerInfoStruct)
                + 33,  # A little extra for no reason
            ):
                raise ValueError('CNC_Get_Game_State (PLAYER_INFO)')

    def get_game_state(self, state_request, player_index):
        player = self.players[player_index]
        request_type, result_type = cnc_structs.GameStateRequestEnum[state_request]
        if self.dll.CNC_Get_Game_State(
            ctypes.c_int(request_type),
            ctypes.c_uint64(player.GlyphxPlayerID),
            ctypes.pointer(self.game_state_buffer),
            ctypes.c_int(len(self.game_state_buffer)),
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

    def get_units(self, player_index):
        player = self.players[player_index]
        units = decoders.players_units(self.get_game_state('GAME_STATE_LAYERS', 0), player.House)
        return units

    def advance(self, count=1):
        for player_id, (action, args) in self.actions.items():
            self.handle_request(action, player_id, *args)
        self.actions = {}
        result = self.dll.CNC_Advance_Instance(ctypes.c_uint64(0))
        for _ in range(1, count):
            result = self.dll.CNC_Advance_Instance(ctypes.c_uint64(0))
        return result

    def register_request(self, player_index, request_type, arg1, *args):
        player = self.players[player_index]
        self.actions[player.GlyphxPlayerID] = (request_type, (arg1,) + args)

    def handle_request(self, request_type, player_id, x1, y1=0, x2=0, y2=0):
        if request_type == 'INPUT_REQUEST_SPECIAL_KEYS':
            self.dll.CNC_Handle_Input(
                ctypes.c_int(10),
                ctypes.c_ubyte(x1),
                ctypes.c_uint64(player_id),
                ctypes.c_int(0),
                ctypes.c_int(0),
                ctypes.c_int(0),
                ctypes.c_int(0),
            )
        elif request_type.startswith('INPUT_REQUEST'):
            self.dll.CNC_Handle_Input(
                ctypes.c_int(input_requests.InputRequestEnum[request_type]),
                ctypes.c_ubyte(0),
                ctypes.c_uint64(player_id),
                ctypes.c_int(x1),
                ctypes.c_int(y1),
                ctypes.c_int(x2),
                ctypes.c_int(y2),
            )
        elif request_type == 'SUPERWEAPON_REQUEST_PLACE_SUPER_WEAPON':
            self.dll.CNC_Handle_SuperWeapon_Request(
                ctypes.c_int(0),
                ctypes.c_uint64(player_id),
                ctypes.c_int(x1),
                ctypes.c_int(y1),
                ctypes.c_int(x2),
                ctypes.c_int(y2),
            )
        elif request_type.startswith('INPUT_STRUCTURE'):
            self.dll.CNC_Handle_Structure_Request(
                ctypes.c_int(input_requests.StructureRequestEnum[request_type]),
                ctypes.c_uint64(player_id),
                ctypes.c_int(x1),
            )
        elif request_type.startswith('INPUT_UNIT'):
            self.dll.CNC_Handle_Unit_Request(
                ctypes.c_int(input_requests.UnitRequestEnum[request_type]),
                ctypes.c_uint64(player_id),
            )
        elif request_type.startswith('SIDEBAR'):
            self.dll.CNC_Handle_Sidebar_Request(
                ctypes.c_int(input_requests.SidebarRequestEnum[request_type]),
                ctypes.c_uint64(player_id),
                ctypes.c_int(x1),
                ctypes.c_int(y1),
                ctypes.c_short(x2),
                ctypes.c_short(y2),
            )
        elif request_type.startswith('CONTROL_GROUP_REQUEST'):
            self.dll.CNC_Handle_ControlGroup_Request(
                ctypes.c_int(input_requests.ControlGroupRequestEnum[request_type]),
                ctypes.c_uint64(player_id),
                ctypes.c_ubyte(x1),
            )
        elif request_type.startswith('INPUT_BEACON_'):
            self.dll.CNC_Handle_Beacon_Request(
                ctypes.c_int(input_requests.BeaconRequestEnum[request_type]),
                ctypes.c_uint64(player_id),
                ctypes.c_int(x1),
                ctypes.c_int(y1),
            )
        else:
            raise ValueError(request_type)

    def get_what_player_see(self, player_index):
        player = self.players[player_index]
        dynamic_map = self.get_game_state('GAME_STATE_DYNAMIC_MAP', 0)
        layers = self.get_game_state('GAME_STATE_LAYERS', player_index)
        shroud = decoders.shroud_array(
            self.get_game_state('GAME_STATE_SHROUD', player_index),
            (self.static_map.MapCellHeight, self.static_map.MapCellWidth),
        )
        occupiers = self.get_game_state('GAME_STATE_OCCUPIER', 0)
        fixed_pos_map_assets, fixed_pos_map_shape, actors = decoders.f(
            dynamic_map,
            layers,
            occupiers,
            shroud,
            (self.static_map.MapCellHeight, self.static_map.MapCellWidth),
            player.House,
            player.AllyFlags,
        )
        return fixed_pos_map_assets, fixed_pos_map_shape, actors

    def __del__(self):
        ctypes.windll.kernel32.FreeLibrary(self.dll._handle)
