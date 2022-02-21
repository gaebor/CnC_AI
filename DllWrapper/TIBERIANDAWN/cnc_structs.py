import ctypes
import termcolor
from pathlib import Path


class CncStruct(ctypes.Structure):
    _pack_ = 1

    def __repr__(self):
        self_repr = {}
        for field_name, _ in self._fields_:
            field = getattr(self, field_name)
            if isinstance(field, ctypes.Array):
                field = list(field)
            self_repr[field_name] = field
        self_repr.update(self.__dict__)
        for field in self_repr:
            if isinstance(self_repr[field], list):
                self_repr[field] = self_repr[field][:10]
        return repr(self_repr)


class CNCMultiplayerOptionsStruct(CncStruct):
    _fields_ = [
        ('MPlayerCount', ctypes.c_int),
        ('MPlayerBases', ctypes.c_int),
        ('MPlayerCredits', ctypes.c_int),
        ('MPlayerTiberium', ctypes.c_int),  # 0-9
        ('MPlayerGoodies', ctypes.c_int),
        ('MPlayerGhosts', ctypes.c_int),
        ('MPlayerSolo', ctypes.c_int),
        ('MPlayerUnitCount', ctypes.c_int),  # 0-9
        ('IsMCVDeploy', ctypes.c_bool),
        ('SpawnVisceroids', ctypes.c_bool),
        ('EnableSuperweapons', ctypes.c_bool),
        ('MPlayerShadowRegrow', ctypes.c_bool),
        ('MPlayerAftermathUnits', ctypes.c_bool),
        ('CaptureTheFlag', ctypes.c_bool),
        ('DestroyStructures', ctypes.c_bool),
        ('ModernBalance', ctypes.c_bool),
    ]


MAX_HOUSES = 32
MAX_EXPORT_CELLS = 128 * 128
MAP_MAX_WIDTH = 62
MAP_MAX_HEIGHT = MAP_MAX_WIDTH


class CNCSpiedInfoStruct(CncStruct):
    _fields_ = [('Power', ctypes.c_int), ('Drain', ctypes.c_int), ('Money', ctypes.c_int)]


class CNCPlayerInfoStruct(CncStruct):
    _fields_ = [
        ('Name', ctypes.c_char * 64),
        ('House', ctypes.c_uint8),
        ('ColorIndex', ctypes.c_int),
        ('GlyphxPlayerID', ctypes.c_uint64),
        ('Team', ctypes.c_int),
        ('StartLocationIndex', ctypes.c_int),
        ('HomeCellX', ctypes.c_uint8),
        ('HomeCellY', ctypes.c_uint8),
        ('IsAI', ctypes.c_bool),
        ('AllyFlags', ctypes.c_uint),
        ('IsDefeated', ctypes.c_bool),
        ('SpiedPowerFlags', ctypes.c_uint),
        ('SpiedMoneyFlags', ctypes.c_uint),
        ('SpiedInfo', CNCSpiedInfoStruct * MAX_HOUSES),
        ('SelectedID', ctypes.c_int),
        ('SelectedType', ctypes.c_int),
        ('ActionWithSelected', ctypes.c_uint8 * MAX_EXPORT_CELLS),
        ('ActionWithSelectedCount', ctypes.c_uint),
        ('ScreenShake', ctypes.c_uint),
        ('IsRadarJammed', ctypes.c_bool),
    ]


class StaticTile(CncStruct):
    _fields_ = [
        ('AssetName', ctypes.c_int32),
        ('ShapeIndex', ctypes.c_int32),
    ]


MAX_OBJECT_PIPS = 18


class DynamicObject(CncStruct):
    _fields_ = [
        ('AssetName', ctypes.c_int32),
        ('ShapeIndex', ctypes.c_int32),
        ('Owner', ctypes.c_int32),
        ('Pips', ctypes.c_int32 * MAX_OBJECT_PIPS),
        ('ControlGroup', ctypes.c_int32),
        ('Cloak', ctypes.c_int32),
        ('PositionX', ctypes.c_float),
        ('PositionY', ctypes.c_float),
        ('Strength', ctypes.c_float),
        ('IsSelected', ctypes.c_float),
        ('IsRepairing', ctypes.c_float),
    ]


class SidebarEntry(CncStruct):
    _fields_ = [
        ('AssetName', ctypes.c_int32),
        ('BuildableType', ctypes.c_int32),
        ('BuildableID', ctypes.c_int32),
        ('Progress', ctypes.c_float),
        ('Cost', ctypes.c_float),
        ('BuildTime', ctypes.c_float),
        ('Constructing', ctypes.c_float),
        ('ConstructionOnHold', ctypes.c_float),
        ('Busy', ctypes.c_float),
    ]


class SideBarMembers(CncStruct):
    _fields_ = [
        ('Credits', ctypes.c_float),
        ('PowerProduced', ctypes.c_float),
        ('PowerDrained', ctypes.c_float),
        ('RepairBtnEnabled', ctypes.c_float),
        ('SellBtnEnabled', ctypes.c_float),
        ('RadarMapActive', ctypes.c_float),
    ]


class StaticMap(CncStruct):
    _fields_ = [
        ('MapCellX', ctypes.c_int),
        ('MapCellY', ctypes.c_int),
        ('MapCellWidth', ctypes.c_int),
        ('MapCellHeight', ctypes.c_int),
        ('OriginalMapCellX', ctypes.c_int),
        ('OriginalMapCellY', ctypes.c_int),
        ('OriginalMapCellWidth', ctypes.c_int),
        ('OriginalMapCellHeight', ctypes.c_int),
        ('StaticCells', (StaticTile * MAP_MAX_HEIGHT) * MAP_MAX_WIDTH),
    ]


class StartGameArgs(CncStruct):
    _fields_ = [
        ('multiplayer_info', CNCMultiplayerOptionsStruct),
        ('scenario_index', ctypes.c_int),
        ('build_level', ctypes.c_int),
        ('difficulty', ctypes.c_int),
    ]


class StartGameCustomArgs(CncStruct):
    _fields_ = [
        ('multiplayer_info', CNCMultiplayerOptionsStruct),
        ('directory_path', ctypes.c_char * 256),
        ('scenario_name', ctypes.c_char * 256),
        ('build_level', ctypes.c_int),
    ]


class NoughtRequestArgs(CncStruct):
    _fields_ = [('player_id', ctypes.c_uint32)]


class SidebarRequestArgs(CncStruct):
    _fields_ = [
        ('player_id', ctypes.c_uint32),
        ('requestType', ctypes.c_int),
        ('assetNameIndex', ctypes.c_uint32),
    ]


class InputRequestArgs(CncStruct):
    _fields_ = [
        ('player_id', ctypes.c_uint32),
        ('requestType', ctypes.c_int),
        ('x1', ctypes.c_float),
        ('y1', ctypes.c_float),
    ]


with open(Path(__file__).parent / 'static_tile_names.txt', 'r') as f:
    static_tile_names = [''] + list(map(str.strip, f.readlines()))

with open(Path(__file__).parent / 'dynamic_object_names.txt', 'r') as f:
    dynamic_object_names = [''] + list(map(str.strip, f.readlines()))


def decode_cell(tile_name_index):
    text = static_tile_names[tile_name_index]
    if text.startswith('TI'):  # tiberium
        text = '  '
        background = 'on_green'
    elif text.startswith('CLEAR'):
        text = '  '
        background = 'on_grey'
    else:
        text = '{:2s}'.format(text[:2])
        background = 'on_grey'

    return (text, 'white', background)


color_map = ['yellow', 'blue', 'red', 'white', 'magenta', 'cyan']


def get_game_state_size(game_state_buffer):
    offset = ctypes.sizeof(StaticMap)
    dynamic_objects_count = ctypes.c_uint32.from_buffer_copy(game_state_buffer, offset).value
    offset += ctypes.sizeof(ctypes.c_uint32)
    offset += ctypes.sizeof(DynamicObject) * dynamic_objects_count + ctypes.sizeof(SideBarMembers)
    sidebar_count = ctypes.c_uint32.from_buffer_copy(game_state_buffer, offset).value
    offset += ctypes.sizeof(ctypes.c_uint32)
    offset += ctypes.sizeof(SidebarEntry) * sidebar_count
    return offset


def render_game_state_terminal(game_state):
    offset = 0
    cells = StaticMap.from_buffer_copy(game_state, offset).StaticCells
    offset += ctypes.sizeof(StaticMap)
    map_list = [
        [decode_cell(cells[i][j].AssetName) for j in range(MAP_MAX_WIDTH)]
        for i in range(MAP_MAX_HEIGHT)
    ]
    dynamic_objects_count = ctypes.c_uint32.from_buffer_copy(game_state, offset).value
    offset += ctypes.sizeof(ctypes.c_uint32)
    dynamic_objects = (DynamicObject * dynamic_objects_count).from_buffer_copy(game_state, offset)
    for i in range(dynamic_objects_count):
        thing = dynamic_objects[i]
        x, y = int(thing.PositionY) // 24, int(thing.PositionX) // 24
        color = 'white'
        if thing.Owner != 255:
            color = color_map[thing.Owner]
        map_list[x][y] = (
            '{:2s}'.format(dynamic_object_names[thing.AssetName][:2]),
            color,
            map_list[x][y][2],
        )
    return '\n'.join(map(lambda row: ''.join(termcolor.colored(*cell) for cell in row), map_list))
