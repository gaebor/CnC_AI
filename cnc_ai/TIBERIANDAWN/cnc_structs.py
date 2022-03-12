import ctypes
import termcolor
from pathlib import Path

import numpy


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

with open(Path(__file__).parent / 'all_assets_with_shapes.txt', 'rt') as f:
    all_asset_num_shapes = dict(
        map(lambda x: (x[1].upper(), int(x[0])), (line.strip().split() for line in f))
    )


def calculate_asset_num_shapes(names_list):
    names_dict = {v: i for i, v in enumerate(names_list)}
    asset_num_shapes = {
        names_dict[k]: v for k, v in all_asset_num_shapes.items() if k in static_tile_names
    }
    return asset_num_shapes


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
    map_list = [
        [decode_cell(game_state['StaticAssetName'][i][j]) for j in range(MAP_MAX_WIDTH)]
        for i in range(MAP_MAX_HEIGHT)
    ]
    for AssetName, Owner, PositionX, PositionY in zip(
        game_state['AssetName'],
        game_state['Owner'],
        game_state['Continuous'][:, 0],
        game_state['Continuous'][:, 1],
    ):
        x, y = int(PositionY) // 24, int(PositionX) // 24
        color = 'white'
        if Owner != 255:
            color = ['yellow', 'blue', 'red', 'white', 'magenta', 'cyan'][Owner]
        map_list[x][y] = (
            '{:2s}'.format(dynamic_object_names[AssetName][:2]),
            color,
            map_list[x][y][2],
        )
    return '\n'.join(map(lambda row: ''.join(termcolor.colored(*cell) for cell in row), map_list))


costs = {
    'AFLD': 2000,
    'AFLDMAKE': 2000,
    'APC': 700,
    'ARTY': 450,
    'ATWR': 1000,
    'ATWRMAKE': 1000,
    'BGGY': 300,
    'BIKE': 500,
    'BOAT': 300,
    'BRIK': 100,
    'CYCL': 75,
    'E1': 100,
    'E2': 160,
    'E3': 300,
    'E4': 200,
    'E5': 300,
    'E6': 500,
    'EYE': 2800,
    'EYEMAKE': 2800,
    'FACT': 5001,
    'FACTMAKE': 5001,
    'FIX': 1200,
    'FIXMAKE': 1200,
    'FTNK': 800,
    'GTWR': 500,
    'GTWRMAKE': 500,
    'GUN': 600,
    'GUNMAKE': 600,
    'HAND': 300,
    'HANDMAKE': 300,
    'HARV': 1400,
    'HELI': 1200,
    'HPAD': 1500,
    'HPADMAKE': 1500,
    'HQ': 1000,
    'HQMAKE': 1000,
    'HTNK': 1500,
    'JEEP': 400,
    'LST': 300,
    'LTNK': 600,
    'MCV': 5000,
    'MHQ': 600,
    'MLRS': 750,
    'MSAM': 800,
    'MTNK': 800,
    'NUK2': 700,
    'NUK2MAKE': 700,
    'NUKE': 300,
    'NUKEMAKE': 300,
    'OBLI': 1500,
    'OBLIMAKE': 1500,
    'ORCA': 1200,
    'PROC': 2000,
    'PROCMAKE': 2000,
    'PYLE': 300,
    'PYLEMAKE': 300,
    'RMBO': 1000,
    'SAM': 750,
    'SAMMAKE': 750,
    'SBAG': 50,
    'SILO': 150,
    'SILOMAKE': 150,
    'STNK': 900,
    'TMPL': 3000,
    'TMPLMAKE': 3000,
    'TRAN': 1500,
    'WEAP': 2000,
    'WEAPMAKE': 2000,
}


def score(game_state_arrays, owner):
    result = 0
    for dynamic_object_asset_index, dynamic_object_owner in zip(
        game_state_arrays['AssetName'], game_state_arrays['Owner']
    ):
        if owner == dynamic_object_owner:
            result += costs.get(dynamic_object_names[dynamic_object_asset_index], 0)

    for sidebar_asset_index, sidebar_progress in zip(
        game_state_arrays['SidebarAssetName'], game_state_arrays['SidebarContinuous'][:, 0]
    ):
        result += costs.get(dynamic_object_names[sidebar_asset_index], 0) * sidebar_progress

    return result


def read_array_from_buffer(buffer, offset, Type):
    count = ctypes.c_uint32.from_buffer_copy(buffer, offset).value
    offset += ctypes.sizeof(ctypes.c_uint32)
    number_of_members = ctypes.sizeof(Type) // 4
    indices = numpy.frombuffer(
        buffer, dtype='int32', count=count * number_of_members, offset=offset
    ).reshape(count, number_of_members)
    continuous = numpy.frombuffer(
        buffer, dtype='float32', count=count * number_of_members, offset=offset
    ).reshape(count, number_of_members)
    offset += count * ctypes.sizeof(Type)
    return offset, indices, continuous


def convert_to_np(game_state_buffer):
    offset = 8 * ctypes.sizeof(ctypes.c_int)
    map_cells = numpy.frombuffer(
        game_state_buffer, dtype='int32', count=62 * 62 * 2, offset=offset
    ).reshape((62, 62, 2))
    offset = ctypes.sizeof(StaticMap)

    offset, dynamic_objects_indices, dynamic_objects_continuous = read_array_from_buffer(
        game_state_buffer, offset, DynamicObject
    )

    sidebar_members = numpy.frombuffer(
        game_state_buffer,
        dtype='float32',
        count=ctypes.sizeof(SideBarMembers) // 4,
        offset=offset,
    )
    offset += ctypes.sizeof(SideBarMembers)

    offset, sidebar_entries_indices, sidebar_entries_continuous = read_array_from_buffer(
        game_state_buffer, offset, SidebarEntry
    )

    return {
        'StaticAssetName': map_cells[:, :, 0],
        'StaticShapeIndex': map_cells[:, :, 1],
        'AssetName': dynamic_objects_indices[:, 0],
        'ShapeIndex': dynamic_objects_indices[:, 1],
        'Owner': dynamic_objects_indices[:, 2],
        'Pips': dynamic_objects_indices[:, 3 : 3 + MAX_OBJECT_PIPS],
        'ControlGroup': dynamic_objects_indices[:, 3 + MAX_OBJECT_PIPS],
        'Cloak': dynamic_objects_indices[:, 3 + MAX_OBJECT_PIPS + 1],
        'Continuous': dynamic_objects_continuous[:, -5:],
        'SidebarInfos': sidebar_members,
        'SidebarAssetName': sidebar_entries_indices[:, 0],
        'SidebarContinuous': sidebar_entries_continuous[:, 3:],
    }