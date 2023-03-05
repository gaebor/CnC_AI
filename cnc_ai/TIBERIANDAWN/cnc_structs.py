import ctypes
from pathlib import Path
import json

import termcolor
import numpy

from cnc_ai.TIBERIANDAWN.bridge import GameState


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

    def render_message(self):
        buffer = bytes(ctypes.c_uint32(2))
        buffer += bytes(self)
        return buffer


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


class GameArgs(CncStruct):
    _fields_ = [('multiplayer_info', CNCMultiplayerOptionsStruct)]


class StartGameArgs(GameArgs):
    _fields_ = [
        ('scenario_index', ctypes.c_int),
        ('build_level', ctypes.c_int),
        ('difficulty', ctypes.c_int),
    ]

    def render_message(self) -> bytes:
        return bytes(ctypes.c_uint32(3)) + bytes(self)


class StartGameCustomArgs(GameArgs):
    _fields_ = [
        ('directory_path', ctypes.c_char * 256),
        ('scenario_name', ctypes.c_char * 256),
        ('build_level', ctypes.c_int),
    ]

    def render_message(self) -> bytes:
        return bytes(ctypes.c_uint32(4)) + bytes(self)


class LoadGameArgs:
    def __init__(self, filename: bytes):
        self.filename = filename

    def render_message(self) -> bytes:
        return bytes(ctypes.c_uint32(8)) + self.filename + b'\0'


class ActionRequestArgs(CncStruct):
    _fields_ = [
        ('player_id', ctypes.c_uint32),
        ('action_item', ctypes.c_uint32),
        ('action_type', ctypes.c_uint32),
        ('x', ctypes.c_float),
        ('y', ctypes.c_float),
    ]

    def render_message(self):
        return bytes(ctypes.c_uint32(5)) + bytes(self)


with open(Path(__file__).parent / 'static_tile_names.txt', 'rt') as f:
    static_tile_names = [''] + list(map(str.strip, f.readlines()))

with open(Path(__file__).parent / 'dynamic_object_names.txt', 'rt') as f:
    dynamic_object_names = [''] + list(map(str.strip, f.readlines()))

with open(Path(__file__).parent / 'all_assets_with_shapes.txt', 'rt') as f:
    all_asset_num_shapes = dict(
        map(lambda x: (x[1].upper(), int(x[0])), (line.strip().split() for line in f))
    )


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


with open(Path(__file__).parent / 'costs.json', 'rt') as f:
    costs = json.load(f)


def score(game_state_arrays: GameState, owner):
    result = 0
    for asset_index, shape_index, object_owner in zip(
        game_state_arrays.AssetName, game_state_arrays.ShapeIndex, game_state_arrays.Owner
    ):
        if owner == object_owner:
            asset_name = dynamic_object_names[asset_index]
            result += costs.get(f'{asset_name}_{shape_index}', costs.get(asset_name, 0))

    progress = game_state_arrays.SidebarContinuous[:, 0]
    cost = game_state_arrays.SidebarContinuous[:, 1]
    result += progress.dot(cost)

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


def convert_to_np(game_state_buffer) -> GameState:
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

    return GameState(
        StaticAssetName=map_cells[:, :, 0],
        StaticShapeIndex=map_cells[:, :, 1],
        AssetName=dynamic_objects_indices[:, 0],
        ShapeIndex=dynamic_objects_indices[:, 1],
        Owner=dynamic_objects_indices[:, 2],
        Pips=dynamic_objects_indices[:, 3 : 3 + MAX_OBJECT_PIPS],
        ControlGroup=dynamic_objects_indices[:, 3 + MAX_OBJECT_PIPS],
        Cloak=dynamic_objects_indices[:, 3 + MAX_OBJECT_PIPS + 1],
        Continuous=dynamic_objects_continuous[:, -5:],
        SidebarInfos=sidebar_members,
        SidebarAssetName=numpy.concatenate(
            [numpy.zeros(1, dtype=sidebar_entries_indices.dtype), sidebar_entries_indices[:, 0]]
        ),
        SidebarContinuous=numpy.concatenate(
            [
                numpy.zeros((1, 6), dtype=sidebar_entries_continuous.dtype),
                sidebar_entries_continuous[:, 3:],
            ],
            axis=0,
        ),
    )
