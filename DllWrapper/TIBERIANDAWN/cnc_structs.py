import ctypes
import termcolor


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


CNC_OBJECT_ASSET_NAME_LENGTH = 16


class StaticTile(CncStruct):
    _fields_ = [
        ('AssetName', ctypes.c_char * CNC_OBJECT_ASSET_NAME_LENGTH),
        ('ShapeIndex', ctypes.c_ushort),
    ]


class StaticMapView(CncStruct):
    _fields_ = [
        ('MapCellWidth', ctypes.c_int),
        ('MapCellHeight', ctypes.c_int),
        ('StaticCells', ctypes.POINTER(StaticTile)),
    ]


MAX_OBJECT_PIPS = 18


class DynamicObject(CncStruct):
    _fields_ = [
        ('AssetName', ctypes.c_char * CNC_OBJECT_ASSET_NAME_LENGTH),
        ('PositionX', ctypes.c_int),
        ('PositionY', ctypes.c_int),
        ('Strength', ctypes.c_short),
        ('ShapeIndex', ctypes.c_ushort),
        ('Owner', ctypes.c_ubyte),
        ('IsSelected', ctypes.c_ubyte),
        ('IsRepairing', ctypes.c_bool),
        ('Cloak', ctypes.c_ubyte),
        ('Pips', ctypes.c_int * MAX_OBJECT_PIPS),
        ('ControlGroup', ctypes.c_ubyte),
    ]


class CommonVectorRepresentationView(CncStruct):
    _fields_ = [
        ('map', StaticMapView),
        ('dynamic_objects_count', ctypes.c_size_t),
        ('dynamic_objects', ctypes.POINTER(DynamicObject)),
    ]


def decode_cell(data):
    text = data.AssetName.decode('ascii')[:2]
    if text == 'TI':  # tiberium
        text = '  '
        background = 'on_green'
    elif text == 'CL':  # CLEAR1
        text = '  '
        background = 'on_grey'
    else:
        background = 'on_grey'

    return (text, 'white', background)


def print_game_state(game_state: CommonVectorRepresentationView):
    map_list = [
        [
            decode_cell(game_state.map.StaticCells[i * game_state.map.MapCellWidth + j])
            for j in range(game_state.map.MapCellWidth)
        ]
        for i in range(game_state.map.MapCellHeight)
    ]
    for i in range(game_state.dynamic_objects_count):
        thing = game_state.dynamic_objects[i]
        x, y = thing.PositionY // 24, thing.PositionX // 24
        color = 'white'
        if thing.Owner != 255:
            color = ['yellow', 'blue', 'red', 'white', 'magenta', 'cyan'][thing.Owner]
        map_list[x][y] = (
            '{:2s}'.format(thing.AssetName.decode('ascii')[:2]),
            color,
            map_list[x][y][2],
        )
    return '\n'.join(map(lambda row: ''.join(termcolor.colored(*cell) for cell in row), map_list))
