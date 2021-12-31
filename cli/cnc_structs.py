import ctypes
from functools import partial


class CncStruct(ctypes.Structure):
    _pack_ = 1

    def __repr__(self):
        self_repr = {}
        for field_name, _ in self._fields_:
            field = getattr(self, field_name)
            if isinstance(field, ctypes.Array):
                self_repr[field_name] = list(field)[:10]
            else:
                self_repr[field_name] = field
        return repr(self_repr)


class CNCDifficultyDataStruct(CncStruct):
    _fields_ = [
        ('FirepowerBias', ctypes.c_float),
        ('GroundspeedBias', ctypes.c_float),
        ('AirspeedBias', ctypes.c_float),
        ('ArmorBias', ctypes.c_float),
        ('ROFBias', ctypes.c_float),
        ('CostBias', ctypes.c_float),
        ('BuildSpeedBias', ctypes.c_float),
        ('RepairDelay', ctypes.c_float),
        ('BuildDelay', ctypes.c_float),
        ('IsBuildSlowdown', ctypes.c_bool),
        ('IsWallDestroyer', ctypes.c_bool),
        ('IsContentScan', ctypes.c_bool),
    ]


class CNCRulesDataStruct(CncStruct):
    _fields_ = [('Difficulties', CNCDifficultyDataStruct * 3)]


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


class CNCShroudEntryStruct(CncStruct):
    _fields_ = [
        ('ShadowIndex', ctypes.c_char),
        ('IsVisible', ctypes.c_bool),
        ('IsMapped', ctypes.c_bool),
        ('IsJamming', ctypes.c_bool),
    ]


class CNCShroudStruct(CncStruct):
    _fields_ = [
        ('Count', ctypes.c_int),
        # ('Entries', CNCShroudEntryStruct * 1),
    ]


def parse_variable_length_struct(buffer, cls, variable_field_name, variable_field_type):
    partially_parsed = cls.from_address(ctypes.addressof(buffer))

    class X(CncStruct):
        _fields_ = cls._fields_ + [
            (variable_field_name, variable_field_type * partially_parsed.Count),
        ]

    return X.from_buffer_copy(buffer)


class CNCStaticCellStruct(CncStruct):
    _fields_ = [
        ('TemplateTypeName', ctypes.c_char * 32),
        ('IconNumber', ctypes.c_int),
    ]


class CNCMapDataStruct(CncStruct):
    _fields_ = [
        ('MapCellX', ctypes.c_int),
        ('MapCellY', ctypes.c_int),
        ('MapCellWidth', ctypes.c_int),
        ('MapCellHeight', ctypes.c_int),
        ('OriginalMapCellX', ctypes.c_int),
        ('OriginalMapCellY', ctypes.c_int),
        ('OriginalMapCellWidth', ctypes.c_int),
        ('OriginalMapCellHeight', ctypes.c_int),
        ('Theater', ctypes.c_int),  # CnCTheaterType
        ('ScenarioName', ctypes.c_char * 512),  # _MAX_FNAME+_MAX_EXT
        ('StaticCells', CNCStaticCellStruct * MAX_EXPORT_CELLS),
    ]


MAX_OCCUPY_CELLS = 36


class CNCSidebarEntryStruct(CncStruct):
    _fields_ = [
        ('AssetName', ctypes.c_char * 16),
        ('BuildableType', ctypes.c_int),
        ('BuildableID', ctypes.c_int),
        ('Type', ctypes.c_int),  # DllObjectTypeEnum
        ('SuperWeaponType', ctypes.c_int),  # DllSuperweaponTypeEnum
        ('Cost', ctypes.c_int),
        ('PowerProvided', ctypes.c_int),
        ('BuildTime', ctypes.c_int),
        ('Progress', ctypes.c_float),
        ('PlacementList', ctypes.c_short * MAX_OCCUPY_CELLS),
        ('PlacementListLength', ctypes.c_int),
        ('Completed', ctypes.c_bool),
        ('Constructing', ctypes.c_bool),
        ('ConstructionOnHold', ctypes.c_bool),
        ('Busy', ctypes.c_bool),
        ('BuildableViaCapture', ctypes.c_bool),
        ('Fake', ctypes.c_bool),
    ]


class CNCSidebarStruct(CncStruct):
    _fields_ = [
        ('EntryCount', ctypes.c_int * 2),
        ('Credits', ctypes.c_int),
        ('CreditsCounter', ctypes.c_int),
        ('Tiberium', ctypes.c_int),
        ('MaxTiberium', ctypes.c_int),
        ('PowerProduced', ctypes.c_int),
        ('PowerDrained', ctypes.c_int),
        ('MissionTimer', ctypes.c_int),
        ('UnitsKilled', ctypes.c_uint),
        ('BuildingsKilled', ctypes.c_uint),
        ('UnitsLost', ctypes.c_uint),
        ('BuildingsLost', ctypes.c_uint),
        ('TotalHarvestedCredits', ctypes.c_uint),
        ('RepairBtnEnabled', ctypes.c_bool),
        ('SellBtnEnabled', ctypes.c_bool),
        ('RadarMapActive', ctypes.c_bool),
        # ('Entries', CNCSidebarEntryStruct	 * 1),
    ]


def parse_sidebar_buffer(buffer):
    partially_parsed = CNCSidebarStruct.from_address(ctypes.addressof(buffer))

    class X(CncStruct):
        _fields_ = CNCSidebarStruct._fields_ + [
            (
                'Entries',
                CNCSidebarEntryStruct
                * (partially_parsed.EntryCount[0] + partially_parsed.EntryCount[1]),
            ),
        ]

    return X.from_buffer_copy(buffer)


class CNCPlacementCellInfoStruct(CncStruct):
    _fields_ = [
        ('PassesProximityCheck', ctypes.c_bool),
        ('GenerallyClear', ctypes.c_bool),
    ]


class CNCPlacementInfoStruct(CncStruct):
    _fields_ = [
        ('Count', ctypes.c_int),
        # ('CellInfo', CNCPlacementCellInfoStruct),
    ]


GameStateRequestEnum = {
    'GAME_STATE_NONE': (0, None),
    'GAME_STATE_STATIC_MAP': (1, CNCMapDataStruct.from_buffer_copy),
    'GAME_STATE_DYNAMIC_MAP': (2, None),
    'GAME_STATE_LAYERS': (3, None),
    'GAME_STATE_SIDEBAR': (4, parse_sidebar_buffer),
    'GAME_STATE_PLACEMENT': (
        5,
        partial(
            parse_variable_length_struct,
            cls=CNCPlacementInfoStruct,
            variable_field_name='CellInfo',
            variable_field_type=CNCPlacementCellInfoStruct,
        ),
    ),
    'GAME_STATE_SHROUD': (
        6,
        partial(
            parse_variable_length_struct,
            cls=CNCShroudStruct,
            variable_field_name='Entries',
            variable_field_type=CNCShroudEntryStruct,
        ),
    ),
    'GAME_STATE_OCCUPIER': (7, None),
    'GAME_STATE_PLAYER_INFO': (8, CNCPlayerInfoStruct.from_buffer_copy),
}
