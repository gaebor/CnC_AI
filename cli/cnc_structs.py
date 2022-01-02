import ctypes


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


def parse_variable_length(cls, member, member_cls):
    def parser(buffer):
        header = cls.from_buffer_copy(buffer)
        l = []
        setattr(header, member, l)
        offset = ctypes.sizeof(cls)
        for _ in range(header.Count):
            l.append(member_cls.from_buffer_copy(buffer, offset))
            offset += ctypes.sizeof(member_cls)
        return header

    return parser


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


def parse_sidebar(buffer):
    header = CNCSidebarStruct.from_buffer_copy(buffer)
    l = []
    header.Entries = l
    offset = ctypes.sizeof(CNCSidebarStruct)
    for _ in range(sum(header.EntryCount)):
        l.append(CNCSidebarEntryStruct.from_buffer_copy(buffer, offset))
        offset += ctypes.sizeof(CNCSidebarEntryStruct)
    return header


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


class CNCDynamicMapEntryStruct(CncStruct):
    _fields_ = [
        ('AssetName', ctypes.c_char * 16),
        ('PositionX', ctypes.c_int),
        ('PositionY', ctypes.c_int),
        ('Width', ctypes.c_int),
        ('Height', ctypes.c_int),
        ('Type', ctypes.c_short),
        ('Owner', ctypes.c_char),
        ('DrawFlags', ctypes.c_int),
        ('CellX', ctypes.c_byte),
        ('CellY', ctypes.c_byte),
        ('ShapeIndex', ctypes.c_byte),
        ('IsSmudge', ctypes.c_bool),
        ('IsOverlay', ctypes.c_bool),
        ('IsResource', ctypes.c_bool),
        ('IsSellable', ctypes.c_bool),
        ('IsTheaterShape', ctypes.c_bool),
        ('IsFlag', ctypes.c_bool),
    ]


class CNCDynamicMapStruct(CncStruct):
    _fields_ = [
        ('VortexActive', ctypes.c_bool),
        ('VortexX', ctypes.c_int),
        ('VortexY', ctypes.c_int),
        ('VortexWidth', ctypes.c_int),
        ('VortexHeight', ctypes.c_int),
        ('Count', ctypes.c_int),
        # ('Entries', CNCDynamicMapEntryStruct * 1),
    ]


CNC_OBJECT_ASSET_NAME_LENGTH = 16
MAX_OBJECT_PIPS = 18
MAX_OBJECT_LINES = 3


class CNCObjectLineStruct(CncStruct):
    _fields_ = [
        ('X', ctypes.c_int),
        ('Y', ctypes.c_int),
        ('X1', ctypes.c_int),
        ('Y1', ctypes.c_int),
        ('Frame', ctypes.c_int),
        ('Color', ctypes.c_byte),
    ]


class CNCObjectStruct(CncStruct):
    _fields_ = [
        ('CNCInternalObjectPointer', ctypes.c_void_p),
        ('TypeName', ctypes.c_char * CNC_OBJECT_ASSET_NAME_LENGTH),
        ('AssetName', ctypes.c_char * CNC_OBJECT_ASSET_NAME_LENGTH),
        ('Type', ctypes.c_int),  # DllObjectTypeEnum
        ('ID', ctypes.c_int),
        ('BaseObjectID', ctypes.c_int),
        ('BaseObjectType', ctypes.c_int),  # DllObjectTypeEnum
        ('PositionX', ctypes.c_int),
        ('PositionY', ctypes.c_int),
        ('Width', ctypes.c_int),
        ('Height', ctypes.c_int),
        ('Altitude', ctypes.c_int),
        ('SortOrder', ctypes.c_int),
        ('Scale', ctypes.c_int),
        ('DrawFlags', ctypes.c_int),
        ('MaxStrength', ctypes.c_short),
        ('Strength', ctypes.c_short),
        ('ShapeIndex', ctypes.c_ushort),
        ('CellX', ctypes.c_ushort),
        ('CellY', ctypes.c_ushort),
        ('CenterCoordX', ctypes.c_ushort),
        ('CenterCoordY', ctypes.c_ushort),
        ('SimLeptonX', ctypes.c_short),
        ('SimLeptonY', ctypes.c_short),
        ('DimensionX', ctypes.c_byte),
        ('DimensionY', ctypes.c_byte),
        ('Rotation', ctypes.c_byte),
        ('MaxSpeed', ctypes.c_byte),
        ('Owner', ctypes.c_char),
        ('RemapColor', ctypes.c_char),
        ('SubObject', ctypes.c_char),
        ('IsSelectable', ctypes.c_bool),
        ('IsSelectedMask', ctypes.c_uint),
        ('IsRepairing', ctypes.c_bool),
        ('IsDumping', ctypes.c_bool),
        ('IsTheaterSpecific', ctypes.c_bool),
        ('FlashingFlags', ctypes.c_uint),
        ('Cloak', ctypes.c_byte),
        ('CanRepair', ctypes.c_bool),
        ('CanDemolish', ctypes.c_bool),
        ('CanDemolishUnit', ctypes.c_bool),
        ('OccupyList', ctypes.c_short * MAX_OCCUPY_CELLS),
        ('OccupyListLength', ctypes.c_int),
        ('Pips', ctypes.c_int * MAX_OBJECT_PIPS),
        ('NumPips', ctypes.c_int),
        ('MaxPips', ctypes.c_int),
        ('Lines', CNCObjectLineStruct * MAX_OBJECT_LINES),
        ('NumLines', ctypes.c_int),
        ('RecentlyCreated', ctypes.c_bool),
        ('IsALoaner', ctypes.c_bool),
        ('IsFactory', ctypes.c_bool),
        ('IsPrimaryFactory', ctypes.c_bool),
        ('IsDeployable', ctypes.c_bool),
        ('IsAntiGround', ctypes.c_bool),
        ('IsAntiAircraft', ctypes.c_bool),
        ('IsSubSurface', ctypes.c_bool),
        ('IsNominal', ctypes.c_bool),
        ('IsDog', ctypes.c_bool),
        ('IsIronCurtain', ctypes.c_bool),
        ('IsInFormation', ctypes.c_bool),
        ('CanMove', ctypes.c_bool * MAX_HOUSES),
        ('CanFire', ctypes.c_bool * MAX_HOUSES),
        ('CanDeploy', ctypes.c_bool),
        ('CanHarvest', ctypes.c_bool),
        ('CanPlaceBombs', ctypes.c_bool),
        ('IsFixedWingedAircraft', ctypes.c_bool),
        ('IsFake', ctypes.c_bool),
        ('ControlGroup', ctypes.c_byte),
        ('VisibleFlags', ctypes.c_uint),
        ('SpiedByFlags', ctypes.c_uint),
        ('ProductionAssetName', ctypes.c_char * CNC_OBJECT_ASSET_NAME_LENGTH),
        ('OverrideDisplayName', ctypes.c_char_p),
        ('ActionWithSelected', ctypes.c_byte * MAX_HOUSES),  # DllActionTypeEnum
    ]


class CNCObjectListStruct(CncStruct):
    _fields_ = [
        ('Count', ctypes.c_int),
        # ('Objects', CNCObjectStruct * 1),
    ]


class CNCOccupierObjectStruct(CncStruct):
    _fields_ = [
        ('Type', ctypes.c_int),  # DllObjectTypeEnum
        ('ID', ctypes.c_int),
    ]


class CNCOccupierEntryHeaderStruct(CncStruct):
    _fields_ = [
        ('Count', ctypes.c_int),
    ]


class CNCOccupierHeaderStruct(CncStruct):
    _fields_ = [
        ('Count', ctypes.c_int),
    ]


def parse_occupier(buffer):
    header = CNCOccupierHeaderStruct.from_buffer_copy(buffer)
    offset = ctypes.sizeof(CNCOccupierHeaderStruct)
    entries = []
    setattr(header, 'Entries', entries)
    for _ in range(header.Count):
        entry = CNCOccupierEntryHeaderStruct.from_buffer_copy(buffer, offset)
        offset += ctypes.sizeof(CNCOccupierEntryHeaderStruct)
        objects = []
        setattr(entry, 'Objects', objects)
        for _ in range(entry.Count):
            objects.append(CNCOccupierObjectStruct.from_buffer_copy(buffer, offset))
            offset += ctypes.sizeof(CNCOccupierObjectStruct)
        offset += ctypes.sizeof(CNCOccupierObjectStruct)
        entries.append(entry)
    return header


GameStateRequestEnum = {
    'GAME_STATE_NONE': (0, None),
    'GAME_STATE_STATIC_MAP': (1, CNCMapDataStruct.from_buffer_copy),
    'GAME_STATE_DYNAMIC_MAP': (
        2,
        parse_variable_length(CNCDynamicMapStruct, 'Entries', CNCDynamicMapEntryStruct),
    ),
    'GAME_STATE_LAYERS': (
        3,
        parse_variable_length(CNCObjectListStruct, 'Objects', CNCObjectStruct),
    ),
    'GAME_STATE_SIDEBAR': (4, parse_sidebar),
    'GAME_STATE_PLACEMENT': (
        5,
        parse_variable_length(CNCPlacementInfoStruct, 'CellInfo', CNCPlacementCellInfoStruct),
    ),
    'GAME_STATE_SHROUD': (
        6,
        parse_variable_length(CNCShroudStruct, 'Entries', CNCShroudEntryStruct),
    ),
    'GAME_STATE_OCCUPIER': (7, parse_occupier),
    'GAME_STATE_PLAYER_INFO': (8, CNCPlayerInfoStruct.from_buffer_copy),
}
