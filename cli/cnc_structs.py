import ctypes


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


class GameOverMultiPlayerStatsStruct(CncStruct):
    _fields_ = [
        ('GlyphxPlayerID', ctypes.c_uint64),
        ('IsHuman', ctypes.c_bool),
        ('WasHuman', ctypes.c_bool),
        ('IsWinner', ctypes.c_bool),
        ('ResourcesGathered', ctypes.c_int),
        ('TotalUnitsKilled', ctypes.c_int),
        ('TotalStructuresKilled', ctypes.c_int),
        ('Efficiency', ctypes.c_int),
        ('Score', ctypes.c_int),
    ]
