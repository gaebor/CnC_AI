import numpy as np
import termcolor

import cnc_structs


def staticmap_array(map: cnc_structs.CNCMapDataStruct) -> np.ndarray:
    tile_names = np.zeros(map.MapCellHeight * map.MapCellWidth, dtype='S32')
    for i in range(tile_names.size):
        tile_names[i] = map.StaticCells[i].TemplateTypeName

    return tile_names.reshape((map.MapCellHeight, map.MapCellWidth))


def tiberium_array(map: cnc_structs.CNCDynamicMapStruct, static_map):
    array = np.zeros((static_map.MapCellHeight, static_map.MapCellWidth), dtype=bool)
    for entry in map.Entries:
        array[
            entry.CellY - static_map.MapCellY, entry.CellX - static_map.MapCellX
        ] = entry.IsResource
    return array


def f(dynamic_map, layers, occupiers, MapCellHeight, MapCellWidth, House, AllyFlags):
    fixed_pos_map_assets = np.zeros(MapCellHeight * MapCellWidth, dtype='S32')
    fixed_pos_map_shapes = np.zeros(MapCellHeight * MapCellWidth, dtype='uint8')

    actors = []
    terrains = {}

    for o in layers.Objects:
        if o.Type == 5:  # terrain
            terrains[o.ID] = (o.AssetName.decode('ascii'), o.ShapeIndex)
        else:
            if (ord(o.Owner) & AllyFlags) or o.Cloak != 2:  # CLOAKED
                actors.append(
                    {
                        'Asset': o.AssetName.decode('ascii'),
                        'Shape': o.ShapeIndex,
                        'Position': (o.PositionY, o.PositionX),
                        'Owner': ord(o.Owner),
                        'Strength': o.Strength,
                        'IsSelected': (o.IsSelectedMask & (1 << House)) != 0,
                        'ControlGroup': o.ControlGroup,
                        'IsRepairing': o.IsRepairing,
                        'IsPrimaryFactory': o.IsPrimaryFactory,
                        'Cloak': o.Cloak,
                        'Pips': list(o.Pips),
                    }
                )

    for i, entry in enumerate(dynamic_map.Entries):
        if entry.IsOverlay and entry.Type >= 1 and entry.Type <= 5:  # walls
            actors.append(
                {
                    'Asset': entry.AssetName.decode('ascii'),
                    'Shape': entry.ShapeIndex,
                    'Position': (entry.PositionY, entry.PositionX),
                    'Owner': ord(entry.Owner),
                    'Strength': 0,
                    'IsSelected': False,
                    'ControlGroup': -1,
                    'IsRepairing': False,
                    'IsPrimaryFactory': False,
                    'Cloak': 0,
                    'Pips': [],
                }
            )
        else:
            fixed_pos_map_assets[i] = entry.AssetName
            fixed_pos_map_shapes[i] = entry.ShapeIndex

    for i, o in enumerate(occupiers.Entries):
        if len(o.Objects) == 1 and o.Objects[0].Type == 5:  # terrain
            fixed_pos_map_assets[i], fixed_pos_map_shapes[i] = terrains[o.Objects[0].ID]

    return (
        fixed_pos_map_assets.reshape((MapCellHeight, MapCellWidth)),
        fixed_pos_map_shapes.reshape((MapCellHeight, MapCellWidth)),
        actors,
    )


def layers_array(objects: cnc_structs.CNCObjectListStruct, static_map):
    array = np.zeros((static_map.MapCellHeight, static_map.MapCellWidth), dtype=int)
    for thing in objects.Objects:
        array[thing.CellY - static_map.MapCellY, thing.CellX - static_map.MapCellX] = thing.Type
    return array


def layers_list(layers, static_map):
    return [
        {
            'Owner': ord(o.Owner),
            'Asset': o.AssetName.decode('ascii'),
            'Type': o.Type,
            'ID': o.ID,
            'X': o.CellX - static_map.MapCellX,
            'Y': o.CellY - static_map.MapCellY,
            'OccupyList': o.OccupyList[: o.OccupyListLength],
        }
        for o in layers.Objects
        if o.Type > 0
    ]


def units_and_buildings_dict(layers):
    return {(o.Type, o.ID): o for o in layers.Objects if o.Type >= 1 and o.Type <= 4}


def layers_term(layers, dynamic_map, static_map, occupiers):
    units_and_buildings = units_and_buildings_dict(layers)
    tiberium = tiberium_array(dynamic_map, static_map)

    text = ''

    for i, (occupier, is_tiberium, static_cell) in enumerate(
        zip(occupiers.Entries, tiberium.flat, static_map.StaticCells)
    ):
        if i < static_map.MapCellWidth or i >= static_map.MapCellWidth * (
            static_map.MapCellHeight - 1
        ):
            continue

        cell_text = ' '
        color = 'white'
        background = 'on_green' if is_tiberium else 'on_grey'
        # print(static_cell.TemplateTypeName)
        if i % static_map.MapCellWidth == 0:
            cell_text = '|'
        elif i % static_map.MapCellWidth == static_map.MapCellWidth - 1:
            cell_text = '|\n'
        elif static_cell.TemplateTypeName.startswith(
            b'W'
        ) or static_cell.TemplateTypeName.startswith(
            b'RV'
        ):  # river or water
            background = 'on_blue'
        elif static_cell.TemplateTypeName.startswith(
            b'S'
        ) and not static_cell.TemplateTypeName.startswith(
            b'SH'
        ):  # slope but not shore
            background = 'on_white'

        if len(occupier.Objects) > 0:
            occupier = occupier.Objects[0]
            if (occupier.Type, occupier.ID) in units_and_buildings:
                occupier = units_and_buildings[(occupier.Type, occupier.ID)]
                color = ['yellow', 'blue', 'red', 'white', 'magenta', 'cyan'][
                    ord(occupier.Owner) - 4
                ]
                cell_text = occupier.AssetName.decode('ascii')[0]
                if occupier.Type >= 1 and occupier.Type <= 3:
                    cell_text = cell_text.lower()
                elif occupier.Type == 4:
                    cell_text = cell_text.upper()
        text += termcolor.colored(cell_text, color, background)
    return text.rstrip('\n')


def sidebar_term(sidebar: cnc_structs.CNCSidebarStruct):
    return (
        f'Tiberium: {(100 * sidebar.Tiberium) // sidebar.MaxTiberium if sidebar.MaxTiberium > 0 else 0 :3d}% '
        f'Power: {(100 * sidebar.PowerDrained) // sidebar.PowerProduced if sidebar.PowerProduced > 0 else 0:3d}%'
        f' Credits: {sidebar.Credits} | '
        + ', '.join(
            sidebar.Entries[i].AssetName.decode('ascii') for i in range(sidebar.EntryCount[0])
        ),
        '|',
        ', '.join(
            sidebar.Entries[i].AssetName.decode('ascii')
            for i in range(sidebar.EntryCount[0], sidebar.EntryCount[0] + sidebar.EntryCount[1])
        ),
    )


def players_units(layers, house):
    return [o for o in layers.Objects if ord(o.Owner) == house and o.IsSelectable]


def shroud_array(shrouds: cnc_structs.CNCShroudStruct, static_map):
    return np.array([entry.IsVisible for entry in shrouds.Entries], dtype=bool).reshape(
        (static_map.MapCellHeight, static_map.MapCellWidth)
    )


def occupiers_list(occupiers_struct, static_map):
    return [
        {'X': i % static_map.MapCellWidth, 'Y': i // static_map.MapCellWidth, 'Objects': e.Objects}
        for i, e in enumerate(occupiers_struct.Entries)
        if e.Count > 0
    ]


def occupiers_array(occupiers_struct, static_map):
    return np.array(
        [
            ((-1 if len(e.Objects) == 0 else e.Objects[0].Type) << 32)
            + (-1 if len(e.Objects) == 0 else e.Objects[0].ID)
            for e in occupiers_struct.Entries
        ]
    ).reshape((static_map.MapCellHeight, static_map.MapCellWidth))
