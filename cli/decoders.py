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


def layers_term(layers, dynamic_map, static_map):
    term_array = [
        [' ' for i in range(static_map.MapCellWidth)] for j in range(static_map.MapCellHeight)
    ]
    number_of_units = [
        [0 for i in range(static_map.MapCellWidth)] for j in range(static_map.MapCellHeight)
    ]
    tiberium = tiberium_array(dynamic_map, static_map)

    for o in layers.Objects:
        i, j = o.CellY - static_map.MapCellY, o.CellX - static_map.MapCellX
        if o.Type >= 1 and o.Type <= 3:
            number_of_units[i][j] += 1

    for o in layers.Objects:
        i, j = o.CellY - static_map.MapCellY, o.CellX - static_map.MapCellX

        if tiberium[i, j]:
            background = ('on_green',)
        else:
            background = tuple()

        if o.Owner != b'\xff':
            color = (['yellow', 'blue', 'red', 'white', 'magenta', 'cyan'][ord(o.Owner) - 4],)
        else:
            color = tuple()

        if o.Type == 1:
            term_array[i][j] = termcolor.colored(str(number_of_units[i][j]), *(color + background))
        elif o.Type == 2 or o.Type == 3:
            term_array[i][j] = termcolor.colored(
                o.AssetName.decode('ascii')[0].lower(), *(color + background)
            )
        elif o.Type == 4:
            for tile in o.OccupyList[: o.OccupyListLength]:
                i_inc, j_inc = tile // 64, tile % 64
                term_array[i + i_inc][j + j_inc] = termcolor.colored(
                    o.AssetName.decode('ascii')[0].upper(), *(color + background)
                )

    return '\n'.join(map(''.join, term_array))


def players_units(layers, house):
    return [o for o in layers.Objects if ord(o.Owner) == house and o.IsSelectable]


def shroud_array(shrouds: cnc_structs.CNCShroudStruct, static_map):
    return np.array([entry.IsVisible for entry in shrouds.Entries], dtype=bool).reshape(
        (static_map.MapCellHeight, static_map.MapCellWidth)
    )


def occupiers(occupiers_struct, static_map):
    return [
        {'X': i % static_map.MapCellWidth, 'Y': i // static_map.MapCellWidth, 'Objects': e.Objects}
        for i, e in enumerate(occupiers_struct.Entries)
        if e.Count > 0
    ]
