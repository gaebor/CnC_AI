import numpy as np

import cnc_structs


def staticmap_array(map: cnc_structs.CNCMapDataStruct) -> np.ndarray:
    tile_names = np.zeros(map.MapCellHeight * map.MapCellWidth, dtype='S32')
    for i in range(tile_names.size):
        tile_names[i] = map.StaticCells[i].TemplateTypeName

    return tile_names.reshape((map.MapCellHeight, map.MapCellWidth))


def dynamicmap_array(map: cnc_structs.CNCDynamicMapStruct, static_map):
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
