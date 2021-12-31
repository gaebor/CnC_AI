import numpy as np

import cnc_structs


def staticmap(map: cnc_structs.CNCMapDataStruct) -> np.ndarray:
    tile_names = np.zeros(map.MapCellHeight * map.MapCellWidth, dtype='S32')
    for i in range(tile_names.size):
        tile_names[i] = map.StaticCells[i].TemplateTypeName

    return tile_names.reshape((map.MapCellHeight, map.MapCellWidth))


def dynamicmap(map: cnc_structs.CNCDynamicMapStruct, static_map):
    array = np.zeros((static_map.MapCellHeight, static_map.MapCellWidth), dtype=bool)
    for entry in map.Entries:
        array[
            entry.CellY - static_map.MapCellY, entry.CellX - static_map.MapCellX
        ] = entry.IsResource
    return array


def layers(objects: cnc_structs.CNCObjectListStruct, static_map):
    array = np.zeros((static_map.MapCellHeight, static_map.MapCellWidth), dtype=int)
    for thing in objects.Objects:
        array[thing.CellY - static_map.MapCellY, thing.CellX - static_map.MapCellX] = thing.Type
    return array


def shroud(shrouds: cnc_structs.CNCShroudStruct, static_map):
    return np.array([entry.IsVisible for entry in shrouds.Entries], dtype=bool).reshape(
        (static_map.MapCellHeight, static_map.MapCellWidth)
    )
