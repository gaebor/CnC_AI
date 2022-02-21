import ctypes

import numpy

import cnc_structs


def read_array_from_buffer(buffer, offset, Type):
    count = ctypes.c_uint32.from_buffer_copy(buffer, offset).value
    offset += ctypes.sizeof(ctypes.c_uint32)
    number_of_members = ctypes.sizeof(Type) // 4
    indices = numpy.frombuffer(
        buffer, dtype='uint32', count=count * number_of_members, offset=offset
    ).reshape(count, number_of_members)
    continuous = numpy.frombuffer(
        buffer, dtype='float32', count=count * number_of_members, offset=offset
    ).reshape(count, number_of_members)
    offset += count * ctypes.sizeof(Type)
    return offset, indices, continuous


def convert_to_np(game_state_buffer):
    offset = 8 * ctypes.sizeof(ctypes.c_int)
    map_cells = numpy.frombuffer(
        game_state_buffer, dtype='uint32', count=2 * 62 * 62, offset=offset
    )
    offset = ctypes.sizeof(cnc_structs.StaticMap)

    offset, dynamic_objects_indices, dynamic_objects_continuous = read_array_from_buffer(
        game_state_buffer, offset, cnc_structs.DynamicObject
    )

    sidebar_members = numpy.frombuffer(
        game_state_buffer,
        dtype='float32',
        count=ctypes.sizeof(cnc_structs.SideBarMembers) // 4,
        offset=offset,
    )
    offset += ctypes.sizeof(cnc_structs.SideBarMembers)

    offset, sidebar_entries_indices, sidebar_entries_continuous = read_array_from_buffer(
        game_state_buffer, offset, cnc_structs.SidebarEntry
    )

    return {
        'StaticAssetName': map_cells[0::2].reshape((62, 62)),
        'StaticShapeIndex': map_cells[1::2].reshape((62, 62)),
        'AssetName': dynamic_objects_indices[:, 0],
        'ShapeIndex': dynamic_objects_indices[:, 1],
        'Owner': dynamic_objects_indices[:, 2],
        'Pips': dynamic_objects_indices[:, 3 : 3 + cnc_structs.MAX_OBJECT_PIPS],
        'ControlGroup': dynamic_objects_indices[:, 3 + cnc_structs.MAX_OBJECT_PIPS],
        'Cloak': dynamic_objects_indices[:, 3 + cnc_structs.MAX_OBJECT_PIPS + 1],
        'Continuous': dynamic_objects_continuous[:, -5:],
        'SidebarInfos': sidebar_members,
        'SidebarAssetName': sidebar_entries_indices[:, 0],
        'SidebarContinuous': sidebar_entries_continuous[:, 3:],
    }
