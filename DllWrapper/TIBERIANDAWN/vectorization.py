import ctypes

import numpy

import cnc_structs


def convert_to_np(game_state_buffer):
    offset = 8 * ctypes.sizeof(ctypes.c_int)
    map_cells = numpy.frombuffer(
        game_state_buffer, dtype='uint32', count=2 * 62 * 62, offset=offset
    )
    asset_names = map_cells[0::2].reshape((62, 62))
    shape_indices = map_cells[1::2].reshape((62, 62))

    offset = ctypes.sizeof(cnc_structs.StaticMap)
    dynamic_objects_count = ctypes.c_uint32.from_buffer_copy(game_state_buffer, offset).value
    offset += ctypes.sizeof(ctypes.c_uint32)
