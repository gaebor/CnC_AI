import ctypes

import numpy
from torch.nn.utils.rnn import pad_sequence as pad_sequence_torch
from torch import tensor

from cnc_ai.TIBERIANDAWN import cnc_structs


def pad_sequence(tensors):
    padded_tensors = pad_sequence_torch([tensor(t) for t in tensors], batch_first=True).numpy()
    return padded_tensors


def resize_sequence_to_minimum_length(sequence):
    if len(sequence) == 0:
        return numpy.zeros((1,) + sequence.shape[1:], dtype=sequence.dtype)
    return sequence


def pad_game_states(list_of_game_states):
    auxiliary_sidebar = {
        key: [
            resize_sequence_to_minimum_length(game_state[key])
            for game_state in list_of_game_states
        ]
        for key in [
            'SidebarAssetName',
            'SidebarContinuous',
        ]
    }

    result = {
        'dynamic_mask': compute_key_padding_mask(
            [len(game_state['AssetName']) for game_state in list_of_game_states]
        ),
        'sidebar_mask': compute_key_padding_mask(
            [len(sidebar) for sidebar in auxiliary_sidebar['SidebarAssetName']]
        ),
        **{
            key: numpy.stack([game_state[key] for game_state in list_of_game_states], 0)
            for key in ['StaticAssetName', 'StaticShapeIndex', 'SidebarInfos']
        },
        **{
            key: pad_sequence([game_state[key] for game_state in list_of_game_states])
            for key in [
                'AssetName',
                'ShapeIndex',
                'Owner',
                'Pips',
                'ControlGroup',
                'Cloak',
                'Continuous',
            ]
        },
        **{
            key: pad_sequence(auxiliary_sidebar[key])
            for key in ['SidebarAssetName', 'SidebarContinuous']
        },
    }
    return result


_static_masks = numpy.zeros((0, 0), dtype=bool)


def compute_key_padding_mask(lengths):
    global _static_masks
    """https://discuss.pytorch.org/t/create-a-mask-tensor-using-index/97303/6"""
    max_length = max(lengths)
    if max_length > _static_masks.shape[1]:
        _static_masks = numpy.triu(numpy.ones((max_length + 1, max_length), dtype=bool))
    return _static_masks[lengths, :max_length]


def render_add_player_command(player):
    buffer = bytes(ctypes.c_uint32(2))
    buffer += bytes(player)
    return buffer


def render_action(player_id, action_index, mouse_x, mouse_y):
    return bytes(ctypes.c_uint32(5)) + bytes(
        cnc_structs.ActionRequestArgs(
            player_id=player_id, action_index=action_index, x=mouse_x, y=mouse_y
        )
    )


def encode_list(list_of_strings):
    return b''.join(map(lambda s: str.encode(s, encoding='ascii') + b'\0', list_of_strings))
