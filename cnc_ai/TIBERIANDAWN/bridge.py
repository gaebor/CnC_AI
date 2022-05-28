import ctypes

import numpy
from torch.nn.utils.rnn import pad_sequence
from torch import tensor

from cnc_ai.TIBERIANDAWN import cnc_structs


def pad_game_states(list_of_game_states, device):
    result = {
        **{
            new_key: tensor(
                compute_key_padding_mask(
                    [len(game_state[key]) for game_state in list_of_game_states]
                )
            ).to(device)
            for new_key, key in [
                ('dynamic_mask', 'AssetName'),
                ('sidebar_mask', 'SidebarAssetName'),
            ]
        },
        **{
            key: tensor(
                numpy.stack([game_state[key] for game_state in list_of_game_states], 0)
            ).to(device)
            for key in ['StaticAssetName', 'StaticShapeIndex', 'SidebarInfos']
        },
        **{
            key: pad_sequence(
                [tensor(game_state[key]) for game_state in list_of_game_states],
                batch_first=True,
            ).to(device)
            for key in [
                'AssetName',
                'ShapeIndex',
                'Owner',
                'Pips',
                'ControlGroup',
                'Cloak',
                'Continuous',
                'SidebarAssetName',
                'SidebarContinuous',
            ]
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
    if action_index < 12:  # INPUTREQUEST
        return bytes(ctypes.c_uint32(5)) + bytes(
            cnc_structs.InputRequestArgs(
                player_id=player_id, requestType=action_index, x1=1488 * mouse_x, y1=1488 * mouse_y
            )
        )
    else:  # SIDEBARREQUEST
        sidebar_index = action_index - 12
        action_type, sidebar_element = sidebar_index % 12, sidebar_index // 12
        return bytes(ctypes.c_uint32(6)) + bytes(
            cnc_structs.SidebarRequestArgs(
                player_id=player_id, requestType=action_type, assetNameIndex=sidebar_element
            ),
        )


def encode_list(list_of_strings):
    return b''.join(map(lambda s: str.encode(s, encoding='ascii') + b'\0', list_of_strings))
