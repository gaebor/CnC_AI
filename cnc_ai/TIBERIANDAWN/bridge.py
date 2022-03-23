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


def render_actions(offset, n_players, mouse_action, sidebar_action):
    buffer = b''
    for i, player_id in zip(range(offset, offset + n_players), range(n_players)):
        if sidebar_action.shape[1] > 0 and mouse_action[i].max() < sidebar_action[i].max():
            sidebar_element, action_type = numpy.unravel_index(
                sidebar_action[i].argmax(), sidebar_action[i].shape
            )
        else:
            x, y, sub5, input_type = numpy.unravel_index(
                mouse_action[i].argmax(), mouse_action[i].shape
            )

        buffer += bytes(ctypes.c_uint32(7))  # NOUGHTREQUEST
        buffer += bytes(cnc_structs.NoughtRequestArgs(player_id=player_id))
        # if action_type == 0:
        #     buffer += bytes(ctypes.c_uint32(7))  # NOUGHTREQUEST
        #     buffer += bytes(cnc_structs.NoughtRequestArgs(player_id=player_id))
        # if action_type == 1:
        #     if sidebar_action.shape[1] > 0 and numpy.isnfinite(sidebar_action[i, 0]):
        #         possible_actions = sidebar_action[i]
        #         best_sidebar_element, best_action_type = numpy.unravel_index(
        #             possible_actions.argmax(), possible_actions.shape
        #         )
        #         buffer += bytes(ctypes.c_uint32(6))  # SIDEBARREQUEST
        #         buffer += bytes(
        #             cnc_structs.SidebarRequestArgs(
        #                 player_id=player_id,
        #                 requestType=best_action_type,
        #                 assetNameIndex=best_sidebar_element,
        #             )
        #         )
        # elif action_type == 2:
        #     buffer += bytes(ctypes.c_uint32(5))  # INPUTREQUEST
        #     request_type = input_request_type[i].argmax()
        #     buffer += bytes(
        #         cnc_structs.InputRequestArgs(
        #             player_id=player_id,
        #             requestType=request_type,
        #             x1=mouse_position[i, request_type, 0],
        #             y1=mouse_position[i, request_type, 1],
        #         )
        #     )
    return buffer


def encode_list(list_of_strings):
    return b''.join(map(lambda s: str.encode(s, encoding='ascii') + b'\0', list_of_strings))
