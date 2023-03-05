from typing import List, Union
from dataclasses import dataclass, asdict

import numpy
from numpy.typing import NDArray

from cnc_ai.common import pad_sequence


def resize_sequence_to_minimum_length(sequence):
    if len(sequence) == 0:
        return numpy.zeros((1,) + sequence.shape[1:], dtype=sequence.dtype)
    return sequence


def prepend_zero(tensor):
    result = numpy.concatenate(
        [numpy.zeros((tensor.shape[0], 1) + tensor.shape[2:], dtype=tensor.dtype), tensor],
        axis=1,
    )
    return result


NpFloat = Union[numpy.float64, numpy.float32, numpy.float16]
NpInt = Union[numpy.int32, numpy.int64, numpy.uint32, numpy.uint64]


@dataclass
class GameState:
    dynamic_mask: NDArray[bool] = None
    sidebar_mask: NDArray[bool] = None
    StaticAssetName: NDArray[NpInt]
    StaticShapeIndex: NDArray[NpInt]
    AssetName: NDArray[NpInt]
    ShapeIndex: NDArray[NpInt]
    Owner: NDArray[NpInt]
    Pips: NDArray[NpInt]
    ControlGroup: NDArray[NpInt]
    Cloak: NDArray[NpInt]
    Continuous: NDArray[NpFloat]
    SidebarInfos: NDArray[NpFloat]
    SidebarAssetName: NDArray[NpInt]
    SidebarContinuous: NDArray[NpFloat]

    def __post_init__(self):
        if self.dynamic_mask is None:
            self.dynamic_mask = numpy.ones_like(self.AssetName, dtype=bool)
        if self.sidebar_mask is None:
            self.sidebar_mask = numpy.ones_like(self.SidebarAssetName, dtype=bool)

    def take(self, *slices: slice):
        return GameState(**{member: value[slices] for member, value in asdict(self).items()})


@dataclass
class GameAction:
    button: NDArray[bool]
    mouse_x: NDArray[NpFloat]
    mouse_y: NDArray[NpFloat]


def pad_game_states(list_of_game_states: List[GameState]) -> GameState:
    result = {
        'dynamic_mask': compute_key_padding_mask(
            [len(game_state['AssetName']) for game_state in list_of_game_states]
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
                'SidebarAssetName',
                'SidebarContinuous',
            ]
        },
    }
    result['sidebar_mask'] = compute_key_padding_mask(
        [len(game_state['SidebarAssetName']) + 1 for game_state in list_of_game_states]
    )
    result['SidebarAssetName'] = prepend_zero(result['SidebarAssetName'])
    result['SidebarContinuous'] = prepend_zero(result['SidebarContinuous'])
    return result


def pad_game_actions(list_of_game_actions: List[GameAction]) -> GameAction:
    button_actions = pad_sequence([button for (button, _, _) in list_of_game_actions])
    mouse_x = numpy.array([mouse_x for (_, mouse_x, _) in list_of_game_actions])
    mouse_y = numpy.array([mouse_y for (_, _, mouse_y) in list_of_game_actions])
    return button_actions, mouse_x, mouse_y


_static_masks = numpy.zeros((0, 0), dtype=bool)


def compute_key_padding_mask(lengths):
    global _static_masks
    """https://discuss.pytorch.org/t/create-a-mask-tensor-using-index/97303/6"""
    max_length = max(lengths)
    if max_length > _static_masks.shape[1]:
        _static_masks = numpy.triu(numpy.ones((max_length + 1, max_length), dtype=bool))
    return _static_masks[lengths, :max_length]


def encode_list(list_of_strings):
    return b''.join(map(lambda s: str.encode(s, encoding='ascii') + b'\0', list_of_strings))
