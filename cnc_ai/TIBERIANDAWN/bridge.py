from typing import List, Union
from dataclasses import dataclass
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
    dynamic_mask: NDArray[bool] = None
    sidebar_mask: NDArray[bool] = None

    def __post_init__(self):
        if self.dynamic_mask is None:
            self.dynamic_mask = numpy.zeros_like(self.AssetName, dtype=bool)
        if self.sidebar_mask is None:
            self.sidebar_mask = numpy.zeros_like(self.SidebarAssetName, dtype=bool)

    def apply(self, f):
        return GameState(**{member: f(value) for member, value in self.__dict__.items()})


@dataclass
class GameAction:
    button: NDArray[bool]
    mouse_x: NDArray[NpFloat]
    mouse_y: NDArray[NpFloat]

    def apply(self, f):
        return GameAction(**{member: f(value) for member, value in self.__dict__.items()})


def concatenate_game_states(list_of_game_states: List[GameState]) -> GameState:
    result = {
        **{
            key: pad_sequence(
                [getattr(game_state, key) for game_state in list_of_game_states],
                padding_value=True,
            )
            for key in ['dynamic_mask', 'sidebar_mask']
        },
        **{
            key: numpy.stack([getattr(game_state, key) for game_state in list_of_game_states], 0)
            for key in ['StaticAssetName', 'StaticShapeIndex', 'SidebarInfos']
        },
        **{
            key: pad_sequence([getattr(game_state, key) for game_state in list_of_game_states])
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
    return GameState(**result)


def concatenate_game_actions(list_of_game_actions: List[GameAction]) -> GameAction:
    button_actions = pad_sequence([action.button for action in list_of_game_actions])
    scalar_action = len(list_of_game_actions[0].button.shape) == 2
    merge_function = numpy.array if scalar_action else numpy.concatenate
    mouse_x = merge_function([action.mouse_x for action in list_of_game_actions])
    mouse_y = merge_function([action.mouse_y for action in list_of_game_actions])
    return GameAction(button_actions, mouse_x, mouse_y)


def encode_list(list_of_strings):
    return b''.join(map(lambda s: str.encode(s, encoding='ascii') + b'\0', list_of_strings))
