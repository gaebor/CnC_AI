import numpy
from torch.nn.utils.rnn import pad_sequence as pad_sequence_torch
from torch import tensor


def pad_sequence(tensors):
    padded_tensors = pad_sequence_torch([tensor(t) for t in tensors], batch_first=True).numpy()
    return padded_tensors


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


def pad_game_states(list_of_game_states):
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
