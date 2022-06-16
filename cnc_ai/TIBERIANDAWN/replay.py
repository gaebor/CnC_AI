import argparse
from pathlib import Path
import ctypes
import numpy

from cnc_ai.TIBERIANDAWN import cnc_structs

gHandle = ctypes.windll.kernel32.GetStdHandle(ctypes.c_long(-11))


def move(y, x):
    """https://stackoverflow.com/a/27612978"""
    value = x + (y << 16)
    ctypes.windll.kernel32.SetConsoleCursorPosition(gHandle, ctypes.c_ulong(value))


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('recording', type=str, help='path to the recording (dir)')
    parser.add_argument(
        '-p', '--player', dest='player', default=0, type=int, help='which player to show'
    )
    return parser.parse_args()


def main():
    args = get_args()
    with open(Path(args.recording) / 'game_states.npy', 'rb') as f:
        game_states = numpy.load(f, allow_pickle=True)
    for game_state in game_states:
        move(0, 0)
        print(cnc_structs.render_game_state_terminal(game_state[args.player]))


if __name__ == '__main__':
    main()
