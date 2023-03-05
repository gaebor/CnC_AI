import argparse


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--port', default=8888, type=int, help=' ')
    parser.add_argument(
        '-n', '--n', default=2, type=int, help='number of games to play simultaneously'
    )
    parser.add_argument(
        '-l',
        '--limit',
        '--end-limit',
        '--end_limit',
        dest='end_limit',
        default=10_000,
        type=int,
        help='stop game after this many iterations if still going',
    )
    parser.add_argument(
        '-T',
        '--time-window',
        default=200,
        type=int,
        help='limits the maximum depth in time to backpropagate to',
    )
    parser.add_argument(
        '--exe', default='TIBERIANDAWN_wrapper.exe', metavar='absolute path', help=' '
    )
    parser.add_argument(
        '--dir',
        default=r'C:\Program Files (x86)\Steam\steamapps\common\CnCRemastered',
        metavar='absolute path',
        help='The game installation directory. '
        'Buy the game on Steam: https://store.steampowered.com/app/1213210',
    )
    parser.add_argument(
        '--dll',
        default='TiberianDawn.dll',
        help='Path to the game DLL, absolute or relative from `dir`. '
        'This DLL is released with the game itself but it was opensourced too: '
        'https://github.com/electronicarts/CnC_Remastered_Collection',
    )
    parser.add_argument('-d', '--device', default='cpu', help='pytorch device')
    parser.add_argument(
        '--load',
        default='',
        help='Load model from pytorch model file '
        'rather than initializing a new one on the beginning.',
    )
    parser.add_argument(
        '--save', default='', help='Save model to pytorch model file after the game(s).'
    )
    parser.add_argument(
        '-D',
        dest='spawn',
        default=True,
        action='store_false',
        help="Don't spawn subprocesses to run the DLL (set by '--exe'). "
        "This is good for debugging when you start the dll wrapper separately.",
    )
    parser.add_argument(
        '-P',
        '--print',
        default=False,
        action='store_true',
        help="Print out what was the terminal state of the game(s).",
    )
    parser.add_argument('--train', default=0, type=int, help="Train at the end of the game.")
    parser.add_argument(
        '-r',
        '--record',
        default=False,
        action='store_true',
        help="Save recording of the games for later inspection or training. "
        "If you train at the end of the games than save recording may not be necessary.",
    )
    parser.add_argument(
        '-a',
        '--agents',
        dest='agents',
        default='AIvAI',
        choices=['AIvAI', 'AIvNN', 'NNvAI', 'NNvNN'],
        help=" ",
    )
    parser.add_argument(
        '-lr',
        '--learning-rate' '--learning_rate',
        dest='learning_rate',
        default=1e-6,
        type=float,
        help=" ",
    )
    parser.add_argument(
        '-H', '--half', default=False, action='store_true', help="Use float16 during inference"
    )

    return parser.parse_args()
