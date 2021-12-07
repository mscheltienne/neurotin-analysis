import argparse

from matplotlib import pyplot as plt

from neurotin import set_log_level
from neurotin.commands import helpdict
from neurotin.logs import plot_mml_across_participants


def run():
    """Entrypoint for neurotin.logs.plot_mml_across_participants"""
    parser = argparse.ArgumentParser(
        prog='NeuroTin',
        description='Analyze minimum masking level .csv file.')
    parser.add_argument(
        'csv', type=str,
        help='path to the .csv file containing the MML logs.')
    parser.add_argument(
        '-p', '--participants',
        help=helpdict['participants'], nargs='+', required=True)
    parser.add_argument(
        '--loglevel', type=str, metavar='str', help=helpdict['loglevel'],
        default='info')

    args = parser.parse_args()
    set_log_level(args.loglevel.upper().strip())

    plot_mml_across_participants(args.csv, [int(x) for x in args.p])
    plt.show(block=True)
