import argparse

from matplotlib import pyplot as plt

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
        '-p',
        help='participant id(s) to include.', nargs='+', required=True)

    args = parser.parse_args()
    plot_mml_across_participants(args.csv, [int(x) for x in args.p])
    plt.show(block=True)
