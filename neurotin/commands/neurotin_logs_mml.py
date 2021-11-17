import argparse

from neurotin.logs import plot_mml_across_participants


def run():
    """Entrypoint for neurotin_logs_mml."""
    parser = argparse.ArgumentParser(
        prog='NeuroTin',
        description='Analyze minimum masking level .csv file.')
    parser.add_argument(
        'csv', type=str,
        help='path to the .csv file containing the MML logs.',  required=True)
    parser.add_argument(
        'participants',
        help='participant id(s) to include.', nargs='+', required=True)

    args = parser.parse_args()
    plot_mml_across_participants(args.csv, [int(x) for x in args.participants])
