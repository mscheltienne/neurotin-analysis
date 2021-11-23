import argparse

from neurotin.preprocessing.validation.ica import random_plot_sources


def run():
    """
    Entrypoint for neurotin.preprocessing.validation.ica.random_plot_sources
    """
    parser = argparse.ArgumentParser(
        prog='NeuroTin',
        description='Randomly pick a fitted ICA and the RAW file used to fit '
                    'and plot the sources.')
    parser.add_argument(
        'prepare_raw_dir', type=str,
        help='folder where FIF files processed used to fit the ICA are '
             'stored.')
    parser.add_argument(
        'ica_raw_dir', type=str,
        help='folder where FIF files processed on which ICA has been applied '
             'are stored.')

    args = parser.parse_args()

    random_plot_sources(args.prepare_raw_dir, args.ica_raw_dir)
