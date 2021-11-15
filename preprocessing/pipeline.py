import os
import pickle
import argparse
import traceback
from pathlib import Path
import multiprocessing as mp

import mne

from events import check_events
from bad_channels import PREP_bads_suggestion
from filters import apply_filter_eeg, apply_filter_aux
from utils import (read_raw_fif, read_exclusion, write_exclusion,
                   raw_fif_selection, _check_path, _check_n_jobs)


mne.set_log_level('ERROR')


def prepare_raw(raw):
    """
    Prepare raw instance by checking events, adding events as annotations,
    marking bad channels, adding montage, applying FIR filters, applying CAR
    and interpolating bad channels.

    Parameters
    ----------
    raw : Raw
        Raw instance (will be modified in-place).
    bads : list
        List of interpolated bad channels.

    Returns
    -------
    raw : Raw
        Raw instance (modified in-place).
    """
    # Check sampling frequency
    if raw.info['sfreq'] != 512:
        raw.resample(sfreq=512)

    # Check events
    recording_type = Path(raw.filenames[0]).stem.split('-')[1]
    check_events(raw, recording_type)  # raise if there is a problem

    # Filter AUX
    apply_filter_aux(raw, bandpass=(1., 45.), notch=True)

    # Mark bad channels
    bads = PREP_bads_suggestion(raw)  # operates on a copy
    raw.info['bads'] = bads

    # Reference (projector) and filter EEG
    raw.add_reference_channels(ref_channels='CPz')
    raw.set_montage('standard_1020')  # only after adding ref channel
    apply_filter_eeg(raw, bandpass=(1., 45.), notch=False, car=True)

    # Interpolate bad channels
    raw.interpolate_bads(reset_bads=True, mode='accurate')

    return raw, bads


def pipeline(fname, input_dir_fif, output_dir_fif):
    """
    Pipeline function called on each raw file.

    Parameters
    ----------
    fname : str | Path
        Path to the input '-raw.fif' file to preprocess.
    input_dir_fif : str | Path
        Path to the input raw directory (parent from fname).
    output_dir_fif : str | Path
        Path used to save raw in MNE format with the same structure as in
        fname.

    Returns
    -------
    success : bool
        False if a step raised an Exception.
    fname : str
        Path to the input '-raw.fif' file to preprocess.
    bads : list
        List of interpolated bad channels.
    """
    print (f'Preprocessing: {fname}')
    try:
        # checks paths
        fname = _check_path(fname, 'fname', must_exist=True)
        input_dir_fif = _check_path(input_dir_fif, 'input_dir_fif',
                                    must_exist=True)
        output_dir_fif = _check_path(output_dir_fif, 'output_dir_fif',
                                     must_exist=True)

        # create output file name
        output_fname = _create_output_fname(fname, input_dir_fif,
                                            output_dir_fif)

        # preprocess
        raw = read_raw_fif(fname)
        raw, bads = prepare_raw(raw)
        raw.apply_proj()

        # export
        raw.save(output_fname, fmt="double", overwrite=True)

        return (True, str(fname), bads)

    except Exception:
        print ('----------------------------------------------')
        print ('FAILED: %s -> Skip.' % fname)
        print(traceback.format_exc())
        print ('----------------------------------------------')
        return (False, str(fname), None)


def _create_output_fname(fname, input_dir_fif, output_dir_fif):
    """Creates the output file name based on the relative path between fname
    and input_dir_fif."""
    # this will fail if fname is not in input_dir_fif
    relative_fname = fname.relative_to(input_dir_fif)
    # create output fname
    output_fname = output_dir_fif / relative_fname
    os.makedirs(output_fname.parent, exist_ok=True)
    return output_fname


def main(input_dir_fif, output_dir_fif, n_jobs=1, subject=None, session=None,
         fname=None):
    """
    Main preprocessing pipeline.

    Parameters
    ----------
    input_dir_fif : str | Path
        Path to the folder containing the FIF files to preprocess.
    output_dir_fif : str | Path
        Path to the folder containing the FIF files preprocessed.
    n_jobs : int
        Number of parallel jobs used. Must not exceed the core count. Can be -1
        to use all cores.
    subject : int | None
        Restricts file selection to this subject.
    session : int | None
        Restricts file selection to this session.
    fname : str | Path | None
        Restrict file selection to this file (must be inside input_dir_fif).
    """
    # check arguments
    input_dir_fif = _check_path(input_dir_fif, 'input_dir_fif',
                                must_exist=True)
    output_dir_fif = _check_path(output_dir_fif, 'output_dir_fif')
    n_jobs = _check_n_jobs(n_jobs)

    # create output folder if needed
    os.makedirs(output_dir_fif, exist_ok=True)

    # read excluded files
    exclusion_file = output_dir_fif / 'exclusion.txt'
    exclude = read_exclusion(exclusion_file)

    # list files to preprocess
    fifs_in = raw_fif_selection(input_dir_fif, output_dir_fif, exclude,
                                subject=subject, session=session, fname=fname)

    # create input pool for pipeline based on provided subject info
    input_pool = [(fname, input_dir_fif, output_dir_fif)
                  for fname in fifs_in]
    assert 0 < len(input_pool)  # sanity-check

    with mp.Pool(processes=n_jobs) as p:
        results = p.starmap(pipeline, input_pool)

    with open(output_dir_fif/'bads.pcl', mode='wb') as f:
        pickle.dump([(file, bads) for success, file, bads in results
                     if success], f, -1)

    exclude = [file for success, file, _ in results if not success]
    write_exclusion(exclusion_file, exclude)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='NeuroTin preprocessing pipeline.',
        description='Preprocess NeuroTin raw FIF files.')
    parser.add_argument(
        'input_dir_fif', type=str,
        help='folder containing FIF files to preprocess.')
    parser.add_argument(
        'output_dir_fif', type=str,
        help='folder containing FIF files preprocessed.')
    parser.add_argument(
        '--n_jobs', type=int, metavar='int',
        help='number of parallel jobs.', default=1)
    parser.add_argument(
        '--subject', type=int, metavar='int',
        help='restrict to files with this subject ID.', default=None)
    parser.add_argument(
        '--session', type=int, metavar='int',
        help='restrict with files with this session ID.', default=None)
    parser.add_argument(
        '--fname', type=str, metavar='path',
        help='restrict to this file.', default=None)

    args = parser.parse_args()

    main(args.input_dir_fif, args.output_dir_fif, args.n_jobs, args.subject,
         args.session, args.fname)
