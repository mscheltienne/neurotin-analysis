import os
import traceback
import multiprocessing as mp

from utils import list_raw_fif, write_exclusion
from pipeline_auto import (FOLDER_IN, FOLDER_OUT, preprocessing_pipeline,
                           ICA_pipeline, read_exclusion)


PROCESSES = 3  # Number of parallel processes


def pipeline(fname, sex):
    """
    Pipeline function called by each process.
    """
    print (f'Preprocessing: {fname}')
    try:
        raw = preprocessing_pipeline(fname)
        raw = ICA_pipeline(raw)
        raw.info['subject_info']['sex'] = sex
        raw.info._check_consistency()
        fif_out = FOLDER_OUT / fname.relative_to(FOLDER_IN)
        os.makedirs(fif_out.parent, exist_ok=True)
        raw.save(fif_out, fmt="double")
        return (True, fname)
    except AssertionError:
        print (f'FAILED: {fname}')
        print(traceback.format_exc())
        return (False, fname)


def main(participant_info):
    """
    Main preprocessing pipeline calling a pool of workers on the fif files to
    preprocess.

    Parameters
    ----------
    participant_info : dict
        key : int
            ID of the participant
        value : int
            Sex of the participant. 1: Male - 2: Female.
    """
    os.makedirs(FOLDER_OUT, exist_ok=True)
    exclusion_file = FOLDER_OUT / 'exclusion.txt'
    exclude = read_exclusion(exclusion_file)

    fifs_in = list_raw_fif(FOLDER_IN)
    fifs_out = list_raw_fif(FOLDER_OUT)

    # Remove already preprocessed files and excluded files
    fifs_out_relative = [fif.relative_to(FOLDER_OUT) for fif in fifs_out]
    fifs_in = [fif for fif in fifs_in
               if fif.relative_to(FOLDER_IN) not in fifs_out_relative]
    fifs_in = [fif for fif in fifs_in
               if FOLDER_OUT / fif.relative_to(FOLDER_IN) not in exclude]

    # create input pool for pipeline based on provided participant info
    participants = [int(fname.parent.parent.parent.name) for fname in fifs_in]
    input_pool = [(fifs_in[k], participant_info[participant])
                  for k, participant in enumerate(participants) \
                      if participant in participant_info]

    with mp.Pool(processes=PROCESSES) as p:
        results = p.starmap(pipeline, input_pool)

    exclude = [fname for success, fname in results if not success]
    write_exclusion(exclusion_file, exclude)

if __name__ == '__main__':
    participant_info = {
        80: 1}
    main(participant_info)
