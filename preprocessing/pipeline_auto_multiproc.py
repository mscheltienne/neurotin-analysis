import multiprocessing

from utils import list_raw_fif
from pipeline_auto import (FOLDER_IN, FOLDER_OUT, preprocessing_pipeline,
                           ICA_pipeline)


def pipeline(fname, participant, sex):
    raw = preprocessing_pipeline(fname)
    raw = ICA_pipeline(raw)
    raw.info['subject_info']['sex'] = sex
    raw.info._check_consistency()
    fif_out = FOLDER_OUT / fname.relative_to(FOLDER_IN)
    raw.save(fif_out, fmt="double")


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
    fifs = list_raw_fif(FOLDER_IN)

if __name__ == '__main__':
    pass
