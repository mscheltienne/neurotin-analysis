import os
import re
import pickle
from datetime import datetime

from ..utils.checks import _check_path, _check_type


def write_results(results, results_file):
    """
    Write results from CLI call to pickle. Each entry in results is a tuple
    starting with (bool, str, ...). The bool in position 0 is set to False when
    an error was raised during processing. The str in position 1 is the fname
    processed.

    Parameters
    ----------
    results : list of tuples
        (bool, str, ...) containing the results from the _pipeline functions.
    results_file : str | Path
        Path to the .pcl file where the results are pickled. The datetime is
        appended to the file name stem.
    """
    # check results
    _check_type(results, (list, ), item_name='results')
    for result in results:
        _check_type(result, (tuple, ))
        _check_type(result[0], (bool, ))
        _check_type(result[1], ('path-like', ))
    # check results_file
    results_file = _check_path(results_file, item_name='results_file')
    assert results_file.suffix == '.pcl'
    directory = results_file.parent
    os.makedirs(directory, exist_ok=True)
    appendix = datetime.now().strftime("_%Hh-%Mmn-%d-%m-%Y")
    results_file = directory / str(results_file.stem) + appendix + '.pcl'
    results_file = _check_path(results_file, item_name='results_file')

    # save
    with open(results_file, 'wb') as f:
        pickle.dump(results, f, -1)


def read_results(results_file, *, success_only=False, failure_only=False):
    """
    Read results of CLI call. Each result in the list of results starts with
    (bool, str, ...). The bool in position 0 is set to False when an error was
    raised during processing. The str in position 1 is the fname processed.

    Parameters
    ----------
    results_file : str | Path
        Path to the .pcl file where the results are pickled. The datetime is
        appended to the file name stem.
    success_only : bool
        If True, returns only result entries that were a success.
    failure_only : bool
        If True, returns only results entries that were a failure.

    Returns
    -------
    results : list of tuples
        (bool, str, ...) containing the results from the _pipeline functions.
    date : datetime
        Date and time (hour) at which the results file has been pickled.
    """
    results_file = _check_path(results_file, item_name='results_file',
                               must_exist=True)
    assert results_file.suffix == '.pcl'

    # extract datetime
    pattern = re.compile(r'\d{2}h-\d{2}mn-\d{1,2}-\d{1,2}-\d{4}')
    dates = re.findall(pattern, results_file.stem)
    assert len(dates) == 1
    date = datetime.strptime(dates[0], '%Hh-%Mmn-%d-%m-%Y')

    # read results
    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    # filter
    if success_only:
        results = list(filter(lambda x:x[0], results))
    if failure_only:
        results = list(filter(lambda x: not x[0], results))

    return results, date
