"""Command-line utilities. Inspired from MNE."""

# common docstrings for CLI
helpdict = dict()
helpdict['input_dir_fif'] = 'folder where FIF files to process are stored.'
helpdict['output_dir_fif'] = 'folder where processed FIF files are saved.'
helpdict['output_dir_fif_with_None'] = \
    'folder where processed FIF files are saved (can be None to overwrite ' + \
    'existing files in input_dir_fif).'
helpdict['output_dir_set'] = \
    'folder where converted FIF files are saved to SET.'
helpdict['n_jobs'] = 'number of parallel jobs to run. -1 to use all cores.'
helpdict['participant'] = \
    'restrict processing to files with this participant ID.'
helpdict['session'] = 'restrict processing to files with this session ID.'
helpdict['fname'] = 'restrict processing to this file.'
helpdict['ignore_existing'] = \
    'ignore files already processed and saved in output_dir_fif.'
helpdict['loglevel'] = 'set the log level to one of info, warning, debug.'
helpdict['participants'] = 'participant id(s) to include.'
