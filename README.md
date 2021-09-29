# NeuroTinAnalysis

Scripts and programs to analyze the NeuroTin EEG dataset.
The dataset folder structure is set to:

```
Data
└─ 001
    └─ Session 1
    └─ Session 2
        └─ Calibration
        └─ Model
        └─ Online
        └─ Plots
        └─ RestingState
        └─ bads.txt
        └─ logs.txt
    └─ ...
    └─ Session 15

└─ 002
└─ ...
```

4 `.csv` files are used to log different variables for every participant and
session:

- `mml_logs.csv`: result of the Minimum Masking Level test.
- `model_var_logs.csv`: helmet size (54, 56, 58), model normalization variables
  and bad channels.
- `scores_logs.csv`: neurofeedback scores displayed.
- `sound_stimulus_logs.csv`: sound stimulus settings used.

## logs

`logs` contains scripts to process and analyze the 4 logging `.csv` files.

## preprocessing

`preprocessing` contains scripts to clean the raw data and to fill missing
information. The automatic pipeline does:

- Rename channels and fix channel types.
- Fill `.info['description']`, `.info['device_info']`, `.info['experimenter']`,
  `.info['meas_date']`, `.info['subject_info']`.
- Checks events and add events as annotations.
- Mark bad channels with the PREP pipeline.
- Add reference channel `CPz` and set standard 10/20 montage.
- Filter with FIR bandpass (1., 40.) Hz.
- Apply common average reference (CAR).
- Interpolate bad channels.
- Apply ICA and remove components correlated to EOG and ECG channels.

It can be called via command-line with `python pipeline.py`. Help for the
arguments can be obtained with the `--help` flag.

## TODO

- [ ] Evamed database analysis
- [ ] Sensor Space time-frequency analysis
- [ ] Source space analysis with `fsaverage` template MRI
