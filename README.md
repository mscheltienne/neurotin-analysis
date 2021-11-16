# NeuroTinAnalysis

Scripts and programs to analyze the NeuroTin EEG dataset.
The RAW dataset folder structure is set to:

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

## model

`model` contains scripts to analyze the models used in the neurofeedback
training.

## preprocessing

`preprocessing` contains scripts to clean the raw data and to fill missing
information. The pipeline is splits in different steps creating intermediate
files.

### Step 1

- Rename channels and fix channel types.
- Checks sampling frequency and resample to 512 Hz if needed.
- Checks events and add events as annotations.
- Filter AUX with FIR bandpass (1., 45.) Hz and notch filter for powerline.
- Mark bad channels with the PREP pipeline.
- Add reference channel `CPz` and set standard 10/20 montage.
- Filter EEG with FIR bandpass (1., 45.) Hz.
- Add common average reference (CAR) projector.
- Interpolate bad channels.
- Apply common average reference (CAR) projector.

It can be called via command-line with `python prepare_raw.py`. Help for the
arguments can be obtained with the `--help` flag.

### Step 2

- Apply ICA and remove occular and heartbeat related components.

It can be called via command-line with `python ica.py`. Help for the arguments
can be obtained with the `--help` flag.

### Step 3

- Fill `.info['description']`, `.info['device_info']`, `.info['experimenter']`,
  `.info['meas_date']`, `.info['subject_info']`.

It can be called via command-line with `python meas_info.py`. Help for the
arguments can be obtained with the `--help` flag.
Note that this last step can be operated in-place, overwriting existing files.

## time-frequency

`time-frequency` contains scripts for sensor-level time-frequency analysis,
e.g. PSD computation using welch method or multitapers.

## evamed

`evamed` contains scsripts to parse and analyze the evamed questionnaires.
The questionnaires are provided as an exported compact .csv file.
