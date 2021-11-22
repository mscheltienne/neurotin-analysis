# NeuroTin

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
- `sound_stimulus_logs.csv`: auditory stimulus settings used.

For functions part of the CLI interface, help for the arguments can be obtained
with the `--help` flag.

## logs

`neurotin.logs` contains scripts to process and analyze the 4 logging `.csv`
files.

#### CLI

- `neurotin_logs_mml`: Plot MML for a list of participants.

## model

`neurotin.model` contains scripts to analyze the models used in the
neurofeedback training.

## preprocessing

`neurotin.preprocessing` contains scripts to clean the raw data and to fill
missing information. The pipeline is splits in different steps creating
intermediate files with the same folder structure as the input folder.

### Step 1: `neurotin.preprocessing.prepare_raw`

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

### Step 2: `neurotin.preprocessing.ica`

- Apply ICA and remove occular and heartbeat related components.

Thresholds and methods to select ocular and heartbeat related components have
been adapted for this dataset.

TODO:
- [ ] Replace with a python implementation of ICLabel

### Step 3: `neurotin.preprocessing.meas_info`

- Fill `.info['description']`, `.info['device_info']`, `.info['experimenter']`,
  `.info['meas_date']`, `.info['subject_info']`.

Description includes `subject id`, `session`, `recording type` and
`recording run`.
Device information includes `type`, `model`, `serial` and `website`.
Exprimenter includes the experimenter name.
Measurement date includes the recording datetime (UTC).
Subject information includes `subject id`, `birthday` (optional) and
`sex` (optional).

## time-frequency

`time-frequency` contains scripts for sensor-level time-frequency analysis,
e.g. PSD computation using welch method or multitapers.

## evamed

`evamed` contains scsripts to parse and analyze the evamed questionnaires.
The questionnaires are provided as an exported compact .csv file.

## CLI

- `neurotin_pp_prepare_raw`
- `neurotin_pp_ica`
- `neurotin_pp_meas_info`

## I/O

`neurotin.io` contains functions for I/O operations.
