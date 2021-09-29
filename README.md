# NeuroTinAnalysis

Scripts and programs to analyze the NeuroTin EEG dataset.
The NeuroTin raw dataset folder structure is set to:

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
