# NeuroTin (EEG analysis)

NeuroTin is a clinical trial under the supervision of principal investigator
Prof. Dr. Pascal Senn (HUG, Geneva) and supported by the Wyss Center.
NeuroTin aims to compare tinnitus reduction after 3 different therapeutic
approaches:

- Cognitive Behavioral Therapy (CBT), the current gold-standard treatment
- Neurofeedback with electroencephalography (EEG)
- Neurofeedback with functional magnetic resonance imaging (fMRI)

This repository contains the python implementation of the Neurofeedback
paradigm using electroencephalography. Each session is articulated around 3
main steps: calibration, model, and neurofeedback.

- The calibration uses an auditory stimuli to elicit an N1-P2 evoked response.
- A model applies weights between 0 and 1 to each electrode based on the
  N1-P2 evoked response.
- Neurofeedback runs alternate between phases of non-regulation (also called
  rest) lasting 8 seconds, and phases of regulation lasting 16 seconds. During
  phases of regulation, participants attempt to up-regulate the ratio of
  alpha-band power over delta-band power displayed in real-time.

The implementation of the neurofeedback paradigm using EEG can be found on this
[repository](https://github.com/mscheltienne/neurotin-eeg).

---

## Dataset structure

The raw dataset folder structure is defined as:

```
> Data
> └─ 001
>     └─ Session 1
>     └─ Session 2
>         └─ Calibration
>         └─ Model
>         └─ Online
>         └─ Plots
>         └─ RestingState
>         └─ bads.txt
>         └─ logs.txt
>     └─ ...
>     └─ Session 15
> └─ 002
> └─ ...
```

4 `.csv` files are used to log different variables for every participant and
session:

- `mml_logs.csv`: results of the Minimum Masking Level test repeated at every
  session.
- `sound_stimulus_logs.csv`: auditory stimulus settings used for calibration.
- `model_var_logs.csv`: helmet size (54, 56, 58), model normalization variables
  and bad channels.
- `scores_logs.csv`: neurofeedback scores displayed.

---

## Command-line interface

Many analysis function produces either (pre)processed data files or pandas
DataFrame. A list of functions accessible via the CLI can be displayed with the
command `neurotin`. Help for those functions can be obtained with the `--help`
flag.
