# Deep learning project: Sleep stage classifiction using EEG signals

## Task
Analyzing sequential data or time series is a very relevant and explored task in Deep Learning.
This kind of data appears in many domains and different formats; for example, stock prices,
videos and electrophysiological signals.
In this task, we worked with Electroencephalography (EEG) or brain waves. The data set consists of EEG signals of different subjects while sleeping. There are different stages of sleep characterized by specific kinds of EEG signals. Each sleep stage is then a period during which the EEG signals present specific features or patterns.
The task is to classify EEG signals into sleep stages.

## Data

The data-set consists of EEG sequences of 3000-time steps each and coming from two electrode
locations on the head (Fpz-Cz and Pz-Oz) sampled at 100 Hz. That means that each sample contains
two signals of 3000 samples and that those samples correspond to 30 seconds of recording.
The labels that come along with the data specify six stages labelling them with corresponding
numbers as specified in below table:

### Label Stage Typical Frequencies (Hz)

| Label | Stage | Typical Frequencies (Hz) |
|-------|-------|--------------------------|
| 0     | R     | 15-30                    |
| 1     | 1     | 4-8                      |
| 2     | 2     | 8-15                     |
| 3     | 3     | 1-4                      |
| 4     | 4     | 0.5-2                    |
| 5     | W     | 15-50                    |

W corresponds to the Wake stage, and R to REM sleep also called rapid eye movement, and most commonly known as the dreaming stage. Each sequence in the data set contains only one stage, which is specified by the corresponding
label.
The data set is presented in two different formats, Raw signals and Spectrograms.


