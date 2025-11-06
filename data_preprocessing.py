import os

import numpy
import soundfile as sf
import matplotlib.pyplot as plt


import torch
import torchaudio
from torch.utils.data import DataLoader
from torchaudio.compliance.kaldi import spectrogram
from torchaudio.datasets import SPEECHCOMMANDS


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.__version__)
print(torchaudio.__version__)

# Ściągamy obecną ścieżkę, gdzie znajduje się skrypt, żeby pobrać do niej Dataset
dataset_path = os.path.abspath(os.getcwd())

train_data = SPEECHCOMMANDS(root=dataset_path, download=False, subset="training")
validation_data = SPEECHCOMMANDS(root=dataset_path, download=False, subset="validation")
test_data = SPEECHCOMMANDS(root=dataset_path, download=False, subset="testing")

# TODO: dataset processing with DataLoaders

#train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
#valid_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
#test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)



# NOTE: functions below are utility functions. To be moved

def plot_spectrogram(waveform_data, sample_rate_data, title="Spectrogram"):
    waveform_data = waveform_data.numpy()
    figure, ax = plt.subplots(figsize=(9, 7))
    ax.specgram(waveform_data[0], Fs=sample_rate_data, cmap="magma")
    figure.suptitle(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.show()

def plot_waveform(waveform_data, sample_rate_data, title="Waveform"):

    waveform_data = waveform_data.numpy()
    print(f"First dimension of 'waveform_data' is channel, second is the samples (and their values): {waveform_data.shape}")
    channel, samples = waveform_data.shape

    time = numpy.arange(samples)/sample_rate_data

    figure, ax = plt.subplots(figsize=(9, 7))
    ax.plot(time, waveform_data[0], linewidth=2)
    figure.suptitle(title)
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()



print(f"The training dataset length is: {train_data.__len__()}")
i = 3
# The normal train_data[0] does not work at the local machine, hence the direct call for .__getitem__(i)
waveform, sample_rate, label, *_ = train_data.__getitem__(i)
plot_spectrogram(waveform, sample_rate, title=f"Sample {i}: {label}")
plot_waveform(waveform, sample_rate, title=f"Sample {i}: {label}")

