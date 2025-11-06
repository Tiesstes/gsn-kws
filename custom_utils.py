# NOTE: functions below are utility functions. Ones named with _pytorch are implemented according to PyTorch docs
import librosa
import numpy
import torch
from matplotlib import pyplot as plt


def plot_spectrogram_basic(waveform_data, sample_rate_data, title="Spectrogram"):
    waveform_data = waveform_data.numpy()
    figure, ax = plt.subplots(figsize=(9, 7))
    ax.specgram(waveform_data[0], Fs=sample_rate_data, cmap="magma")
    figure.suptitle(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.show()

def plot_waveform_basic(waveform_data, sample_rate_data, title="Waveform"):

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


def plot_waveform_pytorch(waveform, sample_rate, title="Waveform"):

    waveform_data = waveform.numpy()

    channels, samples = waveform_data.shape
    time_axis = torch.arange(0, samples) / sample_rate

    figure, axes = plt.subplots(channels, 1, figsize=(9, 7))
    axes.plot(time_axis, waveform_data[0], linewidth=1)
    axes.grid(True)
    figure.suptitle(title)
    plt.show(block=False)


def plot_spectrogram_pytorch(spectrogram, title=None, ylabel="freq_bin"):

    fig, axs = plt.subplots(figsize=(9,7))
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spectrogram), origin="lower", aspect="auto", cmap="magma")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)

def plot_fbank_pytorch(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto", cmap="magma")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")
    plt.show(block=False)