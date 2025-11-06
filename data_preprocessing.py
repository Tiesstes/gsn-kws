import os
from pickle import GLOBAL

import custom_utils as utils

import numpy
import soundfile # necessary for torch in Windows to handle audio files; torchcodec does not work


import torch
import torchaudio
import torchaudio.transforms as transforms

from torch.utils.data import DataLoader

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


print(f"The training dataset length is: {train_data.__len__()}")
i = 3
# The normal train_data[0] does not work at the local machine, hence the direct call for .__getitem__(i)
waveform, sample_rate, label, *_ = train_data.__getitem__(i)

# Play with the parameters to see the difference in Mel Spectrogram

n_fft = 1024
win_length = None
hop_length = 160
n_mels = 128

transform = transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm=None,
    n_mels=n_mels,
    mel_scale="htk",
)

melspec = transform(waveform)
utils.plot_spectrogram_pytorch(melspec[0], "Mel-Spectrogram", "mel_freq")

