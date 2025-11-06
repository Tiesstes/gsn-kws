import os

import custom_utils as utils

import soundfile # necessary for torch in Windows to handle audio files; torchcodec does not work


import torch
import torchaudio
import torchaudio.transforms as transforms
import torchaudio.functional as functionals

from torch.utils.data import DataLoader

from torchaudio.datasets import SPEECHCOMMANDS


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.__version__)
print(torchaudio.__version__)

# Ściągamy obecną ścieżkę, gdzie znajduje się skrypt, żeby pobrać do niej Dataset
dataset_path = os.path.abspath(os.getcwd())

# WARNING: To download the dataset, please change the parameter download to True
train_data = SPEECHCOMMANDS(root=dataset_path, download=False, subset="training")
validation_data = SPEECHCOMMANDS(root=dataset_path, download=False, subset="validation")
test_data = SPEECHCOMMANDS(root=dataset_path, download=False, subset="testing")


# TODO: dataset processing with DataLoaders
batch_size = 16

#train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#valid_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


print(f"The training dataset length is: {train_data.__len__()}")
i = 3
# WARNING: The normal train_data[0] does not work at the local machine, hence the direct call for .__getitem__(i)
waveform, sample_rate, label, speaker_id, utterance_number = train_data.__getitem__(i)

utils.plot_waveform_pytorch(waveform, sample_rate, label)

# NOTE: Play with the parameters to see the difference in Mel Spectrogram
n_fft = 512
win_length = None
hop_length = 160
n_mels = 64

transform_ms = transforms.MelSpectrogram(
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

functional_mf = functionals.melscale_fbanks(n_freqs=int(n_fft // 2 + 1), n_mels=n_mels,
    f_min=0.0,
    f_max=sample_rate / 2.0,
    sample_rate=sample_rate,
    norm=None)

melspec = transform_ms(waveform)
utils.plot_spectrogram_pytorch(melspec[0], "Mel-Spectrogram", "mel_freq")

utils.plot_fbank_pytorch(functional_mf, "Mel Filter Banks")

