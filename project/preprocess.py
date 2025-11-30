import os

import torch
import torchaudio
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS

from dataset import SpeechCommandsKWS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.__version__)
print(torchaudio.__version__)


# konfiguracja
dataset_dir = os.path.abspath(os.getcwd())
BATCH_SIZE = 32

print(dataset_dir)
# podzbiory:
train_base = SPEECHCOMMANDS(root=dataset_dir, subset="training")
val_base   = SPEECHCOMMANDS(root=dataset_dir, subset="validation")
test_base  = SPEECHCOMMANDS(root=dataset_dir, subset="testing")

# mapowanie etykiet i mówcy
all_labels = []
all_speakers = []

for _, _, label, speaker, _ in train_base:
    all_labels.append(label)
    all_speakers.append(speaker)

# usuwamy duplikaty wartości i sortujemy
all_labels = sorted(set(all_labels))
all_speakers = sorted(set(all_speakers))

# robimy mapy: string -> int
label_mapping = {}
for i, lbl in enumerate(all_labels):
    label_mapping[lbl] = i

speaker_mapping = {}
for i, s_id in enumerate(all_speakers):
    speaker_mapping[s_id] = i

# datasety prawdziwe:
train_dataset = SpeechCommandsKWS(train_base, label_mapping, speaker_mapping)
val_dataset  = SpeechCommandsKWS(val_base,   label_mapping, speaker_mapping)
test_dataset  = SpeechCommandsKWS(test_base,  label_mapping, speaker_mapping)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

batch = next(iter(train_loader)) # wzięcie pierwszego batcha z dataloadera
x = batch["log_mel_spectrogram"]   # [Batch, 1, 40, ilość ramek czasowych (1s: 30ms okno i 10ms przesunięcie, czyli 0-30ms potem 10-40ms...)]
y = batch["label"]                 # [Batch]
s = batch["speaker_id"]            # [Batch]

print(x.shape, y.shape, s.shape)

