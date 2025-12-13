import os
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS

from dataset import SpeechCommandsKWS

from project.model.arch.blocks import BCResBlock
from project.model.arch.blocks import ConvBNReLU
from project.model.kws_net import KWSNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.__version__)
print(torchaudio.__version__)


# konfiguracja
BASE_DATASET_DIR = os.path.abspath(os.getcwd())
NOISE_PATH = Path(BASE_DATASET_DIR) / "SpeechCommands" / "speech_commands_v0.02" / "_background_noise_"
BATCH_SIZE = 32

print("Ścieżka, w której znajduje się katalog pobranego bazowego dataset'u:" , BASE_DATASET_DIR)
# podzbiory:
train_base = SPEECHCOMMANDS(root=BASE_DATASET_DIR, subset="training")
#val_base   = SPEECHCOMMANDS(root=dataset_dir, subset="validation")
#test_base  = SPEECHCOMMANDS(root=dataset_dir, subset="testing")

# datasety prawdziwe:
train_dataset = SpeechCommandsKWS(train_base, NOISE_PATH)
#val_dataset  = SpeechCommandsKWS(val_base,   label_mapping, speaker_mapping)
#test_dataset  = SpeechCommandsKWS(test_base,  label_mapping, speaker_mapping)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

model = KWSNet(12, train_dataset.speaker_counter.__len__())
model = model.to(device)

def check_forward(data_loader, model):
    model.eval()
    batch = next(iter(data_loader))

    x = batch["log_mel_spectrogram"].to(device)   # Tensor: [B, 1, F, T] u Ciebie
    spk = batch["speaker_id"].to(device)           # LongTensor: [B]

    with torch.no_grad():
        y = model(x, spk)

    assert list(y.shape) == [x.shape[0], 12]  # [B, num_classes]
    
def check_backward(data_loader, model):
    model.train()
    batch = next(iter(data_loader))

    x = batch["log_mel_spectrogram"].to(device)
    spkr_id = batch["speaker_id"].to(device)
    y_true = batch["label"].to(device)             # [B]

    logits = model(x, spkr_id)
    loss = torch.nn.CrossEntropyLoss()(logits, y_true)
    loss.backward()

    assert model.speaker_embedding.weight.grad is not None


if __name__ == "__main__":
    check_forward(train_loader, model)
    check_backward(train_loader, model)