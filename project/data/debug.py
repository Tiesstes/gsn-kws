import time

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS


from dataset import SpeechCommandsKWS
from project.data.dataset import IndexBuilder
from project.model.kws_net import KWSNet

import warnings
# bo głupi torchaudio krzyczy
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torchaudio"
)

# TODO: walidacja, testowanie
# TODO: automatyzacja

"""
Tutaj jest skrypcik kontrolny. W zasadzie to samo co preprocess.py, tylko ma mniej dataset'ów
"""

# konfiguracja
BASE_PATH = Path(__file__).resolve().parent.parent # katalog /project
GSC_PATH = Path(__file__).resolve().parent # katalog /project/data
NOISE_PATH = Path(GSC_PATH) / "SpeechCommands" / "speech_commands_v0.02" / "_background_noise_"
TRAINED_WEIGHTS_PATH = Path(BASE_PATH) / "model" / "KWSNet_weights.pt"

# do treningu
EPOCHS = 22
BATCH_SIZE = 128
WORKERS = 0 # bo na moim komputerze Windows tego nie ogarnia (i ja też nie)
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0 # opcjonalnie
SCHEDULER_STEP_SIZE = 4 # zmiana lr pomaga z tego co sprawdzałam
SCHEDULER_GAMMA = 0.5


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA DEVICE:", torch.cuda.get_device_name(0))
    print("Torch version:", torch.__version__)
    print("")

    # podzbiory od razu:
    base_data = SPEECHCOMMANDS(root=GSC_PATH, subset=None, download=True)

    dataset_indexer = IndexBuilder(base_data)

    indexes = dataset_indexer.build_finetune_splits()

