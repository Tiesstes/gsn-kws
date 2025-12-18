import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS

from torchinfo import summary

from dataset import SpeechCommandsKWS
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

print("Ścieżka, w której znajduje się katalog pobranego bazowego dataset'u:", GSC_PATH)

# podzbiory od razu:
train_base = SPEECHCOMMANDS(root=GSC_PATH, subset="training", download=True)
val_base   = SPEECHCOMMANDS(root=GSC_PATH, subset="validation", download=True)
test_base  = SPEECHCOMMANDS(root=GSC_PATH, subset="testing", download=True)

# datasety prawdziwe (customowe):
train_dataset = SpeechCommandsKWS(train_base, NOISE_PATH)
val_dataset  = SpeechCommandsKWS(val_base, NOISE_PATH)
test_dataset  = SpeechCommandsKWS(test_base, NOISE_PATH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)


if __name__ == "__main__":

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA DEVICE:", torch.cuda.get_device_name(0))
    print("Torch version:", torch.__version__)
    print("")


    # jednak z góry mówimy ile jest speaker'ów, może do poprawy potem
    model = KWSNet(12, train_dataset.speaker_counter.__len__())
    model = model.to(DEVICE)

    print("[Architektura sieci]")
    summary(model,
        input_data=(
            torch.randn(BATCH_SIZE, 1, 40, 101).to(DEVICE),  # tu log_mel_spectrogram: [B, 1, F, T]
            torch.zeros(BATCH_SIZE, dtype=torch.long).to(DEVICE)  # tu speaker_id: [B]
        ),
        col_names=("input_size", "output_size", "num_params", "kernel_size"),
        depth=5)

    criterion = torch.nn.CrossEntropyLoss()

    # weights decay to regularyzacja L2 - karze duże wagi;
    # czyli dodaje karę do straty jako λ * w^2 (to się dziele w optymalizatorze),
    # a ta λ = weight_decay
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # taki prosty co tylko mnoży globalny LR, co określoną liczbę epok
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimiser, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

    epoch = 1

    for epoch_idx in range(EPOCHS):
        print(f"I'm alive. Now starting epoch n: {epoch_idx + 1}")

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        epoch_start = time.time()

        for batch in train_loader:

            x = batch["log_mel_spectrogram"].to(DEVICE)  # ma być [B, 1, C, T]
            speaker = batch["speaker_id"].long().to(DEVICE)
            y = batch["label"].long().to(DEVICE)

            optimiser.zero_grad() # zeruj gradienty, bo przecież to nowy batch
            logits = model(x, speaker)  # (B, num_classes)

            loss = criterion(logits, y) # z czego liczymy stratę/koszt? a no z logitów i poprawnych y
            loss.backward() # mądra głowa liczy gradienty

            optimiser.step() # optymalizator aktualizuje wagi

            running_loss += loss.item() * x.size(0)
            _, predicted_labels = logits.max(1) # max(1), no bo logity to [B, Classes] i bierzemy co model obstawia

            # tu jest tensor, bo przecież wszystko tutaj to tensory, chyba że powiem inaczej
            is_correct = (predicted_labels == y) # jak obstawił dobrze to TRUE

            # PyCharm nie nadąża, że to tensor a nie bool
            correct += torch.sum(is_correct).item() # bo w tym batch'u, więc sumuję ile dobrze obstawił na ten batch
            total += y.size(0) # rozmiar batch'a

        scheduler.step()  # harmonogramator zmienia globalnie lr (jakby się accuracy płaszczyło)
        # i ja go wołam raz na epokę, żeby on sprawdził, czy ma zmienić lr czy to nie czas

        torch.cuda.synchronize() # żeby poczekać aż GPU wszystko policzy i dopiero oznaczyć jako end
        epoch_end = time.time()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        epoch = epoch_idx + 1
        print(f"Epoch {epoch_idx + 1}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")
        print(f"Epoch {epoch} processing time: {epoch_end - epoch_start:.2f} s")
        print("")



    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimiser.state_dict(),
    }

    torch.save(checkpoint, TRAINED_WEIGHTS_PATH)
    print("[Wagi modelu zapisane]", str(TRAINED_WEIGHTS_PATH), "\nDziękuję, dobranoc")