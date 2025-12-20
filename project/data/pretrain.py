import time


import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS
from torchinfo import summary


from project.data.dataset import SplitBuilder
from dataset import SpeechCommandsKWS
from project.model.kws_net import KWSNet
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

# --- CONFIG ---
BASE_PATH = Path(__file__).resolve().parent.parent
GSC_PATH = Path(__file__).resolve().parent
NOISE_PATH = Path(GSC_PATH) / "SpeechCommands" / "speech_commands_v0.02" / "_background_noise_"
CHECKPOINT_PATH = Path(BASE_PATH) / "model" / "pretrain_checkpoint.pt"

EPOCHS = 32
BATCH_SIZE = 128
WORKERS = 0
LR = 0.01
WEIGHT_DECAY = 0.0
SCHEDULER_STEP_SIZE = 4
SCHEDULER_GAMMA = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)
if torch.cuda.is_available():
    print("CUDA:", torch.cuda.get_device_name(0))
print("Torch version:", torch.__version__)
print()


# metoda dla epoki
def run_epoch(model, data_loader, device, optimiser=None):

    if optimiser is not None:
        model.train(True)
        mode = torch.enable_grad()
    else:
        model.train(False)
        mode = torch.no_grad()

    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    #context = torch.enable_grad() if train_mode else torch.no_grad()
    with mode:
        for batch in data_loader:
            x = batch["log_mel_spectrogram"].to(device)
            y = batch["label"].to(device)
            spk = batch["speaker_id"].to(device)

            logits = model(x, spk)
            loss = criterion(logits, y)

            if optimiser is not None:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            total_loss += loss.detach().item() * x.shape[0]
            predictions = logits.detach().argmax(dim=1)
            total_correct += int((predictions == y).sum())
            total_n += x.shape[0]

    return total_loss / max(1, total_n), total_correct / max(1, total_n)


if __name__ == "__main__":
    # tutaj robimy podstawowy dataset z danymi GSC
    print("Ładuję dataset SPEECHCOMMANDS...")
    base_data = SPEECHCOMMANDS(root=GSC_PATH, subset=None, download=True)


    print("Robię splity z SplitBuilder...")

    # Tu definicja podziałów na podzbiory dla fazy pretreningu
    dataset_indexer = SplitBuilder(base_data, fine_tune_max_samples_per_class=6,pretrain_val_ratio=0.1,seed=1234)
    pretrain_split = dataset_indexer.build_pretrain_splits()



    print("Tworzę datasety treningu i walidacji...")


    # zrób datasety oddzielnie: trening i walidacja
    train_dataset = SpeechCommandsKWS(dataset=base_data,
                                      split_indices=pretrain_split["train"],
                                      allowed_speakers=pretrain_split["allowed_speakers"], speaker_id_map=dataset_indexer.speaker_id_map, noise_dir=NOISE_PATH, silence_per_target=1.0, unknown_to_target_ratio=1.0, seed=1234)

    val_dataset = SpeechCommandsKWS(dataset=base_data,
        split_indices=pretrain_split["val"], allowed_speakers=pretrain_split["allowed_speakers"],
        speaker_id_map=dataset_indexer.speaker_id_map, noise_dir=NOISE_PATH,
        silence_per_target=1.0, unknown_to_target_ratio=1.0, seed=1234)

    print(f"Liczność datasetu treningowego: {len(train_dataset)}")
    print(f"Liczność datasetu walidacyjnego:   {len(val_dataset)}")
    print("")

    # data loadery przy pretreningu z podziałem na odpowiednie subsety
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

    # informacje dla modelu (do stworzenia obiektu)
    num_of_classes = len(train_dataset.label_map) # mapa etykiet na inty
    num_of_speakers = len(dataset_indexer.speaker_id_map) # mapa speaker id na inty

    model = KWSNet(num_of_classes=num_of_classes, num_of_speakers=num_of_speakers).to(DEVICE)

    # optymalizator i harmongramator (okresowa zmiana wartości learning_rate

    # weights decay to regularyzacja L2 - karze duże wagi;
    # czyli dodaje karę do straty jako λ * w^2 (to się dziele w optymalizatorze),
    # a ta λ = weight_decay
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA
    )

    # do pętli treningowej
    best_val_acc = -1.0 # żeby mieć pewność, że na pewno wyłapiemy najlepsze accuracy

    torch.cuda.synchronize() # żeby poczekać, aż GPU dokończy obliczenia zanim zmierzymy czas
    start = time.perf_counter()

    print("[Architektura sieci]")

    speakers = model.speaker_embedding.num_embeddings
    x = torch.randn(BATCH_SIZE, 1, 40, 101, device=DEVICE)
    speaker = torch.randint(0, speakers, (BATCH_SIZE,), device=DEVICE)  # LONG!

    summary(model, depth=5, input_data=(x, speaker))



    print("Zaczynam treenować...")
    print("")

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()


        training_loss, training_accuracy = run_epoch(model, train_loader, DEVICE, optimiser=optimiser)
        val_loss, val_accuracy = run_epoch(model, val_loader, DEVICE, optimiser=None)

        scheduler.step() # po epoce (zmieni lr, jeśli epoka przeszła step size

        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start

        print(
            f"Epoch {epoch:02d}, "
            f"train loss {training_loss:.4f} accuracy {training_accuracy:.4f}, "
            f"val loss {val_loss:.4f} accuracy {val_accuracy:.4f} "
            f"time {epoch_time:.1f}s"
        )

        # zapisuj ten najlepszy checkpoint (najlepsze osiągi)
        if val_accuracy > best_val_acc:

            best_val_acc = val_accuracy

            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

            torch.save({"epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimiser_state": optimiser.state_dict(),
                    "speaker_id_map": dataset_indexer.speaker_id_map,
                    "label_map": train_dataset.label_map,
                    "val_accuracy": val_accuracy,},
                CHECKPOINT_PATH,)

            print(f"Zapisany checkpoint (val_accuracy={val_accuracy:.4f})")
            print("")

    # torch.cuda.synchronize()
    total_time = time.perf_counter() - start


    print(f"Zakończono trenowanie w całkowitym czasie: {total_time:.1f}s")
    print(f"NAjlepsza wartość val_accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint zapisany w: {CHECKPOINT_PATH}")

