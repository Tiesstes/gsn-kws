import time
from pathlib import Path
import gc

import torch

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

from project.data.dataset import extract_custom_speakers, split_custom_data, CustomWAVSpeechCommandsKWS
from project.data.data_split import prepare_splits_manifest, load_splits_manifest, build_phase_datasets, build_dataloaders, extend_speaker_id_map

from project.model.kws_net import KWSNet

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


BASE_PATH = Path(__file__).resolve().parent
GSC_DATASET_PATH = Path(BASE_PATH) / "data"
NOISE_PATH = Path(GSC_DATASET_PATH) / "SpeechCommands" / "speech_commands_v0.02" / "_background_noise_"

PRETRAIN_CHECKPOINT = BASE_PATH / "model" / f"pretrain_checkpoint_" # tu trzeba na koniec dodać numerek z epoką! to potem
FINETUNE_CHECKPOINT = BASE_PATH / "model" / f"finetune_checkpoint_"

IMPORT_EPOCH = 31 # to jest wspólna zmienna do odtwarzania checkpoint'ów!
PRETRAIN_IMPORT_CHECKPOINT = BASE_PATH / "model" / f"pretrain_checkpoint_{IMPORT_EPOCH}.pt" # tu trzeba na koniec dodać numerek z epoką! to potem
FINETUNE_IMPORT_CHECKPOINT = BASE_PATH / "model" / f"finetune_checkpoint_{IMPORT_EPOCH}.pt"

# zamień na: f"finetune_checkpoint_{IMPORT_EPOCH}.pt"

DATASPLIT_MANIFEST_PATH = BASE_PATH / "data"/ "splits" / "experiment_v1.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DO_PRETRAIN = True
DO_FINETUNE = False
DO_EVALUATE = False

RESUME_TRAINING = False

PRETRAIN_EPOCHS = 56
FINETUNE_EPOCHS = 24
BATCH_SIZE = 128
WORKERS = 4

PRETRAIN_LR = 0.001
FINETUNE_LR = 0.01
WEIGHT_DECAY = 0.01
FACTOR = 0.7
PATIENCE = 3

def clear_memory():

    if torch.cuda.is_available():

        torch.cuda.empty_cache()
    gc.collect()

    print("Lekkie czyszczenie pamięci, done")

def load_checkpoint(checkpoint_path, model, optimiser, scheduler, device):
    if not checkpoint_path.exists():
        return None

    print(f"\nWczytuję checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])

    optimiser.load_state_dict(checkpoint["optimiser_state"])

    scheduler.load_state_dict(checkpoint["scheduler_state"])


    print(f"Checkpoint załadowany:")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Val accuracy: {checkpoint['val_accuracy']:.4f}\n")

    return {"start_epoch": checkpoint["epoch"] + 1,
            "best_val_acc": checkpoint["val_accuracy"],
            "speaker_id_map": checkpoint["speaker_id_map"],
            "label_map": checkpoint["label_map"]}



# metoda dla epoki
def run_epoch(net_model, data_loader, device, criterion, net_optimiser=None):

    if net_optimiser is not None:
        net_model.train(True)
        mode = torch.enable_grad()


    else:
        net_model.train(False)
        mode = torch.no_grad()

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    #context = torch.enable_grad() if train_mode else torch.no_grad()
    with mode:

        for batch in data_loader:

            x_input = batch["log_mel_spectrogram"].to(device)
            y = batch["label"].to(device)
            current_speaker = batch["speaker_id"].to(device)

            logits = net_model(x_input, current_speaker)
            loss = criterion(logits, y)

            if net_optimiser is not None:
                net_optimiser.zero_grad()
                loss.backward()
                net_optimiser.step()

            total_loss += loss.detach().item() * x_input.shape[0]
            predictions = logits.detach().argmax(dim=1)
            total_correct += int((predictions == y).sum())
            total_n += x_input.shape[0]

    return total_loss / max(1, total_n), total_correct / max(1, total_n)


if __name__ == "__main__":

    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print("CUDA:", torch.cuda.get_device_name(0))

    print("Torch version:", torch.__version__)
    print("")

    # tutaj robimy podstawowy dataset z danymi GSC
    print("Ładuję dataset SPEECHCOMMANDS...")
    #base_data = SPEECHCOMMANDS(root=GSC_DATASET_PATH, subset=None, download=True)

    # do pętli treningowej
    start_epoch = 1
    best_val_acc = -1.0  # żeby mieć pewność, że na pewno wyłapiemy najlepsze accuracy


    # spójrzmy czy jest jakiś plik ze splitami
    if not DATASPLIT_MANIFEST_PATH.exists():

        # jak go nie ma to robimy (na później)
        prepare_splits_manifest(data_manifest_path=DATASPLIT_MANIFEST_PATH, gsc_dataset_path=GSC_DATASET_PATH, noise_dir=NOISE_PATH,
                                seed=1234, finetune_min_samples_per_class=6, pretrain_val_ratio=0.1, duration=1.0, sample_rate=16000,
                                number_of_mel_bands=40, silence_per_target=1.5, unknown_to_target_ratio=3.0) # bo wewnętrzne zróżnicowanie jest gigantyczne (25 podklas)

    # jak jest to wczytanie od razu (albo po utworzeniu)
    split_manifest = load_splits_manifest(DATASPLIT_MANIFEST_PATH)

    num_of_classes = len(split_manifest["maps"]["label_map"])
    # zawsze zacyznamy od pretrain
    num_of_speakers_pretrain = len(split_manifest["maps"]["speaker_id_map_pretrain"])

    model = KWSNet(num_of_classes, num_of_speakers_pretrain) # uwaga bo jak jest mismatch embeddingu między wczytanym splitem
    # a w modelu to jest błąd (niezaimplementowany safeguard)
    model.to(DEVICE)

    optimiser = torch.optim.Adam(model.parameters(), lr=PRETRAIN_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=FACTOR, patience=PATIENCE)

    if DO_PRETRAIN:

        phase_name = "PRETRAIN"
        phase_save_path = PRETRAIN_CHECKPOINT
        epochs_per_phase = PRETRAIN_EPOCHS

        training_dataset, val_dataset, _ = build_phase_datasets(split_manifest, "pretrain")
        active_training_loader, active_val_loader, _ = build_dataloaders(training_dataset, val_dataset, batch_size=BATCH_SIZE, num_workers=WORKERS)


        if RESUME_TRAINING and PRETRAIN_IMPORT_CHECKPOINT.exists():
            checkpoint_data = load_checkpoint(PRETRAIN_IMPORT_CHECKPOINT, model, optimiser, scheduler, DEVICE)

            if checkpoint_data:
                start_epoch = checkpoint_data["start_epoch"]
                best_val_acc = checkpoint_data["best_val_acc"]
                print("Odtworzyłem checkpoint do PRETRENING")


        print("kształt Embedding'u:", model.speaker_embedding.weight.shape)

    elif DO_FINETUNE:

        phase_name = "FINETUNE"
        phase_save_path = FINETUNE_CHECKPOINT
        epochs_per_phase = FINETUNE_EPOCHS

        # !!!!!!!!!!!!!! custom dataset !!!!!!!!!!!!!!
        # insert speakerów custom
        CUSTOM_DATA_ROOT = Path(GSC_DATASET_PATH) / "SpeechCommands" / "custom_speech_commands_v0.02"
        custom_speakers_map = extract_custom_speakers(CUSTOM_DATA_ROOT)

        print(f"Spekaerzy z własnego datasetu: {sorted(custom_speakers_map)}")


        custom_spakers_id_map = extend_speaker_id_map(split_manifest["maps"]["speaker_id_map_pretrain"],
                                                      custom_speakers_map)

        print(f"Starych speakerów, było - {len(split_manifest["maps"]["speaker_id_map_pretrain"])}")

        train_indices, val_indices, test_indices = split_custom_data(CUSTOM_DATA_ROOT,train_ratio=0.7,val_ratio=0.15,
                                                                     test_ratio=0.15,seed=1234)

        label_map = split_manifest["maps"]["label_map"]

        training_dataset = CustomWAVSpeechCommandsKWS(CUSTOM_DATA_ROOT, train_indices, custom_spakers_id_map, label_map,
                                                      sample_rate=16000, duration=1.0, number_of_mel_bands=40,
                                                      deterministic=True, seed=1234)  #  nie ma mieszania który wycinek z silence (chociaż tutaj to bez znaczenia)

        val_dataset = CustomWAVSpeechCommandsKWS(CUSTOM_DATA_ROOT, val_indices, custom_spakers_id_map, label_map,
                                                 sample_rate=16000, duration=1.0, number_of_mel_bands=40,
                                                 deterministic=True, seed=1234)

        test_dataset = CustomWAVSpeechCommandsKWS(CUSTOM_DATA_ROOT, test_indices, custom_spakers_id_map, label_map,
                                                  sample_rate=16000, duration=1.0, number_of_mel_bands=40,
                                                  deterministic=True, seed=1234)


        # bezpośrednie użycie DataLoader zamiast build_dataloaders
        active_training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, persistent_workers=True)
        active_val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, persistent_workers=True)

        # zwiększmy embedding
        num_speakers_ft = len(custom_spakers_id_map)

        print(f"Mapa speakerów została zaktualizowana do liczby -> {num_speakers_ft}")

        model.freeze_backbone()  # zamróź wagi backbone

        # optymalizator i harmongramator (okresowa zmiana wartości learning_rate)
        # weights decay to regularyzacja L2 - karze duże wagi;
        # czyli dodaje karę do straty jako λ * w^2 (to się dziele w optymalizatorze),
        # a ta λ = weight_decay
        optimiser = torch.optim.Adam(model.get_trainable_parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=FACTOR, patience=PATIENCE)


        # łądowanie wag jeżeli możemy
        if RESUME_TRAINING and FINETUNE_IMPORT_CHECKPOINT.exists():

            model.ensure_num_of_speakers(num_speakers_ft)
            checkpoint_finetune = load_checkpoint(FINETUNE_IMPORT_CHECKPOINT, model, optimiser, scheduler, DEVICE)

            if checkpoint_finetune:
                start_epoch = checkpoint_finetune["start_epoch"]
                best_val_acc = checkpoint_finetune["best_val_acc"]

            print(f"Wznawiam finetune od epoki {start_epoch}")
            print(f"Najlepsze val_acc: {best_val_acc:.4f}\n")

        elif PRETRAIN_IMPORT_CHECKPOINT.exists(): # jak nie to pretrain

            checkpoint = torch.load(PRETRAIN_IMPORT_CHECKPOINT, map_location=DEVICE)
            # tutaj musimy ładować 2613 embeddingów i zmienić liczbę speakerów
            model.load_state_dict(checkpoint["model_state"], strict=True)
            print("Pretrain checkpoint załadowany do finetune start (nie było innego)")
            model.ensure_num_of_speakers(num_speakers_ft)


        print("kształt Embedding'u:", model.speaker_embedding.weight.shape)


    criterion = CrossEntropyLoss()

    print()
    print("~~ARCHITEKTURA SIECI~~")
    print()

    speakers = model.speaker_embedding.num_embeddings
    x = torch.randn(BATCH_SIZE, 1, 40, 101, device=DEVICE)
    speaker = torch.randint(0, speakers, (BATCH_SIZE,), device=DEVICE)  # LONG!

    summary(model, depth=5, input_data=(x, speaker))

    torch.cuda.synchronize()  # żeby poczekać, aż GPU dokończy obliczenia zanim zmierzymy czas
    start = time.perf_counter()

    log_dir = BASE_PATH / "runs" / f"{phase_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs will be saved to {log_dir}")

    print(f"Przechodzę do {phase_name} - epoka {start_epoch} z {epochs_per_phase}")
    print("")



    for epoch in range(start_epoch, epochs_per_phase + 1):

        epoch_start = time.perf_counter()

        training_loss, training_accuracy = run_epoch(model, active_training_loader, DEVICE, criterion, optimiser)
        val_loss, val_accuracy = run_epoch(model, active_val_loader, DEVICE, criterion, None)

        scheduler.step(val_loss) # po epoce (zmieni lr, jeśli epoka przeszła step size
        current_lr = optimiser.param_groups[0]['lr']
        print(f"Epoch {epoch}, LR = {current_lr:.6f}")

        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start

        writer.add_scalar(f"{phase_name}/Loss/Train", training_loss, epoch)
        writer.add_scalar(f"{phase_name}/Loss/Val", val_loss, epoch)
        writer.add_scalar(f"{phase_name}/Accuracy/Train", training_accuracy, epoch)
        writer.add_scalar(f"{phase_name}/Accuracy/Val", val_accuracy, epoch)
        writer.add_scalar(f"{phase_name}/LearningRate", current_lr, epoch)

        print("")
        print(f"Epoch of {phase_name} {epoch:02d}, "
              f"train loss {training_loss:.4f} accuracy {training_accuracy:.4f}, "
              f"val loss {val_loss:.4f} accuracy {val_accuracy:.4f} "
              f"time {epoch_time:.1f}s")

        clear_memory()

        # zapisuj ten najlepszy checkpoint (najlepsze osiągi)
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy

            # musi istnieć
            phase_save_path.parent.mkdir(parents=True, exist_ok=True)

            speaker_id_map_to_save = (split_manifest["maps"]["speaker_id_map_pretrain"] if DO_PRETRAIN
                else custom_spakers_id_map)  # roozszerzenie o nowych mówców

            best_val_acc = val_accuracy
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                            "optimiser_state": optimiser.state_dict(), "scheduler_state": scheduler.state_dict(),
                            "val_accuracy": val_accuracy,
                            "speaker_id_map": split_manifest["maps"]["speaker_id_map_pretrain"] if DO_PRETRAIN else custom_spakers_id_map,
                            "label_map": split_manifest["maps"]["label_map"]},
                           f"{phase_save_path}{epoch}.pt")


            print(f"Zapisany checkpoint (val_accuracy={val_accuracy:.4f})")
            print("")

    # torch.cuda.synchronize()
    total_time = time.perf_counter() - start


    print("")
    print(f"Trebiwanie trwało: {total_time:.1f}s")
    print(f"Najlepsza wartość val_accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint zapisany w: {phase_save_path}{epoch}")
    print("")

    writer.close()
