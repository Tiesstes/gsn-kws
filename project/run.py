import time
from pathlib import Path
import gc

import torch

from torch.nn import CrossEntropyLoss
from torchinfo import summary


from project.data.data_split import prepare_splits_manifest, load_splits_manifest, build_phase_datasets, build_dataloaders

from project.model.kws_net import KWSNet

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

BASE_PATH = Path(__file__).resolve().parent
GSC_DATASET_PATH = Path(BASE_PATH) / "data"
NOISE_PATH = Path(GSC_DATASET_PATH) / "SpeechCommands" / "speech_commands_v0.02" / "_background_noise_"
PRETRAIN_CHECKPOINT = BASE_PATH / "model" / "pretrain_checkpoint.pt"
FINETUNE_CHECKPOINT = BASE_PATH / "model" / "finetune_checkpoint.pt"

DATASPLIT_MANIFEST_PATH = BASE_PATH / "data"/ "splits" / "experiment_v1.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DO_PRETRAIN = True
DO_FINETUNE = False
DO_EVALUATE = False

RESUME_TRAINING = True

PRETRAIN_EPOCHS = 36
FINETUNE_EPOCHS = 16
BATCH_SIZE = 128
WORKERS = 0

PRETRAIN_LR = 0.001
FINETUNE_LR = 0.01
WEIGHT_DECAY = 0.0
SCHEDULER_STEP_SIZE = 4
SCHEDULER_GAMMA = 0.5

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


    print(f" Checkpoint załadowany:")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Val accuracy: {checkpoint['val_accuracy']:.4f}\n")

    return {
        "start_epoch": checkpoint["epoch"] + 1,
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

    #criterion = torch.nn.CrossEntropyLoss() # TODO: wywalić na zewnątrz
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

# TODO: resolve the matter of finetune and evaluation integration into one script

if __name__ == "__main__":

    print("Device:", DEVICE)
    if torch.cuda.is_available():
        print("CUDA:", torch.cuda.get_device_name(0))
    print("Torch version:", torch.__version__)
    print()

    # tutaj robimy podstawowy dataset z danymi GSC
    print("Ładuję dataset SPEECHCOMMANDS...")
    #base_data = SPEECHCOMMANDS(root=GSC_DATASET_PATH, subset=None, download=True)


    # spójrzmy czy jest jakiś plik ze splitami
    if not DATASPLIT_MANIFEST_PATH.exists():

        # jak go nie ma to robimy (na później)
        prepare_splits_manifest(data_manifest_path=DATASPLIT_MANIFEST_PATH, gsc_dataset_path=GSC_DATASET_PATH, noise_dir=NOISE_PATH,
                                seed=1234, finetune_min_samples_per_class=6, pretrain_val_ratio=0.1, duration=1.0, sample_rate=16000,
                                number_of_mel_bands=40, silence_per_target=1.0, unknown_to_target_ratio=1.0)

    # jak jest to wczytanie od razu (albo po utworzeniu)
    split_manifest = load_splits_manifest(DATASPLIT_MANIFEST_PATH)

    # TODO: przenieść do logiki flag PRETRAIN etc.
    pretrain_train_dataset, pretrain_val_dataset, _ = build_phase_datasets(split_manifest, "pretrain", False)

    # w środku ustalone, że shuffle się robi tylko na trening
    pretrain_train_ld, pretrain_val_ld, _ = build_dataloaders(pretrain_train_dataset, pretrain_val_dataset,  batch_size=BATCH_SIZE, num_workers=WORKERS)

    num_of_classes = len(split_manifest["maps"]["label_map"])
    num_of_speakers_pretrain = len(split_manifest["maps"]["speaker_id_map_pretrain"])

    model = KWSNet(num_of_classes, num_of_speakers_pretrain)
    model.to(DEVICE)

    criterion = CrossEntropyLoss()
    # optymalizator i harmongramator (okresowa zmiana wartości learning_rate)

    # weights decay to regularyzacja L2 - karze duże wagi;
    # czyli dodaje karę do straty jako λ * w^2 (to się dziele w optymalizatorze),
    # a ta λ = weight_decay
    optimiser = torch.optim.Adam(model.parameters(), lr=PRETRAIN_LR, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

    print("~~ARCHITEKTURA SIECI~~")
    print()

    speakers = model.speaker_embedding.num_embeddings
    x = torch.randn(BATCH_SIZE, 1, 40, 101, device=DEVICE)
    speaker = torch.randint(0, speakers, (BATCH_SIZE,), device=DEVICE)  # LONG!

    summary(model, depth=5, input_data=(x, speaker))

    # do pętli treningowej
    start_epoch = 1
    best_val_acc = -1.0 # żeby mieć pewność, że na pewno wyłapiemy najlepsze accuracy

    if RESUME_TRAINING and PRETRAIN_CHECKPOINT.exists() and DO_PRETRAIN:
        resume_info_pre = load_checkpoint(PRETRAIN_CHECKPOINT, model, optimiser, scheduler, DEVICE)
        print("kształt Embedding'u:", model.speaker_embedding.weight.shape)

        if resume_info_pre:

            start_epoch = resume_info_pre["start_epoch"]
            best_val_acc = resume_info_pre["best_val_acc"]

            print(f"Wznawiam trening od epoki {start_epoch}")
            print(f"Najlepsze val_acc: {best_val_acc:.4f}\n")

    else:
        if not DO_FINETUNE and not DO_EVALUATE:
            print("Rozpoczynam trening od 0\n")


    if FINETUNE_CHECKPOINT.exists() and RESUME_TRAINING and DO_FINETUNE:
        resume_info_ft = load_checkpoint(FINETUNE_CHECKPOINT, model, optimiser, scheduler, DEVICE)
        print("kształt Embedding'u:", model.speaker_embedding.weight.shape)

        if resume_info_ft:
            start_epoch_ft = resume_info_ft["start_epoch"]
            best_val_acc_ft = resume_info_ft["best_val_acc"]
            print(f"Wznawiam finetune od epoki {start_epoch_ft}")
            print(f"Najlepsze val_acc: {best_val_acc_ft:.4f}\n")

    if DO_EVALUATE and not RESUME_TRAINING:

        if FINETUNE_CHECKPOINT.exists():
            evaluation_checkpoint = FINETUNE_CHECKPOINT
            model.load_state_dict(torch.load(evaluation_checkpoint, map_location=DEVICE)["model_state"])
            print("kształt Embedding'u:", model.speaker_embedding.weight.shape)

        elif PRETRAIN_CHECKPOINT.exists():
            evaluation_checkpoint = PRETRAIN_CHECKPOINT
            model.load_state_dict(torch.load(evaluation_checkpoint, map_location=DEVICE)["model_state"])
            print("kształt Embedding'u:", model.speaker_embedding.weight.shape)

        else:
            raise FileNotFoundError("Nie istnieje żaden plik z checkpoint'em")



    torch.cuda.synchronize()  # żeby poczekać, aż GPU dokończy obliczenia zanim zmierzymy czas
    start = time.perf_counter()

    print(f"Przechodzę do treningu PRETRAIN - epoka {start_epoch} z {PRETRAIN_EPOCHS}")
    print("")


    for epoch in range(start_epoch, PRETRAIN_EPOCHS + 1):

        epoch_start = time.perf_counter()

        training_loss, training_accuracy = run_epoch(model, pretrain_train_ld, DEVICE, criterion, net_optimiser=optimiser)
        val_loss, val_accuracy = run_epoch(model, pretrain_val_ld, DEVICE, criterion, net_optimiser=None)

        scheduler.step() # po epoce (zmieni lr, jeśli epoka przeszła step size

        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start

        print(f"Epoch of PRETRAIN {epoch:02d}, "
              f"train loss {training_loss:.4f} accuracy {training_accuracy:.4f}, "
              f"val loss {val_loss:.4f} accuracy {val_accuracy:.4f} "
              f"time {epoch_time:.1f}s")

        clear_memory()

        # zapisuj ten najlepszy checkpoint (najlepsze osiągi)
        if val_accuracy > best_val_acc:

            best_val_acc = val_accuracy

            PRETRAIN_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

            torch.save({"epoch": epoch,
                        "model_state": model.state_dict(), "optimiser_state": optimiser.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "speaker_id_map": split_manifest["maps"]["speaker_id_map_pretrain"],
                        "label_map": split_manifest["maps"]["label_map"],
                        "val_accuracy": val_accuracy},
                       PRETRAIN_CHECKPOINT)


            print(f"Zapisany checkpoint (val_accuracy={val_accuracy:.4f})")
            print("")

    # torch.cuda.synchronize()
    total_time = time.perf_counter() - start


    print(f"Zakończono trenowanie w całkowitym czasie: {total_time:.1f}s")
    print(f"NAjlepsza wartość val_accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint zapisany w: {PRETRAIN_CHECKPOINT}")

