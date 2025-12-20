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

# konfiguracja

BASE_PATH = Path(__file__).resolve().parent.parent
GSC_PATH = Path(__file__).resolve().parent
NOISE_PATH = Path(GSC_PATH) / "SpeechCommands" / "speech_commands_v0.02" / "_background_noise_"
PRETRAIN_CHECKPOINT = Path(BASE_PATH) / "model" / "pretrain_checkpoint.pt"
FINETUNE_CHECKPOINT = Path(BASE_PATH) / "model" / "finetune_checkpoint.pt"

EPOCHS = 32
BATCH_SIZE = 128
WORKERS = 0
LR = 0.001
WEIGHT_DECAY = 0.0
SCHEDULER_STEP_SIZE = 4
SCHEDULER_GAMMA = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)
if torch.cuda.is_available():
    print("CUDA:", torch.cuda.get_device_name(0))
print("Torch version:", torch.__version__)
print("")



def extend_speaker_id_map(old_map: dict[str, int], new_speakers: set[str]) -> dict[str, int]:
    """Rozszerza mapę speakerów o nowych. Stare ID zostają niezmienione."""
    new_map = dict(old_map)
    next_id = max(new_map.values()) + 1 if new_map else 0

    added = 0

    for current_speaker in sorted(new_speakers):
        if current_speaker not in new_map:
            new_map[current_speaker] = next_id
            next_id += 1
            added += 1

    if added > 0:
        print(f"{added} nowych speaker'ów")

    return new_map


def freeze_backbone(net_model):
    """Zamraża parametry w backbone."""

    for param in net_model.backbone.parameters():
        param.requires_grad = False
    print("Backbone zamrożony")


def get_trainable_params(net_model):
    """Zwraca lista trenowalnych parametrów."""
    return [p for p in net_model.parameters() if p.requires_grad]


# jedna epoka, całość
def run_epoch(net_model, data_loader, device, net_optimiser=None):
    """Uruchomia jedną epokę (train lub val)."""

    if net_optimiser is not None:
        net_model.train(True)
        mode = torch.enable_grad()

    else:
        net_model.train(False)
        mode = torch.no_grad()

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    with mode:
        for batch in data_loader:

            x_input = batch["log_mel_spectrogram"].to(device)  # [B, 1, 40, T]
            y = batch["label"].to(device)  # [B]
            current_speakr = batch["speaker_id"].to(device)  # [B]

            logits = net_model(x_input, current_speakr)  # [B, num_classes]
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
    # 1) Załaduj pretrain checkpoint
    print("Ładuję checkpoint pretrainu...")
    if not PRETRAIN_CHECKPOINT.exists():
        raise FileNotFoundError(f"Nie znaleziono: {PRETRAIN_CHECKPOINT}")

    checkpoint = torch.load(PRETRAIN_CHECKPOINT, map_location=DEVICE)
    old_model_state = checkpoint["model_state"]
    old_speaker_id_map = checkpoint["speaker_id_map"]
    old_label_map = checkpoint["label_map"]

    print(f"Załadowany checkpoint (epoka {checkpoint['epoch']}, val_accuracy{checkpoint['val_accuracy']:.4f})")
    print(f"Liczba speaker'ów: {len(old_speaker_id_map)}")
    print("")

    # znowu bazowy dataset GSC
    print("Ładuję dataset SPEECHCOMMANDS...")
    base_data = SPEECHCOMMANDS(root=GSC_PATH, subset=None, download=True)

    # znowu SplitBuilder
    print("Robię splity finetune...")
    dataset_indexer = SplitBuilder(
        base_data, fine_tune_max_samples_per_class=6,pretrain_val_ratio=0.1, seed=1234, )
    finetune_split = dataset_indexer.build_finetune_splits()

    print(f"Liczba speakers finetune: {len(finetune_split['allowed_speakers'])}")

    # Zwiększ teraz speaker mape (jakby Co)
    print("Rozszerzam speaker_id_map...")
    new_speaker_id_map = extend_speaker_id_map(old_speaker_id_map, finetune_split["allowed_speakers"])
    print(f"Teraz speaker'ów jest: {len(new_speaker_id_map)}")
    print()

    # podziały na treningowy, walidacyjny i testowy

    print("Tworzę datasety treningu, walidacji i testu..")
    train_dataset = SpeechCommandsKWS(dataset=base_data,split_indices=finetune_split["train"],
        allowed_speakers=finetune_split["allowed_speakers"], speaker_id_map=new_speaker_id_map,
        noise_dir=NOISE_PATH,silence_per_target=1.0,
        unknown_to_target_ratio=1.0, seed=1234)

    val_dataset = SpeechCommandsKWS(dataset=base_data,split_indices=finetune_split["val"],allowed_speakers=finetune_split["allowed_speakers"],
            speaker_id_map=new_speaker_id_map,noise_dir=NOISE_PATH,silence_per_target=1.0,
                                    unknown_to_target_ratio=1.0,seed=1234)

    test_dataset = SpeechCommandsKWS(dataset=base_data,split_indices=finetune_split["test"],allowed_speakers=finetune_split["allowed_speakers"],
        speaker_id_map=new_speaker_id_map,noise_dir=NOISE_PATH,silence_per_target=1.0,unknown_to_target_ratio=1.0,
                                     seed=1234)


    # data loadery dla datasetów
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

    # do inicjalizacji modelu
    num_of_classes = len(old_label_map)
    num_of_speakers = len(new_speaker_id_map)

    print("Tworzę model...")
    print(f"num_of_classes: {num_of_classes}")
    print(f"num_of_speakers: {num_of_speakers}")

    model = KWSNet(num_of_classes=num_of_classes, num_of_speakers=num_of_speakers).to(DEVICE)


    # weź to co wyuoczne w pretreningu
    print("Ładuję wagi modelu z pliku pretreningowego...")
    model.load_state_dict(old_model_state)

    # zwiększ embedding dla nowych speaker'ów
    model.ensure_num_of_speakers(num_of_speakers)


    # morzimy backbone
    print("Zamrażam backbone...")
    freeze_backbone(model)


    # optymalizator robi trainable params: embedding + classifier; bez reszty
    trainable = get_trainable_params(model)
    print(f"Parametry do treningu: {sum(p.numel() for p in trainable):,}")
    optimiser = torch.optim.Adam(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

    # dalej mamy scheduler do pomocy
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)


    # żeby zobaczyc ile teraz jest parametrów do trenowania (aktualizacji gradientem)
    print("[Architektura sieci]")

    speakers = model.speaker_embedding.num_embeddings
    x = torch.randn(BATCH_SIZE, 1, 40, 101, device=DEVICE)
    speaker = torch.randint(0, speakers, (BATCH_SIZE,), device=DEVICE)

    summary(model, depth=3, input_data=(x, speaker))

    # loop do trenowania embeddingu
    best_val_acc = -1.0
    torch.cuda.synchronize()
    start = time.perf_counter()

    print("Zaczynam fazę finetune...")
    print()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()

        training_loss, training_accuracy = run_epoch(model, train_loader, DEVICE, net_optimiser=optimiser)
        val_loss, val_accuracy = run_epoch(model, val_loader, DEVICE, net_optimiser=None)

        scheduler.step()

        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {training_loss:.4f} accuracy {training_accuracy:.4f} | "
            f"val loss {val_loss:.4f} accuracy {val_accuracy:.4f} | "
            f"time {epoch_time:.1f}s"
        )

        # zapisujemy znowy najlepszy checkpoint
        if val_accuracy > best_val_acc:

            best_val_acc = val_accuracy

            FINETUNE_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimiser_state": optimiser.state_dict(),
                    "speaker_id_map": new_speaker_id_map,
                    "label_map": old_label_map,
                    "val_accuracy": val_accuracy,},FINETUNE_CHECKPOINT,)

            print(f"Zapisałem checkpoint (val_accuracy={val_accuracy:.4f})")
            print("")

    # testowanie

    print("Rozpoczynam ewaluację...")
    test_loss, test_accuracy = run_epoch(model, test_loader, DEVICE, net_optimiser=None)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")
    print("")

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start

    print(f"Finetune zakończony. Całkowity czas: {total_time:.1f}s")
    print(f"Najlepsze val_accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint zapisanty: {FINETUNE_CHECKPOINT}")
