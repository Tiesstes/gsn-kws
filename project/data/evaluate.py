import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from torchaudio.datasets import SPEECHCOMMANDS

from project.data.dataset import SplitBuilder
from dataset import SpeechCommandsKWS
from project.model.kws_net import KWSNet

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

# konfiguracje

BASE_PATH = Path(__file__).resolve().parent.parent
GSC_PATH = Path(__file__).resolve().parent
NOISE_PATH = Path(GSC_PATH) / "SpeechCommands" / "speech_commands_v0.02" / "_background_noise_"

PRETRAIN_CHECKPOINT = Path(BASE_PATH) / "model" / "pretrain_checkpoint.pt"
FINETUNE_CHECKPOINT = Path(BASE_PATH) / "model" / "finetune_checkpoint.pt"

BATCH_SIZE = 128
WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)
if torch.cuda.is_available():
    print("CUDA:", torch.cuda.get_device_name(0))
print("Torch version:", torch.__version__)
print()


# funckje pomocnicze

def evaluate(model, data_loader, device):
    """Ewaluuj model na zbiorze testowym, zwracaj predykcje i etykiety."""
    model.eval()

    all_predictions = []
    all_labels = []
    total_correct = 0
    total_n = 0

    with torch.no_grad():

        for batch in data_loader:

            x = batch["log_mel_spectrogram"].to(device)
            y = batch["label"].to(device)
            spk = batch["speaker_id"].to(device)

            logits = model(x, spk)
            predictions = logits.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            total_correct += int((predictions == y).sum())
            total_n += x.shape[0]

    accuracy = total_correct / total_n

    return all_predictions, all_labels, accuracy


def compute_metrics(all_predictions, all_labels, model_class_names):
    """Oblicza precision, recall i daje nam confusion matrix."""
    precision = precision_score(all_labels, all_predictions, average=None, zero_division=0)
    recall = recall_score(all_labels, all_predictions, average=None, zero_division=0)

    precision_macro = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_predictions, average='macro', zero_division=0)


    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro):    {recall_macro:.4f}")
    print("")

    print("Per-class:")

    for i, class_name in enumerate(model_class_names):
        print(f"{class_name:15s} | Precision: {precision[i]:.4f} | Recall: {recall[i]:.4f}")
    print("")

    cm = confusion_matrix(all_labels, all_predictions)

    return cm, precision, recall


def plot_confusion_matrix(cm, model_class_names, name, save_path=None):

    plt.figure(figsize=(12, 10))

    sns.heatmap(cm, annot=True, fmt='d', cmap='magma', xticklabels=model_class_names, yticklabels=model_class_names)
    plt.ylabel('Prawdziwa etykieta')
    plt.xlabel('Predykcja')
    plt.title(f'Confusion Matrix {name}')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Zapisano: {save_path}")

    plt.show()



if __name__ == "__main__":

    # SPEECHCOMMANDS
    print("Ładuję dataset SPEECHCOMMANDS...")
    base_data = SPEECHCOMMANDS(root=GSC_PATH, subset=None, download=True)

    # splitbuilder
    print("Robię splity...")

    dataset_indexer = SplitBuilder(base_data,
        fine_tune_max_samples_per_class=6,
        pretrain_val_ratio=0.1,
        seed=1234,
    )

    finetune_split = dataset_indexer.build_finetune_splits()


    # pretrain checkpoint
    print("Ładuję pretrain checkpoint...")

    if not PRETRAIN_CHECKPOINT.exists():
        raise FileNotFoundError(f"Nie znaleziono: {PRETRAIN_CHECKPOINT}")

    pretrain_checkpoint = torch.load(PRETRAIN_CHECKPOINT, map_location=DEVICE)
    old_speaker_id_map = pretrain_checkpoint["speaker_id_map"]
    old_label_map = pretrain_checkpoint["label_map"]
    class_names = sorted(old_label_map.keys())

    print(f"Liczba klas: {len(class_names)}")
    print("")

    # finetune checkpoint (jeśli istnieje)

    if FINETUNE_CHECKPOINT.exists():
        print("Ładuję finetune checkpoint...")
        finetune_checkpoint = torch.load(FINETUNE_CHECKPOINT, map_location=DEVICE)
        new_speaker_id_map = finetune_checkpoint["speaker_id_map"]
        has_finetune = True

    else:

        print("Brak finetune checkpointa")
        new_speaker_id_map = old_speaker_id_map
        has_finetune = False

    print("")

    # dataset testowy
    print("Tworzę test dataset...")

    test_dataset = SpeechCommandsKWS(dataset=base_data, split_indices=finetune_split["test"], allowed_speakers=finetune_split["allowed_speakers"], speaker_id_map=new_speaker_id_map,
                                     noise_dir=NOISE_PATH, silence_per_target=1.0, unknown_to_target_ratio=1.0,
                                     seed=1234)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

    print(f"test_ds: {len(test_dataset)}")
    print("")

    # Ewaluacja PRETRAINU
    print("=" * 60)
    print("PRETRAIN")
    print("=" * 60)

    num_speakers_pretrain = len(old_speaker_id_map)
    num_classes = len(old_label_map)

    model_pretrain = KWSNet(num_of_classes=num_classes, num_of_speakers=num_speakers_pretrain).to(DEVICE)
    model_pretrain.load_state_dict(pretrain_checkpoint["model_state"])

    predictions_pretrain, labels, accuracy_pretrain = evaluate(model_pretrain, test_loader, DEVICE)

    print(f"Accuracy: {accuracy_pretrain:.4f}")
    print("")

    cm_pretrain, p_pretrain, r_pretrain = compute_metrics(predictions_pretrain, labels, class_names)

    results_path = Path(BASE_PATH) / "results"
    results_path.mkdir(exist_ok=True)

    plot_confusion_matrix(cm_pretrain, class_names, name="Pretrained",  save_path=results_path / "cm_pretrain.png")

    # Ewaluacja FINETUNE (jeśli istnieje)
    if has_finetune:
        print("=" * 60)
        print("FINETUNE")
        print("=" * 60)

        num_speakers_finetune = len(new_speaker_id_map)

        model_finetune = KWSNet(num_of_classes=num_classes, num_of_speakers=num_speakers_finetune).to(DEVICE)

        model_finetune.load_state_dict(finetune_checkpoint["model_state"])

        predictions_finetune, _, accuracy_finetune = evaluate(model_finetune, test_loader, DEVICE)

        print(f"Accuracy: {accuracy_finetune:.4f}")
        print()

        cm_finetune, p_finetune, r_finetune = compute_metrics(predictions_finetune, labels, class_names)

        plot_confusion_matrix(cm_finetune,class_names, name="Finetuned", save_path=results_path / "cm_finetune.png")

        # porównanie
        print("=" * 60)
        print("PORÓWNANIE MODELU 2 FAZ")
        print("=" * 60)
        print(f"Accuracy improvement: {accuracy_finetune - accuracy_pretrain:+.4f}")
        print(f"Pretrain:  {accuracy_pretrain:.4f}")
        print(f"Finetune:  {accuracy_finetune:.4f}")
