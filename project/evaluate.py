import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from torchaudio.datasets import SPEECHCOMMANDS

from project.data.dataset import CustomWAVSpeechCommandsKWS, split_custom_data
from project.model.kws_net import KWSNet

import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

# konfiguracje
BASE_PATH = Path(__file__).resolve().parent
GSC_PATH = BASE_PATH / "data"

CUSTOM_DATA_ROOT = GSC_PATH / "SpeechCommands" / "custom_speech_commands_v0.02"
NOISE_PATH = GSC_PATH / "SpeechCommands" / "speech_commands_v0.02" / "_background_noise_"
DATASPLIT_MANIFEST_PATH = BASE_PATH / "data" / "splits" / "experiment_v1.pt"

IMPORT_EPOCH_FINETUNE = 36
IMPORT_EPOCH_PRETRAIN = 30

PRETRAIN_IMPORT_CHECKPOINT = Path(BASE_PATH) / "model" / f"pretrain_checkpoint_{IMPORT_EPOCH_PRETRAIN}.pt"
FINETUNE_IMPORT_CHECKPOINT = Path(BASE_PATH) / "model" / f"finetune_checkpoint_{IMPORT_EPOCH_FINETUNE}.pt"

BATCH_SIZE = 64
WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)
if torch.cuda.is_available():
    print("CUDA:", torch.cuda.get_device_name(0))
print("Torch version:", torch.__version__)
print()


def evaluate(model, data_loader, device):
    """
    Ewaluuj model na zbiorze testowym, zwracaj predykcje i etykiety
    Tutaj jest ewaluacja pretrain ignoruje speaker embedding zupełnie
    """

    model.eval()
    all_predictions, all_labels = [], []
    total_correct, total_n = 0, 0

    with torch.no_grad():

        for batch in data_loader:

            x = batch["log_mel_spectrogram"].to(device)
            y = batch["label"].to(device)

            speaker = batch["speaker_id"].to(device)
            logits = model(x, speaker)
            predictions = logits.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            total_correct += int((predictions == y).sum())
            total_n += x.shape[0]

    return all_predictions, all_labels, total_correct / total_n


def compute_metrics(all_predictions, all_labels, model_class_names):
    """Oblicza precision, recall i daje nam confusion matrix."""
    precision = precision_score(all_labels, all_predictions, average=None, zero_division=0)
    recall = recall_score(all_labels, all_predictions, average=None, zero_division=0)

    print(f"Precision (macro): {precision_score(all_labels, all_predictions, labels=list(range(len(model_class_names) - 1)),average='macro', zero_division=0):.4f}")
    print(f"Recall (macro):    {recall_score(all_labels, all_predictions, labels=list(range(len(model_class_names) - 1)) ,average='macro', zero_division=0):.4f}\n")

    for i, class_name in enumerate(model_class_names):
        if class_name == "silence":
            continue
        print(f"{class_name:15s} | Precision: {precision[i]:.4f} | Recall: {recall[i]:.4f}")

    return confusion_matrix(all_labels, all_predictions, labels=list(range(len(model_class_names) - 1))), precision, recall


def plot_confusion_matrix(cm, model_class_names, name, save_path=None):

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma', xticklabels=model_class_names, yticklabels=model_class_names)
    plt.ylabel('Prawdziwa etykieta')
    plt.xlabel('Predykcja')
    plt.title(f'Confusion Matrix {name}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Zapisano: {save_path}")

    plt.show()


if __name__ == "__main__":

    # SPEECHCOMMANDS
    print("Ładuję dataset SPEECHCOMMANDS...")
    os.makedirs(GSC_PATH, exist_ok=True)
    base_data = SPEECHCOMMANDS(root=GSC_PATH, subset=None, download=True)

    # splitbuilder
    # zamiast generować splity na nowo, ładujemy manifest
    print("Wczytuję manifest ze splitami...")
    if not DATASPLIT_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Nie znaleziono manifestu: {DATASPLIT_MANIFEST_PATH}")

    split_manifest = torch.load(DATASPLIT_MANIFEST_PATH, map_location=DEVICE)
    finetune_split = split_manifest["splits"]["finetune"]  # zmienna z manifestu

    # pretrain checkpoint
    print("Ładuję pretrain checkpoint...")
    if not PRETRAIN_IMPORT_CHECKPOINT.exists():
        raise FileNotFoundError(f"Nie znaleziono: {PRETRAIN_IMPORT_CHECKPOINT}")

    pretrain_checkpoint = torch.load(PRETRAIN_IMPORT_CHECKPOINT, map_location=DEVICE)
    old_speaker_id_map = pretrain_checkpoint["speaker_id_map"]
    old_label_map = pretrain_checkpoint["label_map"]

    # koeniczene żeby dało się porównywać
    class_names = sorted(old_label_map.keys(), key=old_label_map.get)

    print(f"Liczba klas: {len(class_names)}")
    print("")


    # finetune checkpoint (jeśli istnieje)
    if FINETUNE_IMPORT_CHECKPOINT.exists():

        print("Ładuję finetune checkpoint...")
        finetune_checkpoint = torch.load(FINETUNE_IMPORT_CHECKPOINT, map_location=DEVICE)
        new_speaker_id_map = finetune_checkpoint["speaker_id_map"]
        has_finetune = True

    else:

        print("Brak finetune checkpointa")
        new_speaker_id_map = old_speaker_id_map
        has_finetune = False

    print("")

    # dataset testowy
    print("Tworzę test dataset...")
    _, _, test_indices = split_custom_data(CUSTOM_DATA_ROOT, seed=1234)
    # pełna mapa finetune
    test_dataset = CustomWAVSpeechCommandsKWS(root_data_dir=CUSTOM_DATA_ROOT,split_indices=test_indices,
                                              speaker_id_map=new_speaker_id_map, label_map=split_manifest["maps"]["label_map"],
                                              deterministic=True)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

    print(f"Dłgiść datasetu treningowego: {len(test_dataset)}")
    print("")

    # Ewaluacja PRETRAINU
    print("~~" * 100)
    print("PRETRAIN")
    print("~~" * 100)

    num_speakers_pretrain = len(old_speaker_id_map)
    num_classes = len(old_label_map)

    model_pretrain = KWSNet(num_of_classes=num_classes, num_of_speakers=num_speakers_pretrain).to(DEVICE)
    model_pretrain.load_state_dict(pretrain_checkpoint["model_state"], strict=False)

    # oszukanie datasetu, bo został zainicjalizowany na finetune
    test_loader.dataset.speaker_id_map = old_speaker_id_map
    # maska w KWSNet, model z pretrain zignoruje nowych mówców z testu (nie znalazł go mapie)
    predictions_pretrain, labels, accuracy_pretrain = evaluate(model_pretrain, test_loader, DEVICE)

    print(f"Accuracy: {accuracy_pretrain:.4f}")
    print("")

    cm_pretrain, p_pretrain, r_pretrain = compute_metrics(predictions_pretrain, labels, class_names)

    results_path = Path(BASE_PATH) / "results"
    results_path.mkdir(exist_ok=True)

    plot_confusion_matrix(cm_pretrain, class_names, name="Pretrained", save_path=results_path / "cm_pretrain.png")

    # FINETUNE jak jest
    if has_finetune:

        print("~~" * 100)
        print("FINETUNE")
        print("~~" * 100)

        num_speakers_finetune = len(new_speaker_id_map)

        # MODYFIKACJA: Zwiększamy embedding (ensure_num_of_speakers) przed ładowaniem wag FT
        model_finetune = KWSNet(num_of_classes=num_classes, num_of_speakers=num_speakers_pretrain).to(DEVICE)
        model_finetune.ensure_num_of_speakers(num_speakers_finetune)
        model_finetune.load_state_dict(finetune_checkpoint["model_state"])

        predictions_finetune, _, accuracy_finetune = evaluate(model_finetune, test_loader, DEVICE)

        print(f"Accuracy: {accuracy_finetune:.4f}")
        print()

        cm_finetune, p_finetune, r_finetune = compute_metrics(predictions_finetune, labels, class_names)
        plot_confusion_matrix(cm_finetune, class_names, name="Finetuned", save_path=results_path / "cm_finetune.png")

        # porównanie
        print("~~" * 100)
        print("PORÓWNANIE Z 2 FAZ")
        print("~~" * 100)
        print(f"Różnica w accuracy: {accuracy_finetune - accuracy_pretrain:+.4f}")
        print(f"Pretrain:  {accuracy_pretrain:.4f}")
        print(f"Finetune:  {accuracy_finetune:.4f}")