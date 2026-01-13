from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Literal

import torch
from torchaudio.datasets import SPEECHCOMMANDS

# SplitBuilder potrzebny, bo liczy splity (pretrain/finetune) i buduje speaker_id_map
# SpeechCommandsKWS potrzebny, bo to wrapper na bazowym SPEECHCOMMANDS, tworzy final_indices (target/unknown/silence)
# TARGET_LABELS to lista 10 naszych komend
from dataset import SplitBuilder, SpeechCommandsKWS, TARGET_LABELS

# TODO: uzupełnić docsstring i przetestować

def build_label_map() -> Dict[str, int]:
    """
    Deterministyczność etykiet!
    IMPORTANT: Obecnie SpeechCommandsKWS i tak buduje self.label_map w swoim __init__

    Za to label_map w pliku to:
    -  źródło prawdy
    - łatwo wymusić, by dataset używał zapisanej mapy
    """
    labels = list(TARGET_LABELS) + ["unknown", "silence"]
    return {label: i for i, label in enumerate(labels)}


# to jest w ogóle funkcja analogiczna do tej z architektury, tylko potrzebna do zapisywania plików
def extend_speaker_id_map(old_map: Dict[str, int], new_speakers: set[str]) -> Dict[str, int]:
    """
    Dodaje nowe identyfikatory mówców do mapy

    - liczba speakerów wpływa na rozmiar embeddingu w modelu
    - jeżeli speaker_id_map się zmienia pomiędzy eksperymentami, embeddingi dostaną inne indeksy i
    to może psuć reprodukowalność

    :param old_map: słownik stawrego mapowania
    :param new_speakers: zestaw nowych speakerów

    :return: Nowa mapa speakerów
    """
    new_map = dict(old_map)
    next_id = (max(new_map.values()) + 1) if new_map else 0

    # sorted(...) dba o powtarzalność kolejności
    for speaker in sorted(new_speakers):
        if speaker not in new_map:
            new_map[speaker] = next_id
            next_id += 1

    return new_map


def prepare_splits_manifest(*, data_manifest_path: Path, gsc_dataset_path: Path, noise_dir: Path, seed: int = 1234,
                            finetune_min_samples_per_class: int = 6, pretrain_val_ratio: float = 0.1, duration: float = 1.0,
                            sample_rate: int = 16000, number_of_mel_bands: int = 40, silence_per_target: float = 1.0,
                            unknown_to_target_ratio: float =1.0) -> Dict[str, Any]:
    """
    Tworzy i zapisuje manifest (plik .pt) opisujący eksperyment.

    Manifest zawiera:
    - parametry datasetu i ścieżki
    - gotowe indeksy (pretrain/finetune) oraz allowed_speakers
    - label_map oraz speaker_id_map (globalną i wersję dla finetune)

    :param data_manifest_path: ścieżka do pliku manifestu
    :param gsc_dataset_path: ścieżka do datasetu Google Speech Commands
    :param noise_dir: ścieżka do folderu z noise (silence)
    :param seed:
    :param finetune_min_samples_per_class:
    :param pretrain_val_ratio:
    :param duration:
    :param sample_rate:
    :param number_of_mel_bands:
    :param silence_per_target:
    :param unknown_to_target_ratio:

    :return:
    """

    # najpierw bazowy dataset. Splity odnoszą się do tego datasetu po indeksie
    base_data = SPEECHCOMMANDS(root=str(gsc_dataset_path), subset=None, download=True)

    # SplitBuilder liczy statystyki mówców i dzieli ich na speaker_poor/speaker_rich; zamiast za każdym razem w skrypcie
    split_builder = SplitBuilder(base_dataset=base_data,fine_tune_min_samples_per_class=finetune_min_samples_per_class,
                                 pretrain_val_ratio=pretrain_val_ratio,seed=seed)

    # tu splity jak dotychczas w pretrain/py, finetune.py etc
    pretrain_split = split_builder.build_pretrain_splits()
    finetune_split = split_builder.build_finetune_splits()

    # mapa speakerów
    speaker_id_map_global = split_builder.speaker_id_map

    # przy finetune mapa speakerów finetune (ta bpgata reprezentacja)
    # dodawani do istniejącej
    speaker_id_map_finetune = extend_speaker_id_map(speaker_id_map_global, finetune_split["allowed_speakers"])

    # manifest właściwy
    # taki plik config z eksperymentu
    manifest: Dict[str, Any] = {

        "config": {

            "seed": seed,
            "fine_tune_min_samples_per_class": finetune_min_samples_per_class,
            "pretrain_val_ratio": pretrain_val_ratio,
            "duration": duration,
            "sample_rate": sample_rate,
            "number_of_mel_bands": number_of_mel_bands,
            "silence_per_target": silence_per_target,
            "unknown_to_target_ratio": unknown_to_target_ratio,
            "gsc_path": str(gsc_dataset_path),
            "noise_dir": str(noise_dir)},

        # tutaj indeksy
        "splits": {

            "pretrain": {
                # deterministycznie, bo wyliczone raz
                "train_indices": pretrain_split["train"],
                "val_indices": pretrain_split["val"],
                # Zbiór speakerów dopuszczonych w tej fazie (wpływa m.in. na unknown_indices).[file:70]
                "allowed_speakers": pretrain_split["allowed_speakers"]
            },

            "finetune": {
                "train_indices": finetune_split["train"],
                "val_indices": finetune_split["val"],
                "test_indices": finetune_split["test"],
                "allowed_speakers": finetune_split["allowed_speakers"]
            },
        },

        "maps": {
            "label_map": build_label_map(),
            "speaker_id_map_global": speaker_id_map_global,
            "speaker_id_map_finetune": speaker_id_map_finetune}
    }

    # zapisywanie
    data_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(manifest, data_manifest_path)

    return manifest


def load_splits_manifest(data_manifest_path: Path) -> Dict[str, Any]:
    """
    Wczytuje manifest splitów

    :param data_manifest_path:

    :return:
    """
    loaded_manifest = torch.load(data_manifest_path, map_location="cpu")

    return loaded_manifest


def build_phase_datasets(*, manifest: Dict[str, Any], phase: Literal["pretrain", "finetune"],
                         deterministic: bool = True) -> Tuple[SpeechCommandsKWS, SpeechCommandsKWS, Optional[SpeechCommandsKWS]]:
    """
    Datasety dla eksperymentów
    - phase="pretrain"  -> zwraca (train_ds, val_ds, None)
    - phase="finetune"  -> zwraca (train_ds, val_ds, test_ds)

    :param manifest:
    :param phase:
    :param deterministic:

    :return: train_ds, val_ds, (test_ds | None)
    """

    config = manifest["config"]
    phase_splits = manifest["splits"][phase]
    maps = manifest["maps"]

    # zawsze ten sam (ta sama wersja, ta sama ścieżka root),
    # bo split_indices to indeksy do tego datasetu i może się wywrócić
    base_data = SPEECHCOMMANDS(root=config["gsc_path"], subset=None, download=True)

    # mapa speakerów zależy od etapu:
    # - pretrain używa globalnej mapy zbudowanej przez SplitBuilder,
    # - finetune używa mapy rozszerzonej (żeby nowe ID speakerów były spójne).
    if phase == "pretrain":
        speaker_id_map = maps["speaker_id_map_global"]

    elif phase == "finetune":
        speaker_id_map = maps["speaker_id_map_finetune"]

    else:
        raise ValueError(f"unknown stage: {phase}")

    # parametry takie same dla dataset'ów różnych faz eksperymentu
    common_kwargs = dict(dataset=base_data,
        allowed_speakers=phase_splits["allowed_speakers"],
        speaker_id_map=speaker_id_map,
        noise_dir=config["noise_dir"],
        duration=config["duration"],
        sample_rate=config["sample_rate"],
        number_of_mel_bands=config["number_of_mel_bands"],
        silence_per_target=config["silence_per_target"],
        unknown_to_target_ratio=config["unknown_to_target_ratio"],
        seed=config["seed"],
        label_map=maps["label_map"])

    # niedeterministyczny (więc losowy crop / losowy "silence")
    train_dataset = SpeechCommandsKWS(split_indices=phase_splits["train_indices"], deterministic=False, **common_kwargs)

    # deterministyczny zawsez
    val_dataset = SpeechCommandsKWS(split_indices=phase_splits["val_indices"], deterministic=deterministic, **common_kwargs)

    test_dataset: Optional[SpeechCommandsKWS]
    test_dataset = None

    # opcjonalny, o tylko przy finetune
    if phase == "finetune":
        test_dataset = SpeechCommandsKWS(split_indices=phase_splits["test_indices"], deterministic=deterministic, **common_kwargs)


    return train_dataset, val_dataset, test_dataset
