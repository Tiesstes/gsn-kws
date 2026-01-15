import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Literal

import torch
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS

# SplitBuilder potrzebny, bo liczy splity (pretrain/finetune) i buduje speaker_id_map
# SpeechCommandsKWS potrzebny, bo to wrapper na bazowym SPEECHCOMMANDS, tworzy final_indices (target/unknown/silence)
# TARGET_LABELS to lista 10 naszych komend
from project.data.dataset import SpeechCommandsKWS, TARGET_LABELS

# TODO: uzupełnić docsstring i przetestować

# pomoc, żeby były stałe loadery i krótki kod
def build_dataloaders(train_ds, val_ds, test_ds=None, batch_size: int = 128,
                      num_workers: int = 0) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:

    """
    Pomocnicza funkcaj, zęby zbudować sobie loader'y

    Fazy:
    - Pretrain: train + val
    - Finetune: train + val + test
    - Evaluate: tylko test

    :param train_ds:
    :param val_ds:
    :param test_ds:
    :param batch_size:
    :param num_workers:

    :return: train dataloader, validation dataloader, test dataloader
    """
    if (train_ds is not None) and (val_ds is not None) and (test_ds is None):
        # warunek na pretrain
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        test_loader = None

    elif (train_ds is not None) and (val_ds is not None) and (test_ds is not None):
        # warunek do finetune
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    elif (test_ds is not None) and (train_ds is None) and (val_ds is None):
        # warunek na ewaluację
        train_loader = None
        val_loader = None
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    else:
        raise ValueError(f"Nieprawidłowa kombinacja datasetów do stworzenia! Nie odpowiada żadnemu z etapów eksperymentu")

    return train_loader, val_loader, test_loader



class SplitBuilder:
    """
    # Klasa, która dzieli nagrania z base dataset'u: SPEECHCOMMANDS
    # na zbiory:
    # treningowy i walidacyjny - dla mówców o reprezentacji <6 - nagrań per klasa z TARGET
    # treningowy, walidacyjny (1 próbka) i testowy (1 próbka) - dla mówców o reprezentacji >=6
    # nagrań per klasa z TARGET

    NOTE: klasa trzyma w sobie wszelkie niezbędne indeksy potrzebne do wyciągnięcia danych
    """

    def __init__(self, base_dataset, fine_tune_min_samples_per_class: int = 6, pretrain_val_ratio: float = 0.1,
                 seed: int = 1234):

        """
        :param base_dataset: dataset bazowy
        :param fine_tune_min_samples_per_class: ile maksymalnie wypowiedzi na klasę może mieć mówca trafiający do datasetu pretreningu
        :param pretrain_val_ratio: procent bazowego datasetu, który ma stanowić zbiór walidacyjny
        :param seed: ziarno losowe

        """

        self.base_dataset = base_dataset

        self.finetune_min_samples_per_class = fine_tune_min_samples_per_class # graniczna liczba nagrań dla pretreningu
        self.pretrain_val_ratio = pretrain_val_ratio
        self.seed = seed
        self._rng = random.Random(seed)

        # Słownik mówców, którego wartościami będą słowniki o parach:
        # komenda - lista
        # z indeksami nagrań dla tego mówcy dla danej klasy KW (TARGET)
        self.speaker_stats = {}
        self.all_speakers = set()
        self._collect_KW_speaker_stats()

        # od razu tworzony jest podział na grupy pretreningowe i finetuningowe (jednorazowe wywołanie funkcji zamiast kilka razy)
        self._speaker_poor, self._speaker_rich  = self._divide_speakers()
        # na tej podstawie tworzona jest też mapa przez prywatną klasę
        self._global_speaker_id_map = self._build_global_speaker_id_map()

    # zmienione na prywatną funkcję
    def _divide_speakers(self) -> tuple[set[str], set[str]]:
        """
        Metoda, która tworzy zbiory mówców na tych, którzy:
        - pójdą na pretrening (uboższa reprezentacja KW) (5- nagrań)
        - pójdą na fine-tuning (bogatsza reprezentacja KW) (6+ nagrań)

        :return: tupla zbiorw mówców
        """
        speaker_rich = set()

        # rich: tylko ci, którzy mają komplet TARGET i min_count >= K
        for speaker in self.speaker_stats:
            has_all_targets = all(lbl in self.speaker_stats[speaker] for lbl in TARGET_LABELS)
            if not has_all_targets:
                continue

            min_target = min(len(self.speaker_stats[speaker][lbl]) for lbl in TARGET_LABELS)
            if min_target >= self.finetune_min_samples_per_class:
                speaker_rich.add(speaker)

        # poor: wszyscy pozostali (w tym AUX-only)
        speaker_poor = set(self.all_speakers) - speaker_rich
        return speaker_poor, speaker_rich

    def _build_global_speaker_id_map(self):
        """
        Funkcja pomocnicza, dla splittera - wszyscy speakerzy w datasetcie
        """
        global_speaker_id_map = set(self.all_speakers)
        speaker_ids_map = {speaker: i for i, speaker in enumerate(sorted(global_speaker_id_map))}

        return speaker_ids_map

    # metoda statyczne, nie używamy na instancji
    @staticmethod
    def build_speaker_id_map_from_speakers(speakers: set[str]):

        speakers = set(speakers)
        speakers.update({"unk"})

        speaker_map_from_speakers = {speaker: i for i, speaker in enumerate(sorted(speakers))}

        return speaker_map_from_speakers

    def _collect_KW_speaker_stats(self):
        """
        Metoda do wyznaczenia słownika speaker stats

        speaker_stats[raw_speaker_id][raw_label]
        """
        for idx in range(len(self.base_dataset)):
            _, _, label, speaker, _ = self.base_dataset[idx]

            self.all_speakers.add(speaker)

            label = label.lower()

            if label in TARGET_LABELS:
                if speaker not in self.speaker_stats:
                    self.speaker_stats[speaker] = {}
                if label not in self.speaker_stats[speaker]:
                    self.speaker_stats[speaker][label] = []
                self.speaker_stats[speaker][label].append(idx)


    # podziały na zbiory pretreningowe i fine-tune

    def build_pretrain_splits(self) -> dict:
        """
        Metoda, która na podstawie zbioru mówców o uboższej reprezentacji
        dokonuje podziału na zbiory: treningowy i walidacyjny, ale także
        przekazuje zbiór mówców o uboższej reprezentacji
        (potem dla custom SpeechCommandsKWS) i statystyki

        :return: "train": indeksy treningowe,
                 "val": indeksy val,
                 "allowed_speakers": set speakerów,
                 "stats": dict mówca - liczba nagrań
        """
        audiofile_indices = []  # Lista z indeksami nagrań z klas TARGET mówców o uboższej reprezentacji

        # Dla każdego mówcy,
        for speaker in self._speaker_poor:

            if speaker not in self.speaker_stats:
                continue
            # dla każdej listy z indeksami nagrań z poszczególnych klas TARGET
            for idxs in self.speaker_stats[speaker].values():
                # zbierz te indeksy razem do jednej listy
                audiofile_indices.extend(idxs)

        # Wymieszaj zebrane indeksy
        self._rng.shuffle(audiofile_indices)
        # Podziel na 90% trening, 10% walidacja, (tu jest truncation)
        split = int((1.0 - self.pretrain_val_ratio) * len(audiofile_indices))

        return {
            "train": audiofile_indices[:split],
            "val": audiofile_indices[split:],
            "allowed_speakers": self._speaker_poor,
            "stats": self._speaker_statistics(self._speaker_poor)
        }

    def build_finetune_splits(self) -> dict:
        """
        Metoda, która na podstawie zbioru mówców o bogatszej reprezentacji
        dokonuje podziału na zbiory: testowy, walidacyjny i treningowy, ale także
        przekazuje zbiór mówców o bogatszej reprezentacji
        (potem dla custom SpeechCommandsKWS) i statystyki

        :return: "train": indeksy treningowe,
                "val": indeksy val,
                "test": indeksy test,
                "allowed_speakers": set speakerów,
                "stats": dict mówca - liczba nagrań
        """

        train, val, test = [], [], []

        # Dla każdego mówcy
        for speaker in self._speaker_rich:
            # Dla każdej pary klasa z TARGET - lista indeksów nagrań
            for label, idxs in self.speaker_stats[speaker].items():

                # Jeśli reprezentacja klasy dla danego mówcy jest mniejsza niż zdefiniowana, pomiń
                if len(idxs) < self.finetune_min_samples_per_class:
                    continue

                # wymieszaj kolejność indeksów nagrań
                self._rng.shuffle(idxs)
                # dodaj 1 próbkę do zbioru walidacyjnego, 1 do testowego i resztę do treningowego
                val.append(idxs[0])
                test.append(idxs[1])
                train.extend(idxs[2:])

        return {
            "train": train,
            "val": val,
            "test": test,
            "allowed_speakers": self._speaker_rich,
            "stats": self._speaker_statistics(self._speaker_rich)}


    def _speaker_statistics(self, speakers) -> dict:
        """
        Metoda wyznaczająca słownik o parach:
        mówca ze zbioru (uboższej lub bogatszej reprezentacji) - liczba jego nagrań z klas TARGET

        :param speakers: zbiór mówców danej reprezentacji wyznaczany przez metodę divide_speakers
        :return: statystyki mówców
        """
        stats = {}
        for speaker in speakers:

            speaker_dict = self.speaker_stats.get(speaker, {})  # WAŻNE
            stats[speaker] = {lbl: len(speaker_dict.get(lbl, [])) for lbl in TARGET_LABELS}

        return stats

    @property
    def speaker_poor(self):
        return self._speaker_poor

    @property
    def speaker_rich(self):
        return self._speaker_rich


def build_label_map() -> Dict[str, int]:
    """
    Deterministyczność etykiet!

    Za to label_map w pliku to:
    -  źródło prawdy
    - łatwo wymusić, by dataset używał zapisanej mapy
    """
    labels = list(TARGET_LABELS) + ["unknown", "silence"]
    return {label: i for i, label in enumerate(labels)}


# to jest w ogóle funkcja analogiczna do tej z architektury, tylko potrzebna do zapisywania plików gdzie indziej
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


def prepare_splits_manifest(data_manifest_path: Path, gsc_dataset_path: Path, noise_dir: Path, seed: int = 1234,
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

    # mapa speakerów (teraz podział na tych co pretrain i finetune)
    # speaker_id_map_global = split_builder.speaker_id_map
    speaker_id_map_pretrain = split_builder.build_speaker_id_map_from_speakers(pretrain_split["allowed_speakers"])

    # przy finetune mapa speakerów finetune (ta bpgata reprezentacja)
    # dodawani do istniejącej
    speaker_id_map_finetune = extend_speaker_id_map(speaker_id_map_pretrain, finetune_split["allowed_speakers"])

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
            "gsc_dataset_path": str(gsc_dataset_path),
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
            "speaker_id_map_pretrain": speaker_id_map_pretrain,
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


def build_phase_datasets(manifest: Dict[str, Any], phase: Literal["pretrain", "finetune"],
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
    base_data = SPEECHCOMMANDS(root=config["gsc_dataset_path"], subset=None, download=True)

    # mapa speakerów zależy od etapu:
    # pretrain używa pretrain mapy zbudowanej przez SplitBuilder (wcześniej global),
    # finetune używa mapy rozszerzonej (żeby nowe ID speakerów były spójne).
    if phase == "pretrain":
        speaker_id_map = maps["speaker_id_map_pretrain"]

    elif phase == "finetune":
        speaker_id_map = maps["speaker_id_map_finetune"]

    else:
        raise ValueError(f"unknown stage: {phase}")

    # parametry takie same dla dataset'ów różnych faz eksperymentu
    common_kwargs = dict(dataset=base_data, allowed_speakers=phase_splits["allowed_speakers"], speaker_id_map=speaker_id_map,
                         noise_dir=config["noise_dir"], duration=config["duration"],
                         sample_rate=config["sample_rate"], number_of_mel_bands=config["number_of_mel_bands"],
                         silence_per_target=config["silence_per_target"], unknown_to_target_ratio=config["unknown_to_target_ratio"],
                         seed=config["seed"], label_map=maps["label_map"])


    # niedeterministyczny (więc losowy crop / losowy "silence") ** to rozpakowanie klucza i wartości
    train_dataset = SpeechCommandsKWS(split_indices=phase_splits["train_indices"], deterministic=False, **common_kwargs)

    # deterministyczny zawsez
    val_dataset = SpeechCommandsKWS(split_indices=phase_splits["val_indices"], deterministic=deterministic, **common_kwargs)

    test_dataset: Optional[SpeechCommandsKWS]
    test_dataset = None

    # opcjonalny, o tylko przy finetune
    if phase == "finetune":
        test_dataset = SpeechCommandsKWS(split_indices=phase_splits["test_indices"], deterministic=deterministic, **common_kwargs)


    return train_dataset, val_dataset, test_dataset
