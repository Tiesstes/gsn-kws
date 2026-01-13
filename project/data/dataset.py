import random
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio import transforms as T
from torch.nn import functional as F

# TODO: dostosować dataset żeby mógł korzystać z data_split.py

# core words z README.MD do SC z PyTorch jest inne, ale robimy zgodnie z benchmarkiem 12 klas jak Qualcomm
TARGET_LABELS = [ "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

# na razie tu 25
AUXILIARY = [ "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila",
    "tree", "wow", "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine",
    "backward", "forward", "follow", "learn", "visual"]


class SplitBuilder:
    """
    # Klasa, która dzieli nagrania z base dataset'u: SPEECHCOMMANDS
    # na zbiory:
    # treningowy i walidacyjny - dla mówców o reprezentacji <6 - nagrań per klasa z TARGET
    # treningowy, walidacyjny (1 próbka) i testowy (1 próbka) - dla mówców o reprezentacji >=6
    # nagrań per klasa z TARGET

    NOTE: klasa trzyma w sobie wszelkie niezbędne indeksy potrzebne do wyciągnięcia danych
    """

    def __init__(self, base_dataset,
            fine_tune_min_samples_per_class: int = 6,
            pretrain_val_ratio: float = 0.1,
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
        self.speaker_id_map = self._build_global_speaker_id_map()

    # zmienione na prywatną funkcję
    def _divide_speakers(self) -> tuple[set[str], set[str]]:
        """
        Metoda, która tworzy zbiory mówców na tych, którzy:
        - pójdą na pretrening (uboższa reprezentacja KW) (5- nagrań)
        - pójdą na fine-tuning (bogatsza reprezentacja KW) (6+ nagrań)
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

        global_speaker_id_map = set(self.all_speakers)
        global_speaker_id_map.update({"none", "unk"})  # dla silence
        speaker_ids_map = {speaker: i for i, speaker in enumerate(sorted(global_speaker_id_map))}

        return speaker_ids_map

    # na razie niepotrzebna
    def build_speaker_id_map_from_speakers(self, speakers: set[str]):
        speakers = set(speakers)
        speakers.update({"none", "unk"})
        return {speaker: i for i, speaker in enumerate(sorted(speakers))}

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



class SpeechCommandsKWS(Dataset):

    def __init__(
        self, dataset,
        split_indices,   # dodane - lista indeksów nagrań z base dataset'u, definiująca split jaki chcemy zainstancjonować
        allowed_speakers, # dodane - dozwolowny zbiór mówców (o bogatszej lub uboższej reprezentacji). Na jego podstawie wiadomo jakich próbek ze znormalizowanej klasy AUXILIARY nie brać
        speaker_id_map,
        noise_dir,
        duration=1.0,
        sample_rate=16000,
        number_of_mel_bands: int = 40,
        silence_per_target: float= 1.0,
        unknown_to_target_ratio: float =1.0,
        seed= 1234,
        deterministic: bool = False # przez to będą wybierane pliki losowo lub deterministycznie (zmienione __getitem__ i _crop_or_pad)
     ):
        """
        Klasa dataset obsługująca GSC v.2 do projektu

        :param dataset: dataset na bazie którego zostanie zrobiony wrapper
        :param split_indices: lista indeksów nagrań z dataset'u, definiująca split jakiego chcemy dokonać (pretreningowego/postreningowego)
        :param speaker_id_map: mapowanie id mówcy na liczbę
        :param noise_dir: ścieżka do folderu gdzie znajdują się pliki z nagraniami klasy "silence"
        :param duration: pożądana długość dla jednego przykładu (bo potem końcowo: sampling rate * czas trwania sygnału)
        :param sample_rate: częstotliwość próbkowania sygnału
        :param number_of_mel_bands: liczba pasm melowych
        :param silence_per_target: mnożnik klasy "silence" względem średniej liczności klas targetowych
        :param unknown_to_target_ratio: mnożnik klasy "unknown" względem średniej liczności klas targetowych
        :param seed: ziarno generatora losowego
        :param deterministic: czy dane mają być deterministyczne (dla walidacji i ewaluacji tak)

        :return: None
        """
        print("Tworzę dataset...")

        self.base_dataset = dataset
        self.indices = list(split_indices)
        self.allowed_speakers = allowed_speakers
        self._rng = random.Random(seed)
        # żeby ustandaryzować czas trwania
        self._target_sample_length = int(duration * sample_rate)
        self._noise_path = sorted(Path(noise_dir).glob("*.wav")) # posortuj
        self.deterministic = deterministic

        # COLLECT UNKNOWN (SAFE)
        self.unknown_indices = []
        for idx in range(len(self.base_dataset)):
            _, _, label, speaker, _ = self.base_dataset[idx]
            #if speaker not in allowed_speakers:
                #continue
            if label.lower() in AUXILIARY and speaker in allowed_speakers:
                self.unknown_indices.append(idx)
            else:
                continue

        # BALANCE UNKNOWN CLASS
        target_count = len(self.indices)
        max_unknown = int(unknown_to_target_ratio * (target_count / 10))
        self._rng.shuffle(self.unknown_indices)
        self.unknown_indices = self.unknown_indices[:max_unknown]

        #SILENCE
        self.silence_count = int(silence_per_target * (target_count / 10))

        # FINAL INDEX TABLE
        self.final_indices = (
            [("target", i) for i in self.indices]
            + [("unknown", i) for i in self.unknown_indices]
            + [("silence", None) for _ in range(self.silence_count)]
        )

        #LABEL & SPEAKER MAP
        labels = TARGET_LABELS + ["unknown", "silence"]
        self.label_map = {lbl: i for i, lbl in enumerate(labels)}

        self.speaker_id_map = speaker_id_map

        # FEATURES
        self.to_melspec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=480,
            hop_length=160,
            n_mels=number_of_mel_bands,
        )
        self.to_db = T.AmplitudeToDB(stype="power")

        print("Dataset istnieje!")

    def __len__(self):
        return len(self.final_indices)

    def __getitem__(self, idx):

        # słabo tylko, że silence jest niedeterministyczny dla nas
        label, index = self.final_indices[idx]

        if label == "silence": # tworzy plik próbki losowo
            if not self.deterministic:
                noise_file = self._rng.choice(self._noise_path)
                waveform, _ = torchaudio.load(noise_file)
            else:
                noise_file = self._noise_path[idx % len(self._noise_path)] # zapewnia cykliczność wyboru pliku z silence
                waveform, _ = torchaudio.load(noise_file)

            speaker = None # speaker zmieniony na none

        else:
            waveform, _, raw_label, speaker, _ = self.base_dataset[index]
            waveform = waveform.clone()

            if raw_label.lower() in TARGET_LABELS:
                label = raw_label.lower()
            else:
                label = "unknown"


        waveform = self._crop_or_pad(waveform)
        mel = self.to_melspec(waveform)
        log_mel = self.to_db(mel)

        speaker_id = self.speaker_id_map.get(speaker, self.speaker_id_map["unk"])
        return {
            "log_mel_spectrogram": log_mel,
            "label": torch.tensor(self.label_map[label], dtype=torch.long),
            "speaker_id": torch.tensor(speaker_id, dtype=torch.long)
        }

    # UTILS
    def _crop_or_pad(self, waveform: torch.Tensor) -> torch.Tensor:

        length = waveform.shape[1]

        if length < self._target_sample_length:
            return F.pad(waveform, (0, self._target_sample_length - length))

        if length > self._target_sample_length:

            if not self.deterministic:
                start = torch.randint(0, length - self._target_sample_length + 1, (1,)).item()
            else:
                start = (length - self._target_sample_length) // 2

            return waveform[:, start:start + self._target_sample_length]

        return waveform

    def number_of_speakers(self, include_silence: bool = False) -> int:

        speakers = set()

        for label_type, index in self.final_indices:
            if (not include_silence) and label_type == "silence":
                continue
            if label_type == "silence":
                speakers.add("none")
            else:
                _, _, _, speaker, _ = self.base_dataset[index]
                speakers.add(speaker)
        return len(speakers)

    def visualise_logmel_per_class(self, n_cols: int = 4, figsize=(14, 10)):


        id_to_name = {value: key for key, value in self.label_map.items()} # ids do nazw etykiet
        label_names = list(self.label_map.keys())  # TARGET + unknown + silence

        # wylosuj 1 przykłąd na każdą klasę
        example_per_class = {}  # label_name -> (idx, logmel_2d)

        for i in range(len(self)):
            sample = self[i]
            label_id = int(sample["label"].item())
            label_name = id_to_name[label_id]

            if label_name not in example_per_class:
                logmel_spectrogram = sample["log_mel_spectrogram"]  # to jest tensor [1, n_mels, T]
                np_logmel_spectrogram = logmel_spectrogram.squeeze(0).detach().cpu().numpy()  # [n_mels, T]
                example_per_class[label_name] = (i, np_logmel_spectrogram)

            if len(example_per_class) == len(label_names):
                break

        # siatka
        n = len(label_names)
        n_rows = (n + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        for ax_i, label_name in enumerate(label_names):
            ax = axes[ax_i]
            if label_name not in example_per_class:
                ax.set_title(f"{label_name} (brak)")
                ax.axis("off")
                continue

            idx, np_logmel_spectrogram = example_per_class[label_name]
            ax.imshow(np_logmel_spectrogram, origin="lower", aspect="auto")
            ax.set_title(f"{label_name} (idx={idx})")
            ax.set_xlabel("Time")
            ax.set_ylabel("Mels")

        for j in range(n, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()
