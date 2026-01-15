import random
from pathlib import Path
from typing import Optional, Dict
from collections import Counter
import matplotlib.pyplot as plt

import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio import transforms as T
from torch.nn import functional as F

import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio import transforms as T
from torch.nn import functional as F


# core words z README.MD do SC z PyTorch jest inne, ale robimy zgodnie z benchmarkiem 12 klas jak Qualcomm
TARGET_LABELS = [ "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

# na razie tu 25
AUXILIARY = [ "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow", "zero", "one", "two",
              "three", "four", "five", "six", "seven", "eight", "nine", "backward", "forward", "follow", "learn", "visual"]


class SpeechCommandsKWS(Dataset):

    def __init__(self, dataset, split_indices,   # dodane - lista indeksów nagrań z base dataset'u, definiująca split jaki chcemy zainstancjonować
                 allowed_speakers, # dodane - dozwolowny zbiór mówców (o bogatszej lub uboższej reprezentacji). Na jego podstawie wiadomo jakich próbek ze znormalizowanej klasy AUXILIARY nie brać
                 speaker_id_map, label_map: Optional[Dict[str, int]], noise_dir, duration=1.0, sample_rate=16000,
                 number_of_mel_bands: int = 40, silence_per_target: float= 1.0, unknown_to_target_ratio: float =1.0,
                 seed= 1234, deterministic: bool = True): # przez to będą wybierane pliki losowo lub deterministycznie (zmienione __getitem__ i _crop_or_pad)
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

        # indeksy i tak powstają dynamicznie, bo to niezależne. SplitBUilder pilnuje indeksów podziałowych

        # COLLECT UNKNOWN (SAFE) - poprawione
        auxiliary_buckets = {command: [] for command in AUXILIARY}

        self.unknown_indices = []
        for idx in range(len(self.base_dataset)):
            _, _, label, speaker, _ = self.base_dataset[idx]
            command = label.lower()

            if command in auxiliary_buckets:
                auxiliary_buckets[command].append(idx)

            """if speaker not in allowed_speakers:
                continue

            if label.lower() in AUXILIARY and speaker in allowed_speakers:
                self.unknown_indices.append(idx)
            else:
                continue"""

        # BALANCE UNKNOWN CLASS
        target_count = len(self.indices)
        max_unknown = int(unknown_to_target_ratio * (target_count / 10))

        num_of_auxiliary = len(AUXILIARY)
        max_share_per_command = max_unknown // num_of_auxiliary
        remaining = max_unknown - (max_share_per_command * num_of_auxiliary)

        unknown_indices = []

        # najpierw max na komendę z każdego kubełka
        for command in AUXILIARY:

            idxs = auxiliary_buckets[command] # indeksy danej komendy z kubełka
            self._rng.shuffle(unknown_indices) # żeby mieszać indeksy, które są aktualnie w unknown_indices
            unknown_indices.extend(idxs[:max_share_per_command]) # zwiększenie listy o max_share_per_command indeksów danej komendy

        fill = [] # temp
        if remaining > 0: # jak zostały jeszcze z maksymalnej ilości ro

            for command in AUXILIARY:
                fill.extend(auxiliary_buckets[command][max_share_per_command:])
                self._rng.shuffle(fill)

        unknown_indices.extend(fill[:remaining])

        self._rng.shuffle(self.unknown_indices)
        self.unknown_indices = unknown_indices
        print("Liczność klasy unknown (unknown_indices) :", len(self.unknown_indices))

        #SILENCE
        self.silence_count = int(silence_per_target * (target_count / 10))

        # FINAL INDEX TABLE
        self.final_indices = ([("target", i) for i in self.indices] + [("unknown", i) for i in self.unknown_indices]
                              + [("silence", None) for _ in range(self.silence_count)])

        #LABEL & SPEAKER MAP -> zamiana, żeby nie liczył indeksów zawsze, tylko mógł brać z pliku
        default_labels = TARGET_LABELS + ["unknown", "silence"]
        default_label_map = {label: index  for index, label in enumerate(default_labels)}

        if label_map is None:
            self.label_map = default_label_map

        else:
            required_labels = set(default_label_map.keys())
            provided_labels = set(label_map.keys())

            # najpierw klucze sprawdzić
            if provided_labels != required_labels:

                missing_labels = required_labels - provided_labels
                extra_labels = provided_labels - required_labels

                raise ValueError(f"label_map niezgodny; brakuje={sorted(missing_labels)}, nadmiarowe={sorted(extra_labels)}")

            # a teraz to w ogóle wartości
            for label_name, expected_idx in default_label_map.items():
                provided_idx = label_map[label_name]

                if provided_idx != expected_idx:
                    raise ValueError(f"label_map['{label_name}'] = {provided_idx}, ale powinno być {expected_idx}")

            self.label_map = label_map

        self.speaker_id_map = speaker_id_map

        # FEATURES
        self.to_melspec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=480,
            hop_length=160,
            n_mels=number_of_mel_bands)

        self.to_db = T.AmplitudeToDB(stype="power")

        print("Dataset istnieje!")

        c = Counter()
        for idx in self.unknown_indices:
            _, _, label, _, _ = self.base_dataset[idx]
            c[label.lower()] += 1
        print("Unknown per AUX label:", c)

    def __len__(self):
        return len(self.final_indices)

    def __getitem__(self, idx):

        # już silence deterministyczny zależnie od wybboru datasetu
        label, index = self.final_indices[idx]

        if label == "silence": # tworzy plik próbki losowo
            speaker_id = -1 # explicite wywalić silence z embeddingu

            if not self.deterministic:
                noise_file = self._rng.choice(self._noise_path)
                waveform, _ = torchaudio.load(noise_file)

            else:
                noise_file = self._noise_path[idx % len(self._noise_path)] # zapewnia cykliczność wyboru pliku z silence
                waveform, _ = torchaudio.load(noise_file)

        else:
            waveform, _, raw_label, speaker, _ = self.base_dataset[index]
            waveform = waveform.clone()

            if raw_label.lower() in TARGET_LABELS:
                label = raw_label.lower()
            else:
                label = "unknown"

            speaker_id = self.speaker_id_map.get(speaker, -1)  # -1 oznacza brak embedding'u, jeśli coś nie tak


        waveform = self._crop_or_pad(waveform)
        mel = self.to_melspec(waveform)
        log_mel = self.to_db(mel)

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
        # francja elegancja
        for label_type, index in self.final_indices:

            if not include_silence and label_type == "silence":
                continue

            if include_silence and label_type == "silence":
                speakers.add("unk")

            else:
                _, _, _, speaker, _ = self.base_dataset[index]
                speakers.add(speaker)

        return len(speakers)

    def visualise_logmel_per_class(self, n_cols: int = 4, figsize=(14, 10)):

        """Pokaż przykład logmel spektrogramu dla każdej z klas

        :param n_cols:
        :param figsize:

        :return: None (plotuje spektrogramy)
        """

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



class CustomWAVSpeechCommandsKWS(Dataset):
    """
    Dataset dla własnych nagrań WAV z hierarchią śceiżek: root_dir/command/*.wav

    Nazwa pliku: {speaker}_{index}.wav
    """

    def __init__(self, root_data_dir: Path, split_indices: List[int],  # indeksy próbek do tego split'u (train/val/test)
                 speaker_id_map: Dict[str, int], label_map: Dict[str, int], sample_rate: int = 16000, duration: float = 1.0,
                 number_of_mel_bands: int = 40, deterministic: bool = True, seed: int = 1234):

        self.root_dir = Path(root_data_dir)
        self.speaker_id_map = speaker_id_map
        self.label_map = label_map
        self._target_sample_length = int(duration * sample_rate)
        self.deterministic = deterministic
        self._rng = random.Random(seed)

        # Zbierz wszystkie pliki (path, label, speaker)
        all_audiofiles = []

        for label_folder in sorted(self.root_dir.iterdir()):

            if not label_folder.is_dir():
                continue

            label = label_folder.name.lower()  # np. "yes"

            if label not in self.label_map:
                print(f"[WARNING] Pomijam folder '{label}' - nie ma w label_map")
                continue

            for wav_file in sorted(label_folder.glob("*.wav")):

                speaker = self._extract_speaker_from_filename(wav_file.name)
                all_audiofiles.append((wav_file, label, speaker))

        # Wybierz tylko próbki z podanych indeksów
        self.samples = [all_audiofiles[i] for i in split_indices]

        # Mel-spektrogram (identycznie jak w SpeechCommandsKWS)
        self.to_melspec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=480,
            hop_length=160,
            n_mels=number_of_mel_bands
        )
        self.to_db = T.AmplitudeToDB(stype="power")

        print(f"CustomWavDataset: {len(self.samples)} próbek, {len(set(s for _, _, s in self.samples))} mówców")

    @staticmethod
    def _extract_speaker_from_filename(filename: str) -> str:
        """
        Zwraca speaker_id jako ekstrakcja z nazwy pliku, np. Bedoes_0.wav-> Bedoes
        """
        return filename.split("_")[0]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label, speaker = self.samples[idx]

        waveform, sample_rate = torchaudio.load(path)

        # automatyczny resampling gdyby coś
        if sample_rate != 16000:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # crop/pad do ustandaryzowanej długości
        waveform = self._crop_or_pad(waveform)

        # transforamcja logmel jak w SpeechCommandsKWS
        mel = self.to_melspec(waveform)
        log_mel = self.to_db(mel)

        # tu jest fallback, ale czy to teraz ważne? W tym wypadku nie ma zastosowania (w sumie kalka)
        speaker_id = self.speaker_id_map.get(speaker, self.speaker_id_map.get("unk", 0))
        # w każdym razie, jakby co to jest wartość 0 dla unkwknown

        return {"log_mel_spectrogram": log_mel,
                "label": torch.tensor(self.label_map[label], dtype=torch.long),
                "speaker_id": torch.tensor(speaker_id, dtype=torch.long)}

    def _crop_or_pad(self, waveform: torch.Tensor) -> torch.Tensor:
        """Kalka z SpeechCommandsKWS"""

        length = waveform.shape[1]

        if length < self._target_sample_length:
            return F.pad(waveform, (0, self._target_sample_length - length))

        if length > self._target_sample_length:

            if not self.deterministic: # to w sumie tutaj nie ma sensu, ale już można zostawić
                start = torch.randint(0, length - self._target_sample_length + 1, (1,)).item()
            else:
                start = (length - self._target_sample_length) // 2
            return waveform[:, start:start + self._target_sample_length]

        return waveform


def collect_all_custom_wavs(root_dir: Path) -> List[Tuple[Path, str, str]]:
    """
    Przegląda root_dir i zwraca listę tupli (path, label, speaker) dla wszystkich plików

    WAŻNE: Nazewnicwo jest na sztywno!
    """
    wavs = []
    for label_folder in sorted(root_dir.iterdir()):
        if not label_folder.is_dir():
            continue
        label = label_folder.name.lower()

        for wav_file in sorted(label_folder.glob("*.wav")):

            speaker = wav_file.name.split("_")[0]  # sztywne założenie naznictwa
            wavs.append((wav_file, label, speaker))

    return wavs


def split_custom_data( root_dir: Path, train_ratio: float = 0.7, val_ratio: float = 0.15,
                       test_ratio: float = 0.15, seed: int = 1234) -> Tuple[List[int], List[int], List[int]]:
    """
    Dzieli indeksy próbek na train/val/test

    :param root_dir:
    :param train_ratio:
    :param val_ratio:
    :param test_ratio: to w sumie kosmetyczne, można porem się zastanowić czy to wykorzystać
    :param seed:

    :return:
    """
    all_samples = collect_all_custom_wavs(root_dir)
    num_of_samples = len(all_samples)

    indices = list(range(num_of_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_end = int(num_of_samples * train_ratio)
    val_end = train_end + int(num_of_samples * val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    print(f"[INFORMACJAAAAAAA!!!] Custom splity train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    return train_indices, val_indices, test_indices


def extract_custom_speakers(root_dir: Path) -> set:
    """
    Zestaw nazwy mówców - na podstawie nazwy pliky ofc

    :param root_dir:

    :retrun:
    """
    speakers = set()

    for label_folder in root_dir.iterdir():

        if not label_folder.is_dir():
            continue

        for wav_file in label_folder.glob("*.wav"):
            speaker = wav_file.name.split("_")[0]
            speakers.add(speaker)

    return speakers

