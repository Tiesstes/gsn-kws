import random
from collections import Counter
from pathlib import Path

import torch
import torchaudio
from torch import tensor
from torch.utils.data import Dataset
from torchaudio import transforms as T
from torch.nn import functional as F

# core words z README.MD do SC z PyTorch jest inne, ale robimy zgodnie z benchmarkiem 12 klas jak Qualcomm
TARGET_LABELS = [ "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

# na razie tu 25
AUXILIARY = [ "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila",
    "tree", "wow", "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine",
    "backward", "forward", "follow", "learn", "visual"]


class SpeechCommandsKWS(Dataset):

    def __init__(self, dataset,
                 noise_dir,
                 duration=1.0,
                 sample_rate=16000,
                 number_of_mel_bands=40,
                 silence_per_target=1.0,  # 1.0 => silence ~= średnia liczność klasy target
                 unknown_to_target_ratio=1.0,  # np. 2.0 => unknown <= 2x(średnia target)
                 seed=1234):
        """
        Klasa dataset obsługująca GSC v.2 do projektu

        :param dataset: dataset na bazie którego zostanie zrobiony wrapper
        :param label_mapping: mapowanie etykiet na liczbę
        :param speaker_mapping: mapowanie id mówcy na liczbę
        :param duration: pożądana długość dla jednego przykładu (bo potem końcowo: sampling rate * czas trwania sygnału)
        :param sample_rate: częstotliwość próbkowania sygnału

        :return: None
        """
        print("Tworzę dataset...")
        # bo po tym głupim _backround_noise_ SPEECHCOMMANDS nie iteruje oczywiście :)
        self.noise_path = Path(noise_dir)
        # żeby ustandaryzować czas trwania
        self.target_sample_length = int(duration * sample_rate)

        base_list = list(dataset) # bazowy dataset rzutowany na lisę

        # liczenie tymczasowe - żeby potem poprwić liczność
        tmp_counts = Counter()
        for _, _, label, _, _ in base_list:

            lbl = label.lower()

            if lbl in TARGET_LABELS:
                tmp_counts[lbl] += 1

            elif lbl in AUXILIARY:
                tmp_counts["unknown"] += 1

            else:
                tmp_counts["silence"] += 1

        target_counts = [tmp_counts[label] for label in TARGET_LABELS]

        avg_target_counts = int(sum(target_counts) / max(1, len(target_counts)))

        # silence tyle ile wspł * średnia liczność target
        target_silence_counts = int(silence_per_target * avg_target_counts)

        print("Policzyłem wartości tmp, przechodzę do noise_list")


        noise_list = []
        #trzeba było zmienić generację plików do silence, bo było ich tylko 6 XD
        files = list(self.noise_path.glob("*.wav"))
        counts_per_file = max(1, target_silence_counts // len(files)) # TODO: co jak nie ma plików

        # teraz wygeneruj tyle ile targetów (około)
        for audio_file in files:
            waveform, sample_rate = torchaudio.load(audio_file)  # [1, 16kHz]
            file_length = waveform.shape[1]

            for _ in range(counts_per_file):

                if file_length <= self.target_sample_length:
                    start = 0

                else:
                    start = torch.randint(0, file_length - self.target_sample_length + 1,(1,)).item() # losuję gdzie zacząć w pliku
                    
                audio_chunk = waveform[:, start:start + self.target_sample_length]
                noise_list.append((audio_chunk, sample_rate, "noise", "none", audio_file)) # na końcu źródło, ale nie jest tutaj istotne

        # random generator, żeby wybierał pliki losowo
        rng = random.Random(seed)

        print("Skończyłem generować noise, sprawdzam czy nie ma za mało")

        # dopóki nie mamy tyle ise ile chcemy mieć - w sumie można to pominąć to już mniej wpływa
        while len(noise_list) < target_silence_counts:

            audio_file = rng.choice(files)
            waveform, sample_rate = torchaudio.load(audio_file)
            file_length = waveform.shape[1]

            if file_length <= self.target_sample_length:
                start = 0
            else:
                start = rng.randrange(0, file_length - self.target_sample_length + 1)

            audio_chunk = waveform[:, start:(start + self.target_sample_length)]
            noise_list.append((audio_chunk, sample_rate, "noise", "none", audio_file))


        print("Ograniczam klasę unknown")

        # teraz ograniczyć unknown!
        max_unknown_counts = int(unknown_to_target_ratio * avg_target_counts)

        unknown_indices = []
        to_keep_indices = [] # indeksy rtykiet z datasetu których nie będziemy ruszać

        for index, item in enumerate(base_list):
            _, _, label, _, _ = item

            lbl = label.lower()

            if lbl in AUXILIARY:
                unknown_indices.append(index)

            else:
                to_keep_indices.append(index)

        if len(unknown_indices) > max_unknown_counts:

            rng.shuffle(unknown_indices)
            unknown_indices = unknown_indices[:max_unknown_counts]

        balanced = to_keep_indices + unknown_indices
        balanced.sort() # to jest tylko dla stabilności indeksowania (żeby było stałe)

        balanced_base = [base_list[index] for index in balanced]

        print("Wszystko wydaje się być okej, będę teraz liczył prawdziwą liczność")

        # no i w końcu można dataset zrobić normalnie
        # agregujemy sobie te dane -> (waveform, sample rate, etykieta, speaker, path)
        self.balanced_data = balanced_base + noise_list

        # będą nam zliczać ile czego jest do kontroli
        self.label_counter = Counter()
        self.speaker_counter = Counter()

        for _, _, label, speaker, _ in self.balanced_data:

            label = label.lower()  # upewnij się, że są z małej litery

            if label in TARGET_LABELS:
                mapped_label = label

            elif label in AUXILIARY:
                mapped_label = "unknown"
            else:
                mapped_label = "silence" # dorobić sygnały do silence

            self.label_counter[mapped_label] += 1
            self.speaker_counter[speaker] += 1

        # smarter podejście bez list i setów
        all_labels = sorted(self.label_counter.keys()) # tutaj już lecą zmapowane wartości - keys
        all_speakers = sorted(self.speaker_counter.keys())

        # robimy mapy: string -> int; dla sieci
        self.label_mapping = {}
        for index, lbl in enumerate(all_labels):
            self.label_mapping[lbl] = index

        self.speaker_mapping = {}
        for index, s_id in enumerate(all_speakers):
            self.speaker_mapping[s_id] = index


        # transformacja, której będą poddane dane do wsadu dla sieci -> win_length = 30ms * 16kHz; hop_length = 10ms * 16kHz
        self.to_melspec = T.MelSpectrogram(sample_rate, n_fft=512, win_length=480,
                                                     hop_length=160, n_mels=number_of_mel_bands)
        # transformacja z liniowej na moc
        self.to_db = T.AmplitudeToDB(stype='power')

        print("[Dystrybucja etykiet w datasecie]")
        for lbl, cnt in self.label_counter.items():
            print(f"{lbl:8s} -> {cnt:6d} próbek, id={self.label_mapping[lbl]}")


    def __len__(self):
        return len(self.balanced_data)

    def __getitem__(self, index):

        """
        :return:
            "log_mel_spectrogram": torch.Tensor
            "speaker_id": torch.Tensor int64
            "label": torch.Tensor int64
            "mel_spectrogram": mel_spectrogram: torch.Tensor
        """

        waveform, sample_rate, label, speaker_id, _ = self.balanced_data[index]

        label = label.lower() # upewnij się, że małe litery

        # zamień na etykietę o znormalizowanej nazwie
        if label in TARGET_LABELS:
            mapped_label = label
        elif label in AUXILIARY:
            mapped_label = "unknown"
        else: # ta klasa "noise" zamieni się na "silence" jak w artykułach
            mapped_label = "silence"

        # .clone() kopiuje dane oryginalnego tensor'a, a zostawia ten z datasetu w spokoju
        waveform = waveform.clone() # bierzemy teraz całe nagranie i zaraz będziemy się nim bawić w if'ach
        waveform_length = waveform.shape[1]

        # jak nagranie krótsze to padding zerami
        if waveform_length < self.target_sample_length:
            to_pad = self.target_sample_length - waveform.shape[1] # różnica między chcianą długością a realną długością danych
            waveform = F.pad(waveform, (0, to_pad))

        # każde wywołanie __getitem__ to i tak inny kawałek nagrania z noise - automatyczny "shuffle"
        elif waveform_length > self.target_sample_length:

            # losowanie jednego elementu torchem i zamiana na int
            start = torch.randint(0, waveform_length - self.target_sample_length + 1,(1,)).item()  # ten +1 to żeby indeks był maks
            waveform = waveform[:, start:start + self.target_sample_length]

        # mel spektrogram i jeszcze do skali log
        mel_spectrogram = self.to_melspec(waveform)
        log_mel_spectrogram = self.to_db(mel_spectrogram)


        return {
            "log_mel_spectrogram": log_mel_spectrogram,
            "speaker_id": tensor(self.speaker_mapping[speaker_id], dtype=torch.long), # casting (rzutowanie) przypisywanej wartości na tensor (int64)
            "label": tensor(self.label_mapping[mapped_label], dtype=torch.long), # casting na tensor
            "mel_spectrogram": mel_spectrogram
        }
