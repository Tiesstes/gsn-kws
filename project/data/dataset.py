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
    def __init__(self, dataset, noise_dir, duration=1.0, sample_rate=16000, number_of_mel_bands=40):
        """
        Klasa dataset obsługująca GSC v.2 do projektu

        :param dataset: dataset na bazie którego zostanie zrobiony wrapper
        :param label_mapping: mapowanie etykiet na liczbę
        :param speaker_mapping: mapowanie id mówcy na liczbę
        :param duration: pożądana długość dla jednego przykładu (bo potem końcowo: sampling rate * czas trwania sygnału)
        :param sample_rate: częstotliwość próbkowania sygnału

        :return: None
        """
        # bo po tym głupim _backround_noise_ SPEECHCOMMANDS nie iteruje oczywiście :)
        self.path = Path(noise_dir)

        noise_list = []
        for audio_file in self.path.glob("*.wav"):
            waveform, sample_rate = torchaudio.load(audio_file)
            noise_list.append((waveform, sample_rate, "noise", "none", audio_file)) # dodaje tuplę w formie, którą lubi ten dataset

        # agregujemy sobie te dane -> (waveform, sample rate, etykieta, speaker, path)
        self.base_data = list(dataset) + noise_list

        self.label_counter = Counter()
        self.speaker_counter = Counter()

        for _, _, label, speaker, _ in self.base_data:

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
        for i, lbl in enumerate(all_labels):
            self.label_mapping[lbl] = i

        self.speaker_mapping = {}
        for i, s_id in enumerate(all_speakers):
            self.speaker_mapping[s_id] = i


        # żeby ustandaryzować czas trwania
        self.target_sample_length = int(duration * sample_rate)
        # transformacja, której będą poddane dane do wsadu dla sieci -> win_length = 30ms * 16kHz; hop_length = 10ms * 16kHz
        self.to_melspec = T.MelSpectrogram(sample_rate, n_fft=512, win_length=480,
                                                     hop_length=160, n_mels=number_of_mel_bands)
        # transformacja z liniowej na moc
        self.to_db = T.AmplitudeToDB(stype='power')

    def __len__(self):
        return len(self.base_data)

    def __getitem__(self, index):

        waveform, sample_rate, label, speaker_id, _ = self.base_data[index]

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
            "speaker_id": tensor(self.speaker_mapping[speaker_id], dtype=torch.long), # casting (rzutowanie) na tensor
            "label": tensor(self.label_mapping[mapped_label], dtype=torch.long), # casting na tensor
            "mel_spectrogram": mel_spectrogram
        }
