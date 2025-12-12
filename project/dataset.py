from collections import Counter

import torch
from torch import tensor, dtype
from torch.utils.data import Dataset
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio import transforms as T
from torch.nn import functional as F

# core words z README.MD do SC z PyTorch jest inne, ale robimy zgodnie z benchmarkiem 12 klas
TARGET_LABELS = [ "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

# na razie tu 25
AUXILIARY = [ "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila",
    "tree", "wow", "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine",
    "backward", "forward", "follow", "learn", "visual"]


class SpeechCommandsKWS(Dataset):

    def __init__(self, dataset, duration=1.0, sample_rate=16000, number_of_mel_bands=40):
        """
        Klasa dataset obsługująca GSC v.2 do projektu

        :param dataset: dataset na bazie którego zostanie zrobiony wrapper
        :param label_mapping: mapowanie etykiet na liczbę
        :param speaker_mapping: mapowanie id mówcy na liczbę
        :param duration: pożądana długość dla jednego przykładu (bo potem końcowo: sampling rate * czas trwania sygnału)
        :param sample_rate: częstotliwość próbkowania sygnału

        :return: None
        """

        self.base_data = dataset

        self.label_counter = Counter()
        self.speaker_counter = Counter()

        for _, _, label, speaker, _ in self.base_data:

            label = label.lower()  # upewnij się, że są z małej litery

            if label in TARGET_LABELS:
                mapped = label

            elif label in AUXILIARY:
                mapped = "unknown"
            else:
                mapped = "silence"

            self.label_counter[mapped] += 1
            self.speaker_counter[speaker] += 1

        # smarter podejście bez list i sortowania i setów
        all_labels = sorted(self.label_counter.keys())
        all_speakers = sorted(self.speaker_counter.keys())

        # robimy mapy: string -> int
        self.label_mapping = {}
        for i, lbl in enumerate(all_labels):
            self.label_mapping[lbl] = i

        self.speaker_mapping = {}
        for i, s_id in enumerate(all_speakers):
            self.speaker_mapping[s_id] = i

        #return label_mapping, speaker_mapping, label_counter

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

        # cięcie jeśli != duration. clone() kopiuje dane oryginalnego tensor'a, a zostawia ten z datasetu w spokoju
        waveform = waveform[:, :self.target_sample_length].clone() # kształt [kanały, czas trwania] -> kanały wszystkie, ale czas nie

        # jak nagranie krótsze to padding zerami
        if waveform.shape[1] < self.target_sample_length:
            pad = self.target_sample_length - waveform.shape[1] # różnica między chcianą długością a realną długością danych
            waveform = F.pad(waveform, (0, pad))

        # mel spektrogram i jeszcze do skali log
        mel_spectrogram = self.to_melspec(waveform)
        log_mel_spectrogram = self.to_db(mel_spectrogram)


        return {
            "log_mel_spectrogram": log_mel_spectrogram,
            "speaker_id": tensor(self.speaker_mapping[speaker_id], dtype=torch.long), # casting (rzutowanie) do tensora
            "label": tensor(self.label_mapping[label], dtype=torch.long), # casting do tensora
            "mel_spectrogram": mel_spectrogram
        }


