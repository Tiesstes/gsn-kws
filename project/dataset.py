import torch
from torch import tensor, dtype
from torch.utils.data import Dataset
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio import transforms as T
from torch.nn import functional as F


class SpeechCommandsKWS(Dataset):

    def __init__(self, dataset, label_mapping, speaker_mapping, duration=1.0, sample_rate=16000, number_of_mel_bands=40):
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

        # mapowanie etykiet na numerki
        self.label_mapping = label_mapping
        # mapowanie speaker_id na numerki
        self.speaker_mapping = speaker_mapping
        # żeby ustandaryzować czas trwania
        self.target_length = int(duration * sample_rate)
        # transformacja, której będą poddane dane do wsadu dla sieci -> win_length = 30ms * 16kHz; hop_length = 10ms * 16kHz
        self.to_melspec = T.MelSpectrogram(sample_rate, n_fft=512, win_length=320,
                                                     hop_length=160, n_mels=number_of_mel_bands)
        # transformacja z liniowej na moc
        self.to_db = T.AmplitudeToDB(stype='power')

    def __len__(self):
        return len(self.base_data)

    def __getitem__(self, index):

        waveform, sample_rate, label, speaker_id, _ = self.base_data[index]

        # cięcie jeśli != duration. clone() kopiuje dane oryginalnego tensor'a
        waveform = waveform[:, :self.target_length].clone()

        # jak nagranie krótsze to padding zerami
        if waveform.shape[1] < self.target_length:
            pad = self.target_length - waveform.shape[1] # różnica między chcianą długością a realną danych
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


