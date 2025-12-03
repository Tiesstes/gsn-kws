from typing import Literal

import torch
from torch import nn

# na wejście wchodzi tensor [B, 1, 40, W] batch, kanał, ilość pasm częstotliwości i czas trwania

AFFINE_TRANSFORMATION = Literal["all", "sub_band"]

# https://arxiv.org/pdf/2103.13620
class SSN(nn.Module):

    def __init__(self, num_of_channels, num_of_subspec_bands, affine_transformation: AFFINE_TRANSFORMATION):
        """
        SubSpectral Normalisation - batch notmalisation, ale oddzielnie dla każdego podpasma częstotliwości.
        Clue programu to podzielić częstotliwości jednej mapy cech (liczba channels) na podpasma i dla każdego liczyć
        z osobna normalizację, ale to wszystko jest wciąż w ramach JEDNEJ mapy cech

        WAŻNE: SSN(num_of_subspec_bands=1, AFFINE_TRANSFORMATION=All) tożsame z BATCHNORM
               SSN(num_of_subspec_bands=1, AFFINE_TRANSFORMATION=Sub) tożsame z BATCHNORM

        :param num_of_channels: ile jest kanałów wejściowych
        :param num_of_subspec_bands: na ile podpasm ma być podzielone wejście
        :param affine_transformation: jaki typ SSN: {"all", "sub_band"}

        """
        super().__init__()

        self.num_of_channels = num_of_channels
        self.num_of_subspec_bands = num_of_subspec_bands
        self.affine_transformation = affine_transformation
        self.epsilon = 1e-6 # jak w liczbach dualnych hehe, ale chyba nie od tego się to wzięło

        # te gamma i beta mają postać [1, kanały, 1, 1, 1] - w zależności od potrzeb, wymiary = 1
        # są powielane (broadcastowane) po odpowiednich osiach np.: batch czy time
        if affine_transformation == "all":
            # po całej częstotliwości jest jedna gamma i jedna beta:
            self.gamma = nn.Parameter(torch.ones(1, num_of_channels, 1, 1)) # nie ma na początku skalowania
            self.beta = nn.Parameter(torch.zeros(1, num_of_channels, 1, 1)) # nie chcemy na początku przesunięć

        elif affine_transformation == "sub_band":
            # tutaj jest jak batch normalisation, ale na C * S kanałach -> jak w paper
            # trochę jak implementacja pojedynczego perceptronu tbh tylko zamiast 'w' jest γ, a zamiast 'b' jest β
            # to jest to samo prawie co wyżej, ale będę mieć więcej w tensorze kanałów o skalowania przez liczbę podpasm
            self.gamma = nn.Parameter(torch.ones(1, num_of_channels * self.num_of_channels, 1, 1))  # nie ma na początku skalowania
            self.beta = nn.Parameter(torch.zeros(1, num_of_channels * self.num_of_channels, 1, 1))  # nie chcemy na początku przesunięć


        else:
            raise ValueError(f"Niedozwolony rodzaj transformacji afinicznej: {affine_transformation}"
                             f"Musi to być jedna z: {AFFINE_TRANSFORMATION}")


    def forward(self, x):
        # nasz x to będzie miał postać [B, C, F, W] (nie wiem czemu w artykule używają W, skoro chodzi o czas)
        # no dobrze, w takim razie, zgodnie z SNN chcemy podzielić F na S (liczba podpasm) i dostaniemy szerokość podpasma

        batch, channels, frequency_range, time = x.shape # bierzemy wymiary naszego tensora
        s = self.num_of_subspec_bands

        if frequency_range % s != 0: # jeśli częstotliwość widma nie jest podzielne przez ilość podpasm to dupa
            raise ValueError(f"Częstotliwości F = {frequency_range} nie są podzielne przez "
                             f"wybraną, do SSN, liczbę podpasm = {s}!")

        # liczymy szerokość pasma subspektralnego
        sub_band_width = frequency_range // s

        # i teraz ta magia z artykułu
        # F trzeba podzielone na S i to jest to co wyżej, czyli nasza szerokość podpasma
        # zamieniamy ten x co wszedł na tensor, gdzie mamy więcej kanałów, ale częstotliwości tylko tyle ile podpasmo
        # czyli jak zmniejszyliśmy jeden wymiar /S to musimy gdzieś pomnożyć *S -> nie może wyparować w eter
        x = x.view(batch, channels * s, sub_band_width, time)

        # to, co się dzieje poniżej, można zrobić z batch normalisation (bo SSN to specjalny przypadek BN), ale tu jest
        # implementacja pokazująca podstawowe rozumienie BN

        if self.affine_transformation == "all":

            # nie kombinujemy z oddzielnymi gamma i beta, tylko zwykły BN
            mean = x.mean([0, 2, 3]).view(1, channels * s, 1, 1)
            variance = x.var([0, 2, 3]).view(1, channels * s, 1, 1)
            # ze wzorku na normalizację ogółem
            x = (x - mean) / torch.sqrt(variance + self.epsilon)

            # i znowu mamy nasz tensor [B, C, F, W]
            x = x.view(batch, channels, frequency_range, time)
            # ze wzorku na ogólne SSN
            x = self.gamma * x + self.beta # to jest super confusing, bo to są operacje WEKTOROWE i tu jest BROADCAST!

            return x

        elif self.affine_transformation == "sub_band":
            # jak batch normalisation, każde podpasmo ma swoje gamma i beta
            x = x.view(batch, channels * s, frequency_range, time)
            mean = x.mean([0, 2, 3]).view(1, channels * s, 1, 1)
            variance = x.var([0, 2, 3]).view(1, channels * s, 1, 1)
            # ze wzorku na normalizację ogółem
            x = (x - mean) / torch.sqrt(variance + self.epsilon)
            x = x * self.gamma + self.beta
            # i znowu mamy nasz tensor [B, C, F, W]
            x = x.view(batch, channels, frequency_range, time)
            # ze wzorku na ogólne SSN

            return x




class ConvBNReLu(nn.Module):
    """
    konwolucja z batch normalisation i aktywacją ReLU - w sieci jest pierwsza i przedostatnia w architekturze

    input: channel×frequency×time, total time steps W -> [1, 40, W]
    stride = {(2,1), 1}
    dilation = {1, 1}
    channels_out = {16, 32}

    ilość wyjść dla wymiaru = ⌊(input + 2*padding − kernel)/stride⌋ + 1, gdzie padding = k//2

    """
    def __init__(self, channels_in: int, channels_out: int, kernel_size, stride=1, padding=0):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channels_in,
                      out_channels=channels_out,
                      kernel_size=kernel_size, stride=stride, padding=padding, dilation=1,
                      bias=False), # bias jest w batch normalisation - taka konwencja

            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

        def forward(self, x):

            y = self.block(x)

            return y


# TODO: blok Residual

class BCResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, is_transition=False):
        super().__init__()

        """
        Podstawowy blok dla sieci ResNet
        
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        :param is_transition:
        
        """

        # jak jest mismatch a nie powiedziano, że przejściowy to błąd
        if is_transition == False and in_channels != out_channels:
            raise ValueError("Określono jako blok nieprzejściowy, ale powinien być przejściowy!"
                             f"kanały wejścia: {in_channels}, kanały wyjścia: {out_channels}")

        self.is_transition = is_transition
        self.in_channels = in_channels
        self.out_channels = out_channels

        if is_transition == True:
            pass

        elif is_transition == False:
            pass
