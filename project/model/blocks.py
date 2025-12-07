from typing import Literal

import torch
from torch import nn

from torch.nn import functional as F

# na wejście wchodzi tensor [B, 1, 40, W] batch, kanał, ilość pasm częstotliwości i czas trwania

AFFINE_TRANSFORMATION = Literal["all", "sub_band"]

# artykuł naukowy o SSN: https://arxiv.org/pdf/2103.13620
class SSN(nn.Module):

    def __init__(self, num_of_channels: int, num_of_subspec_bands: int, affine_transformation: AFFINE_TRANSFORMATION):

        super().__init__()

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
            # tutaj jest jak batch normalisation, ale na Ch * S kanałach -> jak w paper
            # trochę jak implementacja pojedynczego perceptronu tbh tylko zamiast 'w' jest γ, a zamiast 'b' jest β
            # to jest to samo prawie co wyżej, ale będę mieć więcej w tensorze kanałów o skalowania przez liczbę podpasm
            self.gamma = nn.Parameter(torch.ones(1, num_of_channels * self.num_of_subspec_bands, 1, 1))  # nie ma na początku skalowania
            self.beta = nn.Parameter(torch.zeros(1, num_of_channels * self.num_of_subspec_bands, 1, 1))  # nie chcemy na początku przesunięć


        else:
            raise ValueError(f"Niedozwolony rodzaj transformacji afinicznej: {affine_transformation}"
                             f"Musi to być jedna z: {AFFINE_TRANSFORMATION}")


    def forward(self, x):
        # nasz x to będzie miał postać [B, Ch, F, W] (nie wiem czemu w artykule używają W, skoro chodzi o czas)
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

            # i znowu mamy nasz tensor [B, Ch, F, W]
            x = x.view(batch, channels, frequency_range, time)
            # ze wzorku na ogólne SSN
            x = self.gamma * x + self.beta # to jest super confusing, bo to są operacje WEKTOROWE i tu jest BROADCAST!

            return x

        elif self.affine_transformation == "sub_band":
            # jak batch normalisation, każde podpasmo ma swoje gamma i beta
            x = x.view(batch, channels * s, sub_band_width, time)
            mean = x.mean([0, 2, 3]).view(1, channels * s, 1, 1)
            variance = x.var([0, 2, 3]).view(1, channels * s, 1, 1)
            # ze wzorku na normalizację ogółem
            x = (x - mean) / torch.sqrt(variance + self.epsilon)
            x = x * self.gamma + self.beta
            # i znowu mamy nasz tensor [B, Ch, F, W]
            x = x.view(batch, channels, frequency_range, time)
            # ze wzorku na ogólne SSN

            return x




class ConvBNReLu(nn.Module):
    """
    konwolucja z batch normalisation i aktywacją ReLU - w sieci jest pierwsza i przedostatnia w architekturze:

    input1: channel×frequency×time, total time steps W -> [1, 40, W]
    stride = {(2,1), 1}
    dilation = {1, 1}
    channels_out = {16, 32}

    ilość wyjść dla wymiaru = ⌊(input + 2*padding − kernel)/stride⌋ + 1, gdzie padding = k//2

    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0):
        super().__init__()

        """
        :param channels_in: kanały wejściowe
        :param channels_out: kanały wyjściowe
        :param kernel_size: rozmiar filtra
        :param stride: co ile przesuwa się filtr
        :param padding: ile dołożyć rzędów/kolumn etc. zer
        
        """

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding, dilation=1,
                      bias=False), # bias jest w batch normalisation - taka konwencja

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        def forward(self, x):

            y = self.block(x)

            return y


# TODO: blok Residual

# y = x + f2(x) + BC(f1(avgpool(f2(x))))
class BCResBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, ssn_subbands: int, dropout_rate: float, kernel_size, stride=1, padding=0,
                 dilation=1, bias=False, is_transition = False):
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
        # na razie te selfy zostawiam, może się przydadzą
        self.is_transition = is_transition
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        f1 = []
        f2 = []


        # wejście [B, Ch, F, W] batch, kanał, częstotliwość, czas!
        if self.is_transition == True and self.in_channels != self.out_channels:

            conv_1x1_f2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(1,1), bias=False)
            bn_f2 = nn.BatchNorm2d(self.out_channels)
            relu_f2 = nn.ReLU(inplace=True)

            f2.append(conv_1x1_f2)
            f2.append(bn_f2)
            f2.append(relu_f2)

            # jeśli wyjście inne niż wejście to grup depthwise będzie = out_channels
            depthwise_channels = self.out_channels

        elif self.is_transition == False and self.in_channels == self.out_channels:
            # jeśli wyjście to samo co wejście, to jest to ilość grup do depthwise (bez różnicy czy in czy out channels)
            depthwise_channels = self.in_channels

        else:
            raise ValueError("Określono jako nieodpowiedni rodzaj bloku ResNet"
                             f"kanały wejścia: {self.in_channels}, kanały wyjścia: {self.out_channels}, "
                             f"określono transition jako: {self.is_transition}")

        """
        PyTorch docs: https://docs.pytorch.org/docs/2.8/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
        When groups == in_channels and out_channels == K * in_channels, where K is a positive integer, 
        this operation is also known as a “depthwise convolution”.
        """

        # "(...) The 2D feature part, f2 consists of a 3x1 frequency-depthwise convolution
        # and SubSpectral Normalization (SSN)"
        fd_conv = nn.Conv2d(in_channels=depthwise_channels, out_channels=depthwise_channels, kernel_size=(3,1), stride=(1, 1),
                            padding=(1, 0), groups = depthwise_channels , bias=False)

        # jeśli ssn_subbands = 1 to i tak to będzie affine = "all"
        # TODO: można zmienić implementację SSN bo w sumie nie trzeba mieć affine "all"
        ssn = SSN(num_of_channels=depthwise_channels, num_of_subspec_bands=ssn_subbands, affine_transformation="sub_band")

        f2.append(fd_conv)
        f2.append(ssn)

        self.f2 = nn.Sequential(*f2)

        self.f_avg_pool = nn.AdaptiveAvgPool2d((1, None)) # [B, Ch, F, W] -> [B, Ch, 1, W] bierze F do uśrednienia, ale nie W (czas), bo tu jest podany kernel size jaki ma wyjść

        # "(...) The f1 is a composite of a 1x3 temporal depthwise convolution followed by BN,
        # swish activation [15], 1x1 pointwise convolution, and channel-wise dropout of dropout rate p."

        td_conv = nn.Conv2d(in_channels=depthwise_channels, out_channels=depthwise_channels, kernel_size=(1, 3), stride=(1, 1),
                            padding=(0, 1), groups=depthwise_channels, bias=False)
        bn_f1 = nn.BatchNorm2d(depthwise_channels)
        # Swish = SiLU: https://docs.pytorch.org/docs/2.8/generated/torch.nn.SiLU.html#torch.nn.SiLU
        swish_f1 = nn.SiLU(inplace=True)
        conv_1x1_f1 = nn.Conv2d(in_channels=depthwise_channels, out_channels=depthwise_channels, kernel_size=(1,1), bias=False)
        dropout_f1 = nn.Dropout2d(p=self.dropout_rate)
        # IMPORTANT: PyTorch sam realizuje broadcast, jeśli wymiary na to pozwalają

        f1.append(td_conv)
        f1.append(bn_f1)
        f1.append(swish_f1)
        f1.append(conv_1x1_f1)
        f1.append(dropout_f1)

        self.f1 = nn.Sequential(*f1)


    def forward(self, x):
        # wejście będzie [B, Ch, F, W]

        if self.is_transition == False:
            identity = x

            y_f2 = self.f2(x)

            x_f1 = self.f_avg_pool(y_f2) # -> [B, Ch, 1, W]

            y_f1 = self.f1(x_f1)

            # tutaj jawny Broadcast jak na obrazku budowy bloków BCResNet
            broadcasted_y_f1 = y_f1.expand_as(y_f2) # Tensor.expand_as(other) → Tensor https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.expand_as.html

            y = F.relu(identity + y_f2 + broadcasted_y_f1)

            return y

        elif self.is_transition == True:

            y_f2 = self.f2(x)

            identity_y_f2 = y_f2
            x_f1 = self.f_avg_pool(y_f2)

            y_f1 = self.f1(x_f1)

            broadcasted_y_f1 = y_f1.expand_as(y_f2)

            y = F.relu(identity_y_f2 + broadcasted_y_f1)

            return y