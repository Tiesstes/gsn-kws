from torch import nn


class SubSpectralNormalisation(nn.Module):

    def __init__(self):
        super().__init__()

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


class BCResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, is_transition=False):
        super().__init__()

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
