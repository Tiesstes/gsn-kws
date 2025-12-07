from blocks import *


# TODO: forward function

class BCResNet1(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels

        self.head = ConvBNReLU(in_channels=self.in_channels, out_channels=16, kernel_size=(5,5),
                                      stride=(2,1), padding=(2, 2))

        # IMPORTANT: "(...) Changes in number of channels and downsampling by stride s belong to the
        # first block of each sequence of BC-ResBlocks."
        self.bcresnet_stage1 = nn.Sequential(BCResBlock(in_channels=16, out_channels=8, ssn_subbands=2,
                                                        dropout_rate=0.01,
                                                        is_transition=True),
                                             BCResBlock(in_channels=8, out_channels=8, ssn_subbands=2,
                                                        dropout_rate=0.01,
                                                        is_transition=False))


        self.bcresnet_stage2 = nn.Sequential(BCResBlock(in_channels=8, out_channels=12, ssn_subbands=2,
                                                        dropout_rate=0.01, stride=(2,1), dilation=(1, 2),
                                                        is_transition=True),
                                             BCResBlock(in_channels=12, out_channels=12, ssn_subbands=2,
                                                        dropout_rate=0.01, dilation=(1, 2),
                                                        is_transition=False))

        self.bcresnet_stage3 = nn.Sequential(BCResBlock(in_channels=12, out_channels=16, ssn_subbands=2,
                                                        dropout_rate=0.01, stride=(2,1), dilation=(1, 4),
                                                        is_transition=True),
                                             BCResBlock(in_channels=16, out_channels=16, ssn_subbands=2,
                                                        dropout_rate=0.01, dilation=(1, 4),
                                                        is_transition=False),
                                             BCResBlock(in_channels=16, out_channels=16, ssn_subbands=2,
                                                        dropout_rate=0.01, dilation=(1, 4),
                                                        is_transition=False),
                                             BCResBlock(in_channels=16, out_channels=16, ssn_subbands=2,
                                                        dropout_rate=0.01, dilation=(1, 4),
                                                        is_transition=False))

        self.bcresnet_stage4 = nn.Sequential(BCResBlock(in_channels=16, out_channels=20, ssn_subbands=2,
                                                        dropout_rate=0.01, dilation=(1, 8),
                                                        is_transition=True),
                                             BCResBlock(in_channels=20, out_channels=20, ssn_subbands=2,
                                                        dropout_rate=0.01, dilation=(1, 8),
                                                        is_transition=False),
                                             BCResBlock(in_channels=20, out_channels=20, ssn_subbands=2,
                                                        dropout_rate=0.01, dilation=(1, 8),
                                                        is_transition=False),
                                             BCResBlock(in_channels=20, out_channels=20, ssn_subbands=2,
                                                        dropout_rate=0.01, dilation=(1, 8),
                                                        is_transition=False)
                                             )

        # "(...)After the BC-ResBlocks, there is a 5x5 depthwise convolution without zero-padding in frequency dimension
        # followed by a pointwise convolution that increases the number of channels before average pooling."

        self.tail = nn.Sequential(nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(5,5),
                                   stride=(1,1), dilation=(1,1), padding=(0,2), groups=20, bias=False),
                                  ConvBNReLU(in_channels=20, out_channels=32, kernel_size=(1,1)),
                                  nn.AdaptiveAvgPool2d((1, 1)), # uśrednienie zarówno po czestotliwości jak i czasie - global
                                  nn.Conv2d(in_channels=32, out_channels=12, kernel_size=(1,1)))
                                    # czy robić jednak nie na 12 klas a na 35?^

        # tensor z aartykułu ma być: [B, 12, 1, 1]

    def forward(self, x):

        # chyba to jakoś ładniej trzeba
        y = self.head(x)
        y = self.bcresnet_stage1(y)
        y = self.bcresnet_stage2(y)
        y = self.bcresnet_stage3(y)
        y = self.bcresnet_stage4(y)
        y = self.tail(y)

        y = y.squeeze(-1).squeeze(-1) # usunięcie wymiarów, na indeksie określonym jako (-1), czyli ostatni

        return y