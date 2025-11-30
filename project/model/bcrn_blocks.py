from torch import nn


class BCResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, is_transition=False):
        super(BCResBlock, self).__init__()

