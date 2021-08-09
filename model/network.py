"""Universal network struture unit definition."""
from torch import nn


def define_pointwise_squeeze_unit(basic_channel_size):
    """Define a 1x1 squeeze convolution with norm and activation."""
    conv = nn.Conv2d(2 * basic_channel_size, basic_channel_size, kernel_size=1,
                     stride=1, padding=0, bias=False)
    norm = nn.BatchNorm2d(basic_channel_size)
    relu = nn.LeakyReLU(0.1,inplace=False)
    layers = [conv, norm, relu]
    return layers


def define_pointwise_expand_unit(basic_channel_size):
    """Define a 1x1 expand convolution with norm and activation."""
    conv = nn.Conv2d(basic_channel_size, 2 * basic_channel_size, kernel_size=1,
                     stride=1, padding=0, bias=False)
    norm = nn.BatchNorm2d(2 * basic_channel_size)
    relu = nn.LeakyReLU(0.1,inplace=False)
    layers = [conv, norm, relu]
    return layers


def define_depthwise_unit(basic_channel_size):
    """Define a 3x3 depthwise convolution with norm and activation."""
    conv = nn.Conv2d(basic_channel_size, basic_channel_size, kernel_size=3,
                     stride=1, padding=1, bias=False,
                     groups=basic_channel_size)
    norm = nn.BatchNorm2d(basic_channel_size)
    relu = nn.LeakyReLU(0.1,inplace=False)
    layers = [conv, norm, relu]
    return layers


def define_expand_unit(basic_channel_size):
    """Define a 3x3 expand convolution with norm and activation."""
    conv = nn.Conv2d(basic_channel_size, 2 * basic_channel_size, kernel_size=3,
                     stride=1, padding=1, bias=False)
    norm = nn.BatchNorm2d(2 * basic_channel_size)
    relu = nn.LeakyReLU(0.1,inplace=False)
    layers = [conv, norm, relu]
    return layers


def define_halve_unit(basic_channel_size):
    """Define a 4x4 stride 2 expand convolution with norm and activation."""
    conv = nn.Conv2d(basic_channel_size, 2 * basic_channel_size, kernel_size=3,
                     stride=2, padding=1, bias=False)
    norm = nn.BatchNorm2d(2 * basic_channel_size)
    relu = nn.LeakyReLU(0.1,inplace=False)
    layers = [conv, norm, relu]
    return layers


def define_detector_block(basic_channel_size):
    """Define a unit composite of a squeeze and expand unit."""
    layers = []
    layers += define_pointwise_squeeze_unit(basic_channel_size)
    layers += define_expand_unit(basic_channel_size)
    return layers


class BottleneckBlock(nn.modules.Module):
    """Ideas of Inverted Residuals"""
    def __init__(self, basic_channel_size):
        super(BottleneckBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(basic_channel_size, 2*basic_channel_size, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2*basic_channel_size),
            nn.LeakyReLU(0.1,inplace=False),
            nn.Conv2d(2*basic_channel_size, basic_channel_size, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(basic_channel_size)
        )

    def forward(self, *x):
        return self.model(x[0]) + x[0]
