# Chengxi Chu, Universiti Malaya
from torch import nn
from torch.nn import functional as F
import torch


class Resblk(nn.Module):

    def __init__(self, input_channels, output_channels, strid=1):
        super(Resblk, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = strid
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=strid, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channels)
        )
        self.shortcut = nn.Sequential()
        if input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=strid, padding=0),
                nn.BatchNorm2d(output_channels)
            )

    def forward(self, x):
        """

        :param x:
        :return:
        """
        out = self.conv1(x)
        out = self.conv2(out)
        return out + self.shortcut(x)


# temp = torch.randn(2, 3, 32, 32)
# model = Resblk(3, 64, 2)
# output = model(temp)
# print(output.shape)


class ResNet10(nn.Module):

    def __init__(self, input_channels):
        super(ResNet10, self).__init__()
        self.input_channels = input_channels
        # [b, in_ch, 32, 32] => [b, 64, 32, 32]
        self.conv_pre = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # [b, 64, 32, 32] => [b, 128, 16, 16]
        self.blk1 = Resblk(64, 128, 2)
        # [b, 128, 16, 16] => [b, 256, 8, 8]
        self.blk2 = Resblk(128, 256, 2)
        # [b, 256, 8, 8] => [b, 512, 4, 4]
        self.blk3 = Resblk(256, 512, 2)
        # [b, 512, 4, 4] => [b, 1024, 2, 2]
        self.blk4 = Resblk(512, 1024, 2)

        self.avgpooling = nn.AdaptiveAvgPool2d([1, 1])
        self.fc_layer = nn.Linear(1024, 10)

    def forward(self, x):
        """

        :param x: [b, input_channels, 32, 32]
        :return: [b, 10]
        """
        x = self.conv_pre(x)
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = self.avgpooling(x)
        x = x.view(x.size(0), 1024)
        x = self.fc_layer(x)
        return x


# temp = torch.randn(2, 3, 32, 32)
# model = ResNet10(3)
# out = model(temp)
# print(out.shape)

def generate_model(model_name, input_channels):
    if model_name == 'ResNet10':
        model = ResNet10(input_channels)
    else:
        raise 'out of options'

    return model






