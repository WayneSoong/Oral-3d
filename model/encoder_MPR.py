from model.layer import *
import torch.nn as nn
import torch


class Encoder_MPR(nn.Module):
    def __init__(self, name='Encoder_MPR'):
        super(Encoder_MPR, self).__init__()
        self.name = name
        # input: [1, 160, 576]
        self.down_block_0 = ConvRelu(1, 10)
        # down_0: [10, 160, 576]
        self.down_block_1 = nn.Sequential(DenseBlock(10, 20, conv_n=3, growth_rate=10), nn.MaxPool2d(2))
        # down_1: [20, 80, 288]
        self.down_block_2 = nn.Sequential(DenseBlock(20, 40, conv_n=3, growth_rate=20), nn.MaxPool2d(2))
        # down_2: [40, 40, 144]
        self.down_block_3 = nn.Sequential(DenseBlock(40, 80, conv_n=3, growth_rate=40), nn.MaxPool2d(2))
        # down_3: [80, 20, 72]

        self.up_block_3 = UpSampleBlock(in_chns=80, pass_chns=40, out_chns=80)
        # up_2: [80, 40, 144]
        self.up_block_2 = UpSampleBlock(in_chns=80, pass_chns=20, out_chns=80)
        # up_1: [80, 80, 288]
        self.up_block_1 = UpSampleBlock(in_chns=80, pass_chns=10, out_chns=80)
        # up_0: [80, 160, 576]
        self.up_block_0 = nn.Conv2d(in_channels=80, out_channels=80, padding=1, kernel_size=3)

    def forward(self, input_tensor):
        input_tensor = input_tensor / 256
        # down sample
        down_0 = self.down_block_0(input_tensor)
        down_1 = self.down_block_1(down_0)
        down_2 = self.down_block_2(down_1)
        down_3 = self.down_block_3(down_2)

        up_3 = self.up_block_3(down_3, down_2)
        up_2 = self.up_block_2(up_3, down_1)
        up_1 = self.up_block_1(up_2, down_0)
        up_0 = self.up_block_0(up_1)
        out = torch.tanh(up_0)
        return out

    def generate(self, input_tensor, VAL=True):
        return self.forward(input_tensor)
