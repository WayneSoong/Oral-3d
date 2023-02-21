# The network architecture comes from
# "Single-image Tomography: 3D Volumes from 2D Cranial X-Rays"

from networks.layer import *


class ResidualEncoder(nn.Module):
    def __init__(self, name='ResidualEncoder'):
        super(ResidualEncoder, self).__init__()
        self.name = name
        # encoder network
        # input: [1, 160, 576]
        self.down_block_0 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)
        # down_0: [32, 80, 288]
        self.down_block_1 = nn.Sequential(DenseBlock(32, 64, conv_n=3, growth_rate=8), nn.MaxPool2d((1, 2)),
                                          DenseBlock(64, 128, conv_n=3, growth_rate=16))
        # down_1: [256, 80, 144]
        self.down_block_2 = ResDownSample(128)
        # down_2: [256, 40, 72]
        self.down_block_3 = ResDownSample(128)
        # down_3: [256, 20, 36]
        self.down_block_4 = ResDownSample(128)
        down_4: [256, 10, 18]

        self.up_block_3 = UpSampleBlock(in_chns=128)
        # up_3: [128, 20, 36]
        self.up_block_2 = UpSampleBlock(in_chns=128)
        # up_2: [128, 40, 72]
        self.up_block_1 = UpSampleBlock(in_chns=128)
        # up_1: [128, 80, 144]
        self.up_block_0 = VoxelUp(chns=128)
        # up_0: [256, 160, 288]

    def forward(self, input_tensor):
        # down sample
        down_0 = self.down_block_0(input_tensor)
        down_1 = self.down_block_1(down_0)
        down_2 = self.down_block_2(down_1)
        down_3 = self.down_block_3(down_2)
        down_4 = self.down_block_4(down_3)

        up_3 = self.up_block_3(down_4, down_3)
        up_2 = self.up_block_2(up_3, down_2)
        up_1 = self.up_block_1(up_2, down_1)
        up_0 = self.up_block_0(up_1)
        out = torch.tanh(up_0)
        return out

    def generate(self, input_tensor, VAL=True):
        return self.forward(input_tensor)


class UpSampleBlock(nn.Module):
    def __init__(self, in_chns, pass_chns=None, out_chns=None):
        super(UpSampleBlock, self).__init__()
        out_chns = out_chns if out_chns else in_chns
        pass_chns = pass_chns if pass_chns else in_chns
        self.up = nn.ConvTranspose2d(in_chns, in_chns, 2, stride=2, padding=0)
        self.res = ResidualBlock(in_chns + pass_chns, out_chns)

    def forward(self, input_tensor, pass_tensor):
        out = self.up(input_tensor)
        out = torch.cat((out, pass_tensor), dim=1)
        out = self.res(out)
        return out


class ResDownSample(nn.Module):
    def __init__(self, chns):
        super(ResDownSample, self).__init__()
        self.res_seq = nn.Sequential(ResidualBlock(chns), ResidualBlock(chns), ResidualBlock(chns))
        self.down = nn.MaxPool2d(2)

    def forward(self, input_tensor):
        out = self.res_seq(input_tensor)
        out = self.down(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_chns, out_chns=None, block_n=3):
        super(ResidualBlock, self).__init__()
        self.block_n = block_n
        self.seq = nn.Sequential()
        self.out_chns = out_chns
        for loop_i in range(block_n):
            self.seq.add_module('res_layer%d' % loop_i, ConvRelu(in_chns, in_chns))
        if out_chns:
            self.out_layer = ConvRelu(in_chns, out_chns)

    def forward(self, input_tensor):
        out = self.seq(input_tensor)
        out += input_tensor
        if self.out_chns:
            out = self.out_layer(out)
        return out