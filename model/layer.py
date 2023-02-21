import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from utils.local_io import scio
from utils.basic import join_path

CONV_METHOD = {'1D': nn.Conv1d, '2D': nn.Conv2d, '3D': nn.Conv3d}
TCONV_METHOD = {'1D': nn.ConvTranspose1d, '2D': nn.ConvTranspose2d, '3D': nn.ConvTranspose3d}
NORM_METHOD = {'1D': nn.InstanceNorm1d, '2D': nn.InstanceNorm2d, '3D': nn.InstanceNorm3d}


class ConvRelu(nn.Module):
    def __init__(self, in_chns, out_chns, k=3, s=1, p=None, method='2D', NORM=True, act_funct=nn.ReLU()):
        super(ConvRelu, self).__init__()
        p = (k-1)//2 if p is None else p
        self.NORM = NORM
        self.conv = CONV_METHOD[method](in_chns, out_chns, kernel_size=k, stride=s, padding=p)
        if NORM:
            self.norm = NORM_METHOD[method](out_chns)
        self.act_funct = act_funct

    def forward(self, input_tensor):
        out = self.conv(input_tensor)
        if self.NORM:
            out = self.norm(out)
        out = self.act_funct(out)
        return out


class DenseModule(nn.ModuleDict):
    def __init__(self, input_chns, growth_rate, conv_n, method='2D'):
        super(DenseModule, self).__init__()
        self.input_chns = input_chns
        for loop_id in range(conv_n):
            layer = ConvRelu(input_chns + loop_id*growth_rate, growth_rate, method=method, NORM=True)
            self.add_module('dense_%d' % loop_id, layer)

    def forward(self, input_tensor):
        features = input_tensor
        for name, layer in self.items():
            new_features = layer(features)
            features = torch.cat((new_features, features), dim=1)
        return features


class DenseBlock(nn.Module):
    def __init__(self, input_chns, output_chns, growth_rate=16, conv_n=3, method='2D'):
        super(DenseBlock, self).__init__()
        self.dense_module = DenseModule(input_chns, growth_rate, conv_n, method=method)
        self.transition = ConvRelu(input_chns + conv_n*growth_rate, output_chns, k=1, method=method)

    def forward(self, input_tensor):
        features = self.dense_module(input_tensor)
        features = self.transition(features)

        return features


class UpSampleBlock(nn.Module):
    def __init__(self, in_chns, pass_chns, out_chns=None, method='2D'):
        super(UpSampleBlock, self).__init__()
        out_chns = out_chns if out_chns else in_chns
        self.up = ConvUpSample(in_chns)
        self.res = DenseBlock(in_chns + pass_chns, out_chns, method=method)
        # self.res = DenseBlock(in_chns + pass_chns, out_chns, out_chns, method=method)

    def forward(self, input_tensor, pass_tensor):
        out = self.up(input_tensor)
        out = torch.cat((out, pass_tensor), dim=1)
        out = self.res(out)
        return out


class ConvUpSample(nn.Module):
    def __init__(self, chns, k=3, method='2D'):
        super(ConvUpSample, self).__init__()
        self.up = TCONV_METHOD[method](chns, chns, k, stride=2, padding=1, output_padding=1)

    def forward(self, input_tensor):
        return self.up(input_tensor)


class VoxelUp(nn.Module):
    def __init__(self, chns, method='2D'):
        super(VoxelUp, self).__init__()
        self.up = TCONV_METHOD[method](chns, 2*chns, kernel_size=4, stride=2, padding=1)

    def forward(self, input_tensor):
        return self.up(input_tensor)


class RandomAug(nn.Module):
    def __init__(self, bias_factor=0.1, contrast_factor=0.1, brightness_factor=0.2, gamma_factor=0.2, spine_factor=0.3):
        super(RandomAug, self).__init__()
        self.contrast_factor = contrast_factor
        self.brightness_factor = brightness_factor
        self.gamma_factor = gamma_factor
        self.bias_factor = bias_factor
        self.spine_factor = spine_factor
        spine_img = scio.loadmat(join_path('data', 'spine.mat'))['Spine']
        spine_tensor = torch.tensor(spine_img / 256, dtype=torch.float, requires_grad=False)
        self.spine_tensor = spine_tensor.unsqueeze(0).cuda(1)

    def _random_adjust(self, input_img):
        # add spine column
        input_img += self.spine_factor*self.spine_tensor

        random_bias = random.random() * self.bias_factor
        input_img = input_img + random_bias
        input_img = self._bound_img(input_img)

        # adjust brightness
        random_brightness = random.choice([-1, 1]) * random.random() * self.brightness_factor + 1
        input_img = random_brightness * input_img
        input_img = self._bound_img(input_img)

        # adjust contrast
        random_contrast = random.random() * self.contrast_factor
        input_img = (input_img - 0.5) * random_contrast + 0.5
        input_img = self._bound_img(input_img)

        # adjust gamma
        random_gamma = random.choice([-1, 1]) * random.random() * self.gamma_factor + 1
        input_img = input_img ** random_gamma

        return input_img

    @staticmethod
    def _bound_img(img):
        img[img < 0] = 0
        img[img > 1] = 1
        return img

    def forward(self, input_tensor):
        batch_size = input_tensor.size(0)
        for batch_id in range(batch_size):
            input_tensor[batch_id] = self._random_adjust(input_tensor[batch_id])
        return input_tensor