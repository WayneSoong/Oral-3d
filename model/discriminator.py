import torch
import torch.nn as nn
from model.layer import ConvRelu
import random


# Patch Discriminator
class PatchDiscriminator(nn.Module):
    def __init__(self, patch_n=5, cuda_id=None):
        super(PatchDiscriminator, self).__init__()
        self.name = 'Patch'
        self.patch_n = patch_n
        self.chns = [16, 32, 64, 128]
        self.patch_window = 70
        self.cuda_id = cuda_id

        # 70
        self.layer_1 = ConvNormRelu(1,           self.chns[0], k=4, s=2, p=0)
        # 34
        self.layer_2 = ConvNormRelu(self.chns[0], self.chns[1], k=4, s=2, p=0)
        # 16
        self.layer_3 = ConvNormRelu(self.chns[1], self.chns[2], k=4, s=2, p=0)
        # 7
        self.layer_4 = ConvNormRelu(self.chns[2], self.chns[3], k=4, s=1, p=0)
        # 4
        self.layer_5 = nn.Conv3d(self.chns[3], 1, kernel_size=4)
        # 1

    def discriminate(self, gr_tensor, gt_tensor):
        gr_patches, gt_patches = self.get_patches(gr_tensor, gt_tensor)
        gr_prediction = self.forward(gr_patches)
        gt_prediction = self.forward(gt_patches)
        loss_gr = (gr_prediction - torch.zeros(self.patch_n).cuda(self.cuda_id))**2
        loss_gt = (gt_prediction - torch.ones(self.patch_n).cuda(self.cuda_id))**2
        return (loss_gr.mean() + loss_gt.mean())/2

    def inference(self, gr_tensor):
        # update for generator
        gr_patches = self.get_patches(gr_tensor)[0]
        gr_prediction = self.forward(gr_patches)
        loss_gr = (gr_prediction - torch.ones(self.patch_n).cuda(self.cuda_id))**2
        return loss_gr.mean()

    def forward(self, patches):
        """
        :param patches: [patch_size, d, h, w]
        :return: a tensor with shape [patch_size]
        """

        features = self.layer_1(patches)
        features = self.layer_2(features)
        features = self.layer_3(features)
        features = self.layer_4(features)
        features = self.layer_5(features)
        return torch.sigmoid(features).view(self.patch_n)

    def get_patches(self, *tensors):
        tensor_shape = tensors[0].shape
        tensor_n = len(tensors)
        # patches = [torch.zeros(size=[self.patch_n, 1] + [self.patch_window]*3).cuda(self.cuda_id)]*tensor_n
        random_list = []
        for id in range(self.patch_n):
            random_id = random.randint(0, tensor_shape[0] - 1)
            random_x = random.randint(0, tensor_shape[1] - 1 - self.patch_window)
            random_y = random.randint(0, tensor_shape[2] - 1 - self.patch_window)
            random_z = random.randint(0, tensor_shape[3] - 1 - self.patch_window)
            random_list.append([random_id, random_x, random_y, random_z])

        patches_list = []
        for tensor in tensors:
            patches = [self.sample_tensor(tensor, random_info) for random_info in random_list]
            patches_list.append(torch.stack(patches))
        return patches_list

    def sample_tensor(self, tensor, random_info):
        random_id, random_x, random_y, random_z = random_info
        tensor_patch = tensor[random_id, random_x:random_x+self.patch_window,
               random_y:random_y+self.patch_window, random_z:random_z+self.patch_window]
        return tensor_patch.unsqueeze(0)


class ConvNormRelu(nn.Module):
    def __init__(self, in_chns, out_chns, k=3, s=1, p=None):
        super(ConvNormRelu, self).__init__()
        p = (k-1)//2 if p is None else p

        self.conv = nn.Conv3d(in_chns, out_chns, kernel_size=k, stride=s, padding=p)
        self.norm = nn.BatchNorm3d(out_chns)
        self.relu = nn.LeakyReLU()

    def forward(self, input_tensor):
        out = self.conv(input_tensor)
        out = self.norm(out)
        out = self.relu(out)
        return out
