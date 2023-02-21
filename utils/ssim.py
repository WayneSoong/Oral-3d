import torch
import torch.nn.functional as F
import numpy as np
import math


def mse_loss(gr, gt):
    loss = F.mse_loss(gr, gt)
    return loss


class MSE_loss(torch.nn.Module):
    def __init__(self, scalar=1):
        super(MSE_loss, self).__init__()
        self.scalar = scalar

    def forward(self, input_tensor, gr_tensor, weight_map=None):
        error_sq = (input_tensor - gr_tensor)**2
        if weight_map:
            error_background = torch.mean(error_sq[weight_map == 0])
            error_foreground = torch.mean(error_sq[weight_map > 0])
            scaled_error = (error_background + self.scalar*error_foreground)/(1+self.scalar)
        else:
            scaled_error = error_sq.mean()
        return scaled_error


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, sigma=1.5, size_average=True, cuda_id=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.cuda_id = cuda_id
        self.window = self.get_gaussian_kernel([window_size] * 3, sigma)

    def convert2tensor(self, img):
        # expand channel
        img_tensor = torch.tensor(np.expand_dims(img, axis=0), dtype=torch.float)
        if self.cuda_id is not None:
            img_tensor = img_tensor.cuda(self.cuda_id)
        return img_tensor.unsqueeze(1)

    def get_ssim_loss(self, gr, gt):
        gr = (gr + 1) / 2
        gt = (gt + 1) / 2

        gr = gr.unsqueeze(0)
        gt = gt.unsqueeze(0)

        ssim = self._get_ssim(gr, gt)
        return 1-ssim

    def _get_ssim(self, gr_tensor, gt_tensor):
        mu1 = self.local_conv(gr_tensor)
        mu2 = self.local_conv(gt_tensor)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)

        mu1_mu2 = mu1 * mu2

        sigma1_sq = self.local_conv(gr_tensor * gr_tensor) - mu1_sq
        sigma2_sq = self.local_conv(gt_tensor * gt_tensor) - mu2_sq
        sigma12 = self.local_conv(gr_tensor * gt_tensor) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim = ssim_map.mean()
        return ssim

    def eval_ssim(self, gr, gt):
        """
        :parameter
        gr: generation result
        gt: ground truth
        """
        gr = self.convert2tensor(gr)
        gt = self.convert2tensor(gt)
        ssim = self._get_ssim(gr, gt)

        if self.cuda_id is not None:
            ssim = ssim.cpu().numpy()
        return ssim

    def local_conv(self, input_tensor):
        result = F.conv3d(input_tensor, self.window, padding=self.window_size // 2)
        return result

    def get_gaussian_kernel(self, window_size, sigma):
        d, h, w = window_size
        center_x = (w - 1) / 2
        center_y = (h - 1) / 2
        center_z = (d - 1) / 2
        weights = np.zeros(window_size)
        denominator = math.sqrt(2 * math.pi) * sigma
        for id_x in range(w):
            for id_y in range(h):
                for id_z in range(d):
                    distance = (id_x - center_x) ** 2 + (id_y - center_y) ** 2 + (id_z - center_z) ** 2
                    weight = math.exp(-distance / (2 * sigma ** 2)) / (denominator ** 2)
                    weights[id_z, id_y, id_x] = weight

        weights = np.expand_dims(weights / np.sum(weights), 0)
        weights = np.expand_dims(weights, 0)

        weights = torch.tensor(weights, dtype=torch.float)
        if self.cuda_id is not None:
            weights = weights.cuda(self.cuda_id)
        return weights
