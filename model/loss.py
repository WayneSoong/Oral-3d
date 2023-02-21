import torch
import torch.nn.functional as F


def rec_loss(generation, gt):
    loss = F.mse_loss(generation, gt)
    return loss


def proj_loss(generation, gt):
    loss_1 = F.mse_loss(torch.mean(generation, dim=1), torch.mean(gt, dim=1))
    loss_2 = F.mse_loss(torch.mean(generation, dim=2), torch.mean(gt, dim=2))
    loss_3 = F.mse_loss(torch.mean(generation, dim=3), torch.mean(gt, dim=3))
    return (loss_1 + loss_2 + loss_3) / 3