import numpy as np
import math
from scipy.ndimage.morphology import binary_fill_holes


def get_psnr(img_1, img_2, SCALE=2.0):
    mse = np.mean((img_1 - img_2)**2)
    psnr = 20 * np.log10(SCALE / np.sqrt(mse))
    return psnr


def get_iou(img_1, img_2):
    mask_1 = img_1 > 0
    mask_2 = img_2 > 0
    IOU = np.sum(np.logical_and(mask_1, mask_2)) / np.sum(np.logical_or(mask_1, mask_2))
    return IOU


def get_dice(img_1, img_2):
    mask_1 = img_1 > 0
    mask_2 = img_2 > 0
    dice = 2 * np.sum(np.logical_and(mask_1, mask_2)) / (np.sum(mask_1) + np.sum(mask_2))
    return dice


def mse2distance(mse):
    return math.sqrt(mse)*math.sqrt(288**2+256**2)