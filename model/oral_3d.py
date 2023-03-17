import torch
from model.generator import Encoder_MPR
from model.discriminator import PatchDiscriminator


class Oral3D(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.name = 'Oral_3D'
        self.device = device
        self.generator = Encoder_MPR()
        self.discriminator = PatchDiscriminator(device=device)

    def generate(self, input_tensor, VAL):
        generations = self.generator.generate(input_tensor, VAL)
        return generations.squeeze(1)