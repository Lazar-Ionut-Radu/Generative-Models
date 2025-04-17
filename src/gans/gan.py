import torch
import torch.nn as nn
from src.gans.template import GANTemplate

class GAN(GANTemplate):
    def __init__(self, G, D, batch_size=64, steps_d=1):
        super().__init__(G, D, batch_size, steps_d)

    def _calculate_discriminator_loss(self, samples_real, labels_real, samples_fake, labels_fake):
        ones = torch.ones(samples_real.size(0), 1, device=self.device)
        zeros = torch.zeros(samples_fake.size(0), 1, device=self.device) 

        criterion = nn.BCELoss()
        loss_real = criterion(self.D(samples_real, labels_real), ones)
        loss_fake = criterion(self.D(samples_fake, labels_fake), zeros)
        return loss_real + loss_fake
        
    def _calculate_generator_loss(self, samples_real, labels_real, samples_fake, labels_fake):
        ones = torch.ones(samples_fake.size(0), 1, device=self.device)

        criterion = nn.BCELoss()
        loss = criterion(self.D(samples_fake, labels_fake), ones)

        return loss