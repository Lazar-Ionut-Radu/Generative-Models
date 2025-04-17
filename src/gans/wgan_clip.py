import torch
import torch.nn as nn
from src.gans.template import GANTemplate

class WGANClip(GANTemplate):
    def __init__(self, G, D, batch_size=64, steps_d=5, clip_value=0.01):
        super(WGANClip, self).__init__(G, D, batch_size, steps_d)
        
        self.clip_value = clip_value
    
    def _calculate_discriminator_loss(self, samples_real, labels_real, samples_fake, labels_fake):
        d_real = self.D(samples_real, labels_real)
        d_fake = self.D(samples_fake, labels_fake)

        wasserstein_loss = d_fake.mean() - d_real.mean()

        return wasserstein_loss
    
    def _calculate_generator_loss(self, samples_real, labels_real, samples_fake, labels_fake):
        return -self.D(samples_fake, labels_fake).mean()
    
    def _train_discriminator_epoch(self, samples_real, labels_real, samples_fake, labels_fake):
        loss_d = super()._train_discriminator_epoch(samples_real, labels_real, samples_fake, labels_fake)
    
        # Weight clipping to the parameters of the discriminator
        for p in self.D.parameters():
            p.data.clamp_(-self.clip_value, self.clip_value)

        return loss_d
    