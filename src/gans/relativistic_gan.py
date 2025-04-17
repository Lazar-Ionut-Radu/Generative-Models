import torch
import torch.nn as nn
from src.gans.template import GANTemplate

class RelativisticGAN(GANTemplate):
    def __init__(self, G, D, batch_size=64, steps_d=1):
        super(RelativisticGAN, self).__init__(G, D, batch_size, steps_d)

        # Samples the real distribution for training the Generator
        self.n_real_batches += 1
        
    def _sample_for_generator(self, iter_real, batch_size):
        # Sample a real batch (if there are any)
        try:
            batch_real = next(iter_real)
            samples_real = batch_real[0].to(self.device)
            labels_real = batch_real[1].to(self.device)
        except StopIteration:
            return False, None, None, None, None

        # Sample a fake batch 
        samples_fake, labels_fake = self.G.generate(n=samples_real.size(0), labels=None)
        
        return True, samples_real, labels_real, samples_fake, labels_fake
    
    def _calculate_discriminator_loss(self, samples_real, labels_real, samples_fake, labels_fake):
        # Get the discriminator's predictions for real and fake batches
        d_real = self.D(samples_real, labels_real)
        d_fake = self.D(samples_fake, labels_fake)
        
        # Relativistic discriminator loss
        loss_d = -torch.log(torch.sigmoid(d_real - d_fake)).mean()
        return loss_d

    def _calculate_generator_loss(self, samples_real, labels_real, samples_fake, labels_fake):
        # Get the discriminator's predictions for real and fake samples
        d_real = self.D(samples_real, labels_real)
        d_fake = self.D(samples_fake, labels_fake)
    
        # Relativistic generator loss:
        loss_g = -torch.log(torch.sigmoid(d_fake - d_real)).mean() 
        return loss_g

