import torch
import torch.nn as nn
from src.gans.relativistic_gan import RelativisticGAN

class R3GAN(RelativisticGAN):
    def __init__(self, G, D, batch_size=64, steps_d=1, lambda_r1=10, lambda_r2=10):
        super(R3GAN, self).__init__(G, D, batch_size, steps_d)

        self.lambda_r1 = 10
        self.lambda_r2 = 10
        
    def _calculate_discriminator_loss(self, samples_real, labels_real, samples_fake, labels_fake):
        # Ensure I can calculate gradients
        samples_real = samples_real.requires_grad_(True)
        samples_fake = samples_fake.requires_grad_(True)
                
        # Get the discriminator's predictions for real and fake batches
        d_real = self.D(samples_real, labels_real)
        d_fake = self.D(samples_fake, labels_fake)
        
        # Relativistic discriminator loss
        relativistic_loss = -torch.log(torch.sigmoid(d_real - d_fake)).mean()
        
        # R1 loss
        r1_reg_loss = torch.mean(torch.autograd.grad(outputs=d_real.sum(), inputs=samples_real, create_graph=True)[0] ** 2)
        
        # R2 loss
        r2_reg_loss = torch.mean(torch.autograd.grad(outputs=d_fake.sum(), inputs=samples_fake, create_graph=True)[0] ** 2)
        
        return relativistic_loss + self.lambda_r1 * r1_reg_loss / 2 + self.lambda_r2 * r2_reg_loss / 2

    def _calculate_generator_loss(self, samples_real, labels_real, samples_fake, labels_fake):
        # Get the discriminator's predictions for real and fake samples
        d_real = self.D(samples_real, labels_real) 
        d_fake = self.D(samples_fake, labels_fake)
    
        # Relativistic generator loss:
        loss_g = -torch.log(torch.sigmoid(d_fake - d_real)).mean() 
        return loss_g
