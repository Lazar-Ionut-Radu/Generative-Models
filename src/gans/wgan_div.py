import torch
import torch.nn as nn
from src.gans.template import GANTemplate

class WGANDiv(GANTemplate):
    def __init__(self, G, D, batch_size=64, steps_d=1, k=2, p=6):
        super(WGANDiv, self).__init__(G, D, batch_size, steps_d)
        
        # Parameters of the regularization term
        self.k = k
        self.p = p

    def _sample_for_discriminator(self, iter_real):
        # Sample a real batch (if there are any)
        try:
            batch_real = next(iter_real)
            samples_real = batch_real[0].to(self.device)
            labels_real = batch_real[1].to(self.device)
        except StopIteration:
            return False, None, None, None, None

        # Sample a fake batch with the same labels as the real samples.
        samples_fake, labels_fake = self.G.generate(n=samples_real.size(0), labels=labels_real)
        
        return True, samples_real, labels_real, samples_fake, labels_fake
    
    def _calculate_regularization_term(self, samples_real, labels_real, samples_fake, labels_fake):
        batch_size = samples_real.shape[0]
        
        # Get random interpolation between real and fake samples
        alpha = torch.rand(batch_size, 1, 1, 1, device=samples_real.device)
        interpolates = (alpha * samples_real + (1 - alpha) * samples_fake).requires_grad_(True)
        d_interpolates = self.D(interpolates, labels_real)

        # Get the gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Compute gradient penalty
        grad = gradients.view(batch_size, -1)
        grad_norm = grad.norm(2, dim=1)
        reg = self.k * (grad_norm ** self.p).mean()

        return reg

    def _calculate_discriminator_loss(self, samples_real, labels_real, samples_fake, labels_fake):
        d_real = self.D(samples_real, labels_real)
        d_fake = self.D(samples_fake, labels_fake)

        wasserstein_loss = d_fake.mean() - d_real.mean()
        regularization = self._calculate_regularization_term(samples_real, labels_real, samples_fake, labels_fake)

        return wasserstein_loss + regularization

    def _calculate_generator_loss(self, samples_real, labels_real, samples_fake, labels_fake):
        return -self.D(samples_fake, labels_fake).mean()
