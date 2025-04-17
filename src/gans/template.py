import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

class GeneratorTemplate(nn.Module):
    def __init__(self, data_shape, latent_dim=100):
        super(GeneratorTemplate, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_shape = data_shape
        self.optim = None
        self.latent_dim = latent_dim

    def _get_random_labels(self, n):
        return None # [TODO]: Add checks for no labels in the other methods.
    
    def _get_z(self, n):
        return torch.randn(n, self.latent_dim, device=self.device)
    
    def forward(self, z, labels=None):
        raise NotImplementedError("You need to implement this in your generator subclass.")

    def generate(self, n=1, labels=None):
        z = self._get_z(n)

        if labels is None:
            labels = self._get_random_labels(n)
            
        return self.forward(z, labels), labels
    
    # [TODO]: Allow for labels=None
    # The fact that I don't allow for labels=None rn is not a problem in reality.
    # If I work with a GAN that is not conditional the 'labels' parameter will just
    # be useless to the forward function.
    # However, somebody building a GAN using this template might use a dataset that
    # does not provide labels, in which case they, slightly annoyingly, need to add
    # useless ones just for compatibility. 
    # [NOTE]: Is there a more elegant way to call 'self.generate()' in batches?
    def get_dataloader(self, labels, batch_size=64):
        self.eval()
        
        samples = []
        labels = labels.clone()
        
        with torch.no_grad():
            num_samples = labels.size(0)
            num_full_batches = num_samples // batch_size
            remainder = num_samples % batch_size

            # Generate the full batches
            for i in range(num_full_batches):
                batch_labels = labels[i*batch_size:(i+1)*batch_size]
                batch_samples, _ = self.generate(n=batch_labels.size(0), labels=batch_labels)
                samples.append(batch_samples)

            # Generate the incomplete batch (if needed)
            if remainder > 0:
                batch_labels = labels[-remainder:]
                batch_samples, _ = self.generate(n=batch_labels.size(0), labels=batch_labels)
                samples.append(batch_samples)

        samples = torch.cat(samples, dim=0)
        return DataLoader(TensorDataset(samples, labels), batch_size=batch_size, shuffle=False)

class DiscriminatorTemplate(nn.Module):
    def __init__(self, data_shape):
        super(DiscriminatorTemplate, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optim = None
        self.data_shape = data_shape

    def forward(self, samples, labels=None):
        pass

class GANTemplate(object):
    def __init__(self, G, D, batch_size=64, steps_d=1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.G = G
        self.D = D
        self.batch_size = batch_size
        self.steps_d = steps_d
        
        # Number of batches that need to be sampled from the dataset in order
        # to complete one iteration inside 'self._train_epoch()'.
        # Some GANs may or may not sample the real distribution when updating 
        # the Generator. Only needed for the tqdm progress bar.
        self.n_real_batches = self.steps_d

    def _calculate_discriminator_loss(self, samples_real, labels_real, samples_fake, labels_fake):
        raise NotImplementedError("You need to implement this in your GAN subclass.")

    def _calculate_generator_loss(self, samples_real, labels_real, samples_fake, labels_fake):
        raise NotImplementedError("You need to implement this in your GAN subclass.")

    def _train_discriminator_epoch(self, samples_real, labels_real, samples_fake, labels_fake):
        # Calculate loss & step the discriminator's optimizer
        self.D.optim.zero_grad()
        loss_d = self._calculate_discriminator_loss(samples_real, labels_real, samples_fake, labels_fake)
        loss_d.backward()
        self.D.optim.step()
        
        return loss_d.item()
    
    def _train_generator_epoch(self, samples_real, labels_real, samples_fake, labels_fake):
        # Calculate loss & step the generator's optimizer
        self.G.optim.zero_grad()
        loss_g = self._calculate_generator_loss(samples_real, labels_real, samples_fake, labels_fake)
        loss_g.backward()
        self.G.optim.step()
        
        return loss_g.item()

    # [NOTE]: 'iter_real' is here if I need to sample a batch of real data. Regardless, the fake batch
    # is generated using random labels. Should probably be None by default or something else?
    # I don't think so because, in the cases it is needed, None is never an appropriate value. Maybe I 
    # should raise such an error for the programmer to know to look for it in the training procedure.
    def _sample_for_generator(self, iter_real, batch_size):
        # Sample a batch of fake data, with random labels (or None)
        samples_fake, labels_fake = self.G.generate(n=batch_size)
        
        return True, None, None, samples_fake, labels_fake
    
    def _sample_for_discriminator(self, iter_real):
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
    
    def validate(self, dataset_real):
        # Init dataloaders for real and generated samples.
        valid_labels = torch.tensor([label for _, label in dataset_real]).to(self.device)
        dataloader_fake = self.G.get_dataloader(valid_labels, batch_size=self.batch_size)
        dataloader_real = DataLoader(dataset_real, batch_size=self.batch_size, shuffle=False)

        # Initialize the metrics
        fid = FrechetInceptionDistance(normalize=True).to(self.device)
        inception = InceptionScore().to(self.device)
    
        # Update the metrics with the real images
        for batch_real in dataloader_real:
            batch_real = batch_real[0].to(self.device)
            
            # Make sure the input has 3 channels
            if batch_real.shape[1] == 1:
                batch_real = batch_real.repeat(1, 3, 1, 1)
        
            fid.update(batch_real, real=True)
            
        # Update the metrics with the generated images
        for batch_fake in dataloader_fake:
            batch_fake = batch_fake[0].to(self.device) # For now the generator does not provide labels
            
            # Make sure the input has 3 channels
            if batch_fake.shape[1] == 1:
                batch_fake = batch_fake.repeat(1, 3, 1, 1)
                
            batch_fake = (batch_fake * 255).clamp(0, 255).to(torch.uint8)
            
            fid.update(batch_fake, real=False)
            inception.update(batch_fake)
            
        # Compute metrics
        fid_score = fid.compute().item()
        is_score, is_std = inception.compute()
        
        print(f"\n[Validation] FID: {fid_score:.4f}, IS: {is_score:.4f} ± {is_std:.4f}")
        
    def _train_epoch(self, dataloader, n_epoch):
        # Save the training metrics.
        metrics_dict = {"loss_d":[], "loss_g":[]}

        # Initialise the progress bar
        num_batches = len(dataloader)
        num_generator_batches = num_batches // self.n_real_batches + (0 if num_batches % self.n_real_batches else 1)
        progress_bar = tqdm(total=num_generator_batches, desc=f"Epoch {n_epoch}", leave=True)

        # Create an iterator for real batches.
        iter_real = iter(dataloader)

        # Train as long as there are real batches in the dataloader
        done = False
        while not done:
            # List of the steps_d losses.
            losses_d = []
            
            # 'steps_d' batches of training the discriminator
            for _ in range(self.steps_d):
                # Sample real and fake batches for the discriminator.
                can_train, samples_real, labels_real, samples_fake, labels_fake = self._sample_for_discriminator(iter_real)
                
                # Train the discriminator.
                if not can_train:
                    done = True
                else:
                    loss_d = self._train_discriminator_epoch(samples_real, labels_real, samples_fake, labels_fake)
                
                # Save the losses
                losses_d.append(loss_d)
            
            # Sample real and fake batches for the generator. 
            can_train, samples_real, labels_real, samples_fake, labels_fake = self._sample_for_generator(iter_real, self.batch_size)
            
            # Train the generator if possible (If there are batches of real data left)
            if not can_train:
                done = True
            else:
                loss_g = self._train_generator_epoch(samples_real, labels_real, samples_fake, labels_fake)            

            # Update the list of losses for this epoch.
            metrics_dict['loss_d'].append(losses_d)
            metrics_dict['loss_g'].append(loss_g)

            # Update the progress bar.
            progress_bar.update(1)
            progress_bar.set_postfix(loss_d=losses_d[0], loss_g=loss_g)
            
        progress_bar.close()
        
        return metrics_dict
    
    def train(self, dataset, dataset_validation, epochs, validation_freq=10):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        batch_metrics_dict = {'loss_d':[], 'loss_g':[]}
        self.G.train()
        self.D.train()

        for epoch in range(epochs):
            # Train this epoch and collect the metrics.
            this_batch_metrics = self._train_epoch(dataloader, epoch)
            for key in this_batch_metrics.keys():
                batch_metrics_dict[key] += this_batch_metrics[key]

            if epoch % validation_freq == 0: 
                self.validate(dataset_validation)

        return batch_metrics_dict
