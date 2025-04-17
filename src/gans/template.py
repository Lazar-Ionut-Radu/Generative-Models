import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

class GeneratorTemplate(nn.Module):
    """
    Base class for the generator network in GAN architectures. May be part of
    either a conditional or unconditional GAN.

    Args:
        data_shape (tuple): The shape of the generated data.
        latent_dim (int): Dimention of the latent space inputs.
    """
    
    def __init__(self, data_shape, latent_dim=100):
        super(GeneratorTemplate, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_shape = data_shape
        self.optim = None
        self.latent_dim = latent_dim

    def _get_random_labels(self, n):
        """
        Generates random labels. Implement this in the subclass if it is part
        of a conditional GAN.
        
        Args:
            n (int): Number of labels to be generated

        Returns:
            (torch.Tensor | None): Either tensor containing the labels or None.
        """
        return None 
    
    def _get_z(self, n):
        """
        Generates random latent space vectors.

        Args:
            n (int): Number of samples to be generated

        Returns:
            torch.Tensor: A tensor containing the latent vectors.
        """
        return torch.randn(n, self.latent_dim, device=self.device)
    
    def forward(self, z, labels=None):
        """
        The forward method of the generator, implement this in your subclass.
        Generates samples of the specified labels. You may omit the labels if
        you may wish to implement an unconditional GAN.
        
        Args:
            z (Tensor): Latent vectors input
            labels (Tensor | None): Labels used to generated samples. Defaults to None

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """
        raise NotImplementedError("You need to implement this in your generator subclass.")

    # [TODO]: If labels is not None, n and len(labels) should match.
    def generate(self, n=1, labels=None):
        """
        Generates a specified number of samples given the labels. If
        unspecified, it will use randomly generated labels.

        Args:
            n (int): Number of samples to be generated. Defaults to 1.
            labels (torch.Tensor): Labels used to generate samples. Defaults to None.

        Returns:
            torch.Tensor: The generated samples.
        """
        z = self._get_z(n)

        if labels is None:
            labels = self._get_random_labels(n)
            
        return self.forward(z, labels), labels
    
    def get_dataloader(self, labels, batch_size=64):
        """
        Returns a pytorch Dataloader of generated samples and their labels
        for evaluation.

        Args:
            labels (torch.Tensor): Labels used to generate samples.
            batch_size (int): Batch size of the dataloader. Defaults to 64.

        Returns:
            Dataloader: Dataloader with generated samples and their labels.
        """
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
    """
    Base class for the discriminator network in GAN architectures. May be part
    of either a conditional or unconditional GAN.

    Args:
        data_shape (tuple): The shape of the generated data.
    """
    
    def __init__(self, data_shape):
        super(DiscriminatorTemplate, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optim = None
        self.data_shape = data_shape

    def forward(self, samples, labels=None):
        """
        Forward method of the network, implement this in your subclass. You may
        omit the labels if you wish to implement an unconditional GAN.

        Args:
            samples (torch.Tensor): Input sample
            labels (torch.Tensor): Label of the sample. Defaults to None.

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """
        raise NotImplementedError("You need to implement this in your generator subclass.")

class GANTemplate(object):
    """
    Template class for GAN that implements training and validation loops.
    Leaves unimplemented methods for the loss functions of the generator and 
    discriminator networks for allowing generality, implement them in your
    subclass.
    Allows for both conditional and unconditional GANs. In actuality, The
    labels(or any additional inputs for that matter) are passed along to the
    generator and discriminator classes, whose duty is to process them as
    needed.

    Args:
        G (GeneratorTemplate): Generator model         
        D (DiscriminatorTemplate): Discriminator model
        batch_size (int): Batch size for training. Defaults to 64.
        steps_d (int): Number of discriminator training iterations per
                       generator iteration. Defaults to 1.
    """
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
        """
        Calculates the discriminator loss. Must be implemented this in the
        subclass, per the specifications of the wanted GAN architecture.

        Args:
            samples_real (torch.Tensor): Real samples.
            labels_real (torch.Tensor | None): Labels of the real data (or None).
            samples_fake (torch.Tensor): Generated samples.
            labels_fake (torch.Tensor | None): Labels of the generated samples (or None).

        Raises:
            NotImplementedError: If not implemented in the subclass.
            
        Returns:
            float: The discriminator loss.
        """
        raise NotImplementedError("You need to implement this in your GAN subclass.")

    def _calculate_generator_loss(self, samples_real, labels_real, samples_fake, labels_fake):
        """
        Calculates the generator loss. May or may not need the generated 
        samples depending on the GAN architecture. Must be implemented in the
        subclass.

        Args:
            samples_real (torch.Tensor): Real samples.
            labels_real (torch.Tensor | None): Labels of the real data (or None).
            samples_fake (torch.Tensor): Generated samples.
            labels_fake (torch.Tensor | None): Labels of the generated samples (or None).

        Raises:
            NotImplementedError: If not implemented in the subclass.
            
        Returns:
            float: The generator loss.
        """
        raise NotImplementedError("You need to implement this in your GAN subclass.")

    def _train_discriminator_epoch(self, samples_real, labels_real, samples_fake, labels_fake):
        """
        Perform one optimization step of the discriminator. Useful to have as a
        standalone method if inserting code at this point, inside the training
        loop is needed (ex: For weight clipping in WGANs)
        
        Args:
            samples_real (torch.Tensor): Real samples.
            labels_real (torch.Tensor | None): Labels of the real data (or None).
            samples_fake (torch.Tensor): Generated samples.
            labels_fake (torch.Tensor | None): Labels of the generated samples (or None).

        Returns:
            float: The discriminator loss.
        """
        
        # Calculate loss & step the discriminator's optimizer
        self.D.optim.zero_grad()
        loss_d = self._calculate_discriminator_loss(samples_real, labels_real, samples_fake, labels_fake)
        loss_d.backward()
        self.D.optim.step()
        
        return loss_d.item()
    
    def _train_generator_epoch(self, samples_real, labels_real, samples_fake, labels_fake):
        """
        Perform one optimization step of the generator. Useful to have as a
        standalone method if inserting code at this point, inside the training
        loop is needed.
        
        Args:
            samples_real (torch.Tensor): Real samples.
            labels_real (torch.Tensor | None): Labels of the real data (or None).
            samples_fake (torch.Tensor): Generated samples.
            labels_fake (torch.Tensor | None): Labels of the generated samples (or None).

        Returns:
            float: The generator loss.
        """
        
        # Calculate loss & step the generator's optimizer
        self.G.optim.zero_grad()
        loss_g = self._calculate_generator_loss(samples_real, labels_real, samples_fake, labels_fake)
        loss_g.backward()
        self.G.optim.step()
        
        return loss_g.item()

    def _sample_for_generator(self, iter_real, batch_size):
        """
        Sample data needed for training the generator. Useful as a standalone
        method because different GAN architectures may need different data for
        training the generator. Some may not need real data at all, the
        corresponding outputs being None in that case.

        Args:
            iter_real (iterator): Iterator through the dataloader of. May be 
                                  useful depending on the GAN architecture. 
            batch_size (int): Size needed for this generated batch.

        Returns:
            (Tuple[bool, Torch.Tensor, Torch.tensor, torch.Tensor, torch.Tensor]):
            Success flag, real samples and labels, generated samples and labels.
        """
        
        # Sample a batch of fake data, with random labels (or None)
        samples_fake, labels_fake = self.G.generate(n=batch_size)
        
        return True, None, None, samples_fake, labels_fake
    
    def _sample_for_discriminator(self, iter_real):
        """
        Sample data needed for training the discriminator. First parameter is
        False if no real batches are left.
        
        Args:
            iter_real (iterator): Iterator through the dataloader. 
            batch_size (int): Size needed for this generated batch.

        Returns:
            (Tuple[bool, Torch.Tensor, Torch.tensor, torch.Tensor, torch.Tensor]):
            Success flag, real samples and labels, generated samples and labels.
        """
        
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
    
    # [TODO]: Return metrics, kinda hardcored as of now.
    def validate(self, dataset_real):
        """
        Compute FID and Inception Score using generated and real data.

        Args:
            dataset_real (Dataset): Dataset with real samples.
        """
        
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
        
    # [TODO]: The metrics dict only outputs d_loss and g_loss, however, in
    # different architecture the d_loss may have a real and fake part as well
    # as regulization terms. May be nice to make each GAN subclass have its own
    # metrics dictionaries it modifies inside the _calculate_*_loss() methods. 
    def _train_epoch(self, dataloader, n_epoch):
        """
        Run one full training epoch.

        Args:
            dataloader (DataLoader): Training data loader.
            n_epoch (int): Epoch number (used for display).

        Returns:
            dict: Dictionary with lists of losses for discriminator and generator.
        """
        
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
        """
        Train the GAN for a specified number of epochs.

        Args:
            dataset (Dataset): Training dataset.
            dataset_validation (Dataset): Validation dataset.
            epochs (int): Number of epochs to train.
            validation_freq (int): Frequency of validation (in epochs).

        Returns:
            dict: Dictionary with training loss metrics.
        """
        
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
