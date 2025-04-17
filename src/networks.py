import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        
        assert len(layer_sizes) >= 2, "Must have at least an input and an output layer"
        
        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
        # The final layer, without relu.
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class CNN(nn.Module):
    def __init__(self, in_channels, layer_configs, transpose=False):
        super(CNN, self).__init__()
        layers = []

        in_filters = in_channels
        for out_filters, kernel_size, stride, padding in layer_configs:
            # The convolutional layer and its activation.
            if transpose:
                layer = nn.ConvTranspose2d(
                    in_filters, out_filters, kernel_size, stride=stride, padding=padding
                )
            else:
                layer = nn.Conv2d(
                    in_filters, out_filters, kernel_size, stride=stride, padding=padding
                )
            layers.append(layer)
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            # Modify the input number of filters for the next convolution
            in_filters = out_filters
            
        # Remove the last activation layer
        layers = layers[:-1]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
