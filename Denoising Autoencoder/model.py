import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# HYPERPARAMETERS
batch_size = 64
learning_rate = 1e-3
max_iters = 10000

transform = transforms.Compose([
    transforms.ToTensor(), # Converts the inputs to tensors, scales the values to [0,1]
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Shuffle train data to break artifical correlations in the original order
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# DO NOT shuffle test data to keep consistency
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class DenoisingAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()
        # LAYERS
        self.encoder = nn.Sequential(
            # The first Conv2d layer takes a single image
            # It applies 16 different 3x3 kernels with stride 1
            # We add a padding of 1 for the edges (If we had 5x5 kernels, we should use padding of 2)

            # Dim is (B, 1, 28, 28)
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            # Dim is (B, 32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Dim is (B, 32, 14, 14)
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            # Dim is (B, 16, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Dim is (B, 8, 7, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid() # End with sigmoid since we want to output numbers in [0,1]
        )
            
    def forward(self, x):
        # Input has dim (B, 28, 28)
        # Values already in range of [0,1] due to how we loaded it
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
def visualize_grid(tensor, name):
    grid = make_grid(tensor)

    image = grid.cpu().numpy()

    image = np.transpose(image, (1, 2, 0))

    plt.imshow(image)
    plt.axis('off')
    plt.savefig(f'{name}.png')

    
