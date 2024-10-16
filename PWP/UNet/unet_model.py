import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import torch.nn.functional as F

from lampe.inference import FMPE, FMPELoss
import pdb

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

class UNet(nn.Module):
    def __init__(self, n_parameters, n_points, FMPE_width, FMPE_layers, in_channels, latent_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, latent_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.center = nn.Sequential(
            nn.Conv1d(latent_channels, latent_channels*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(latent_channels*2, latent_channels*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=1, stride=1)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_channels*3, latent_channels*2, kernel_size=2, stride=2),
            nn.Conv1d(latent_channels*2, latent_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(latent_channels, out_channels, kernel_size=1)
        )
        
        # 90 in this case is the number of point divided by two, caused by the convolutions
        self.FMPE_in = FMPE(n_parameters, latent_channels*3*(n_points//2), hidden_features=[FMPE_width] * FMPE_layers)
        self.loss_in = FMPELoss(self.FMPE_in)
        
        # lifting layers from xi to a
        self.fc1 = nn.Linear(n_parameters, latent_channels*3) 
        self.fs1 = nn.Linear(1, (n_points//2)) 
        
    def param_loss(self, theta, x):
        x1 = self.encoder(x)
        x2 = self.center(x1)
        x2 = torch.cat((x1, x2), dim=1)

        x2 = x2.reshape(x.shape[0], -1)
        loss = self.loss_in(theta, x2)
        return loss
    
    def y_prediction(self, theta, n_points=1):
        a = self.fc1(theta)[:,None,:]
        a = self.fs1(a.permute(0,2,1))
        a = F.gelu(a)
        
        x3 = self.decoder(a)
        return x3
        
    def param_prediction(self, x, n_samples = 1000):  
        x1 = self.encoder(x)
        x2 = self.center(x1)
        x2 = torch.cat((x1, x2), dim=1)

        x2 = x2.reshape(x.shape[0], -1)
        # Compute posterior distribution of theta given x
        batch_size = x.shape[0]
        theta = self.FMPE_in.flow(x2).rsample((n_samples,))

        return theta
                                 
    def full_flow(self, x, n_samples = 1000): 
        # predict the posterior distribution over the parameters 
        # and use these to compute the distribution over possible outputs
        x1 = self.encoder(x)
        x2 = self.center(x1)
        x2 = torch.cat((x1, x2), dim=1)

        x2 = x2.reshape(x.shape[0], -1)
        
        # Compute posterior distribution of theta given x
        batch_size = x.shape[0]
        theta = self.FMPE_in.flow(x2).rsample((n_samples,))
        
        
        a = self.fc1(theta)
        a = self.fs1(a.permute(0,2,1))
        a = F.gelu(a)
        
        x3 = self.decoder(a)

        return theta, x3
