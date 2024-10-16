import os
import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
from netCDF4 import Dataset as NetCDFDataset
import json

class LoadTurbTowers():
    def __init__(self):
        a = 0
    def get_dataloaders(self, data_path, n_test = 100):
        # inputs are all the continuous data
        # outputs are the continuous data at the locations furthest from the bubble (10 through 20)
        # parameters come from the input file, visc, diff, amp, bubble shape
        inputs_ = torch.load(f'{data_path}/OOD_continuous.pt')
        parameters_ = torch.load(f'{data_path}/OOD_discrete.pt')

        # get the channels from the several measurement locations
        outputs = torch.zeros(n_test, 2, 10, 181)
        indices = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        # indices = torch.tensor([6])
        outputs[:,0,] = torch.index_select(inputs_[:,0,:], dim=1, index=indices)
        outputs[:,1,] = torch.index_select(inputs_[:,1,:], dim=1, index=indices)
        
        # normalize to [0,1]
        outputs = outputs.view(outputs.shape[0], 20, inputs_.shape[-1])
        
        # Must set the Min/Max from the training data
        mins = torch.tensor([-68.2002, -66.9971, -61.0565, -54.0903, -22.1425, -70.5535, -68.7269,
        -59.6441, -56.0611, -29.9250,  -7.7257, -10.5533, -15.1721, -25.6943,
        -28.0976,  -7.2742, -10.2942, -15.4026, -24.5182, -36.9733])
        
        maxs = torch.tensor([ 3.9365,  3.4960,  5.0944,  4.7840, 29.2092,  6.9290,  5.3521,  6.2691,
         6.2813, 29.7149,  9.3791, 13.0484, 19.6782, 25.9499, 38.7219, 10.9256,
        13.1741, 16.3517, 24.0288, 38.9885])
        
        # Reshape min and max tensors to have dimensions compatible with tensor
        mins = mins.unsqueeze(0).unsqueeze(-1)  # Add dimensions to match tensor shape
        maxs = maxs.unsqueeze(0).unsqueeze(-1)  # Add dimensions to match tensor shape
        # Normalize the tensor
        outputs = (outputs - mins) / (maxs - mins)
        
        p_min = torch.tensor([0.0061, 0.0291, 5.0059, 2.0015, 2.5002, 1.0001])
        p_max = torch.tensor([74.9874, 74.9838, 24.9996,  7.9997,  3.4998,  2.4999])
        for param in range(6):
            parameters_[:,param] = (parameters_[:,param] - p_min[param]) / (p_max[param] - p_min[param])
        
        test_in = outputs[:n_test]
        
        test_param = parameters_[:n_test]
        
        test_dataset = TensorDataset(test_param, test_in, test_in)
        
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        return test_loader, mins, maxs, p_min, p_max, outputs.shape[-1]

    