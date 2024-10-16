import numpy as np
import torch
import pdb
import matplotlib.pyplot as plt
from utilities import range_norm, gaussian_norm, max_norm
from torch.utils.data import DataLoader, Dataset, TensorDataset

class LoadFullBody():
    def __init__(self, data_path  = "../../_Data/ACB/PW_input_data.npz"):
        self.data = np.load(data_path)
        self.keys = self.data.keys()
        
    def get_data(self, norm='max'):
        
        # Continuous data (13 measurement locations)
        pressure = torch.tensor(self.data["pressures"]).float()
        velocity = torch.tensor(self.data["velocities"]).float()
        PPG = torch.tensor(self.data["PPGs"]).float()
        t_out = torch.tensor(self.data["t_out"]).float()
        
        # Discrete parameters (32)
        parameters = torch.tensor(self.data["parameters"]).float()
        
        
        if norm=='gaus':
            pressure = gaussian_norm(pressure)
            velocity = gaussian_norm(velocity)
            PPG = gaussian_norm(PPG)
            
            for param in range(parameters.shape[1]):
                parameters[:,param] = gaussian_norm(parameters[:,param])
                
        elif norm=='max':
            p_max = torch.max(pressure)
            p_min = torch.min(pressure)
            pressure = max_norm(pressure)
            
            v_max = torch.max(velocity)
            v_min = torch.min(velocity)
            velocity = max_norm(velocity)
            
            
            g_max = torch.max(PPG)
            g_min = torch.min(PPG)
            PPG = max_norm(PPG)
            
            
        param_max = torch.zeros(parameters.shape[1])
        param_min = torch.zeros(parameters.shape[1])
        for param in range(parameters.shape[1]):
            param_max[param] = torch.max(parameters[:,param])
            param_min[param] = torch.min(parameters[:,param])
            parameters[:,param] = range_norm(parameters[:,param])
            
            
        # Concatenate the data along the last dimension
        waveforms = torch.cat((pressure, velocity, PPG, t_out), dim=-1)
        # waveforms = torch.cat((pressure, velocity, PPG), dim=-1)

        return waveforms, p_max, p_min, parameters, param_max, param_min
    
    def get_dataloaders(self, n_train, n_val, n_test, batch_size, params='full', norm='max'):
        # Get the waveforms data using get_data
        waveforms, p_max, p_min, parameters, param_max, param_min = self.get_data(norm=norm)
        
        waveforms = waveforms[:n_train+n_val+n_test]
        parameters = parameters[:n_train+n_val+n_test]
        waveforms = waveforms.permute(0,2,1)
        pressures = waveforms[:,:13,:]

        if params=="subset":
            indices = torch.tensor([0,1,2,3,10,20])
            parameters = torch.index_select(parameters, 1, indices)
        

        # train_data, test_data = torch.utils.data.random_split(TensorDataset(parameters, waveforms, pressures), [n_train, n_test])
        train_data = TensorDataset(parameters[:n_train], waveforms[:n_train], pressures[:n_train])
        validation_data = TensorDataset(parameters[-n_val:], waveforms[-n_val:], pressures[-n_val:])
        test_data = TensorDataset(parameters[n_train:n_train+n_test], waveforms[n_train:n_train+n_test], pressures[n_train:n_train+n_test])

        # Create DataLoader objects for training and testing sets
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(validation_data, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

        return train_loader, val_loader, test_loader, waveforms.shape, parameters.shape, pressures.shape, p_max, p_min, param_max, param_min
    
    def get_datasets(self, n_train, n_val, n_test, batch_size, params='full', norm='max'):
        # Get the waveforms data using get_data
        waveforms, p_max, p_min, parameters, param_max, param_min = self.get_data(norm=norm)
        
        waveforms = waveforms[:n_train+n_val+n_test]
        parameters = parameters[:n_train+n_val+n_test]
        
        waveforms = waveforms.permute(0,2,1)
        pressures = waveforms[:,:13,:]


        return waveforms, parameters, pressures.shape, p_max, p_min, param_max, param_min
    

