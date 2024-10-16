import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from timeit import default_timer
import pdb
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import wandb

from DNN_tools import *
from NF_tools import *
from Training_tools import * 

import sys
sys.path.append('../')
from ood_dataset import LoadTurbTowers


sys.path.append('../../')
from utilities import *


seed = 0
torch.manual_seed(seed)            

def main():
    pad = 30
    n_test = 100    
    path  = "../../_Data/ACB/"
    # Load the data from the configs
    loader = LoadTurbTowers()

    plot_sample = False
    
    test_loader, input_mins, input_maxs, p_min, p_max, n_points = loader.get_dataloaders(path)    
    
    # Define the model
    n_parameters = p_min.shape[0]    
    channels_in = input_mins.shape[1]
        
    if torch.cuda.is_available():
        device='cuda:0'
    else:
        device='cpu'
    
    en_model = torch.load('../../../_Models/ACB/InVAErt_TurbTowers_Encoder_model.pt').to(device) # takes the parameters as input, returns velocities
    de_model = torch.load('../../../_Models/ACB/InVAErt_TurbTowers_Decoder_model.pt').to(device) # takes the velocities as input, returns parameters
    
    l1_errors = torch.zeros(n_test)     # errors from the true parameters
    l2_errors = torch.zeros(n_test)     # errors from the true parameters
    
    crps = np.zeros((n_test,6)) # errors from the average of the normalizing flow predictions
    
    l1_errors_from_param = torch.zeros(n_test)     # errors from the true parameters
    l2_errors_from_param = torch.zeros(n_test)     # errors from the true parameters
    
    n_samples = 1000
    iter = 0
    with torch.no_grad():
        en_model.eval()
        de_model.eval()
        
        # Evaluate the errors over the training set
        for test_param, test_in, test_out in test_loader:
            test_param = test_param.to(device)
            test_in = test_in.reshape(test_in.shape[0], test_in.shape[1]*test_in.shape[2])
            test_in = test_in.to(device)
            test_out = test_out.to(device)
            
            y_pred = en_model(test_param)
            y_pred = y_pred.reshape(1, 20, 181)
            l1_errors[iter] = torch.mean(relative_error(y_pred, test_out)).item()
            l2_errors[iter] = torch.mean(relative_error(y_pred, test_out, p=2)).item()
            
            if iter ==0 and plot_sample :
                plt.figure()
                plt.plot(np.arange(181), test_out[0,0,:].cpu().numpy(), label="True velocity")
                plt.plot(np.arange(181), y_pred[0,0,:].cpu().numpy(), label="Predicted velocity")
                plt.savefig("u_velocity_sample.png")
                
                plt.figure()
                plt.plot(np.arange(181), test_out[0,10,:].cpu().numpy(), label="True velocity")
                plt.plot(np.arange(181), y_pred[0,10,:].cpu().numpy(), label="Predicted velocity")
                plt.savefig("w_velocity_sample.png")
            
            
            w_samples = de_model.VAE_sampling(n_samples).to(device)
            decode_input = test_in.repeat(n_samples,1)
            decode_input = torch.cat((decode_input, w_samples), 1)
            param_pred = de_model.Decoder(decode_input)
            
            crps[iter] = crps_per_parameter(param_pred, test_param)
            
            
            y_pred_distribution = en_model(param_pred)
            y_pred_mean = torch.mean(y_pred_distribution, 0)
            y_pred_std = torch.std(y_pred_distribution, 0)
            y_upper = torch.max(y_pred_distribution, 0)[0]
            y_lower = torch.min(y_pred_distribution, 0)[0]
            
            y_pred_mean = y_pred_mean.reshape(1, 20, 181)
            y_upper = y_upper.reshape(1, 20, 181)
            y_lower = y_lower.reshape(1, 20, 181)
            
            l1_errors_from_param[iter] = torch.mean(relative_error(y_pred_mean, test_out)).item()
            l2_errors_from_param[iter] = torch.mean(relative_error(y_pred_mean, test_out, p=2)).item()
            
            if iter ==0 and plot_sample :
                plt.figure()
                plt.plot(np.arange(181), test_out[0,0,:].cpu().numpy(), label="True velocity")
                plt.plot(np.arange(181), y_pred_mean[0,0,:].cpu().numpy(), 'r', label="Predicted velocity")
                plt.fill_between(np.arange(181), y_upper[0,0,:].cpu().numpy(), y_lower[0,0,:].cpu().numpy(), color='red', alpha=0.3, label="Prediction Range")
                plt.savefig("u_velocity_range.png")
                
                plt.figure()
                plt.plot(np.arange(181), test_out[0,10,:].cpu().numpy(), label="True velocity")
                plt.plot(np.arange(181), y_pred_mean[0,10,:].cpu().numpy(), 'r', label="Predicted velocity")
                plt.fill_between(np.arange(181), y_upper[0,10,:].cpu().numpy(), y_lower[0,10,:].cpu().numpy(), color='red', alpha=0.3, label="Prediction Range")
                plt.savefig("w_velocity_range.png")
            
            iter += 1
            
        print(f"l1 error: {torch.mean(l1_errors).item()} +/- {torch.std(l1_errors).item()}")
        print(f"l2 error: {torch.mean(l2_errors).item()} +/- {torch.std(l2_errors).item()}")
        print(f"crps overall: {np.mean(crps)} +/- {np.std(crps)}")
        print(f"crps overall: {np.mean(crps,0)}")
        print(f"crps overall: {np.std(crps,0)}")
        
        print(f"l1 error from param preds: {torch.mean(l1_errors_from_param).item()} +/- {torch.std(l1_errors_from_param).item()}")
        print(f"l2 error from param preds: {torch.mean(l2_errors_from_param).item()} +/- {torch.std(l2_errors_from_param).item()}")
            
    
if __name__ == "__main__":
    main()
    
    
