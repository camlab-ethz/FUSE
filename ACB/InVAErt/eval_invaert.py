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

from utilities import *
from dataset import LoadTurbTowers


seed = 0
torch.manual_seed(seed)            

def main():
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
        
    # Parameters regarding the datasets
    n_train = config['parameters']['n_train']['value']
    n_test = config['parameters']['n_test']['value']
    n_val = config['parameters']['n_val']['value']
    batch_size = config['parameters']['batch_size']['value']
    data_path = config['parameters']['data_path']
    
    # Load the data from the configs
    loader = LoadTurbTowers()
    train_loader, val_loader, test_loader, input_mins, input_maxs, p_min, p_max, n_points = loader.get_dataloaders(n_train, n_val, n_test, batch_size, data_path)
    
    
    
    # Define the model
    n_parameters = p_min.shape[0]    
    channels_in = input_mins.shape[1]
        
    if torch.cuda.is_available():
        device='cuda:0'
    else:
        device='cpu'
        
#     # Define hyperparameters, initiate trainer and define models 
#     #-------------------------------------------------------------------------------#

#     # Define hyperparameters, see, e.g. appendix of the paper
#     # Note: grid search for more optimal parameters TBD
#     para_dim    =   6 #n_parameters            # dim(V). In this case, we do not have auxillary data
#     nB          =   [64,64,64]             # mini-batch size of N_e, N_f, N_v+N_d
#     learning_rate = [1e-3, 1e-3, 1e-3]      # initial learning rate for N_e, N_f, N_v+N_d
#     encoder_para  = [256,6,'silu']           # num_of_neuron, num_of_layer, type_of_act fun for emulator N_e
#     nf_para       = [128,4,4, False]          # num_of_neuron, num_of_layer_of_each_block, num_of_affine_coupling_blocks, \
#                                             # if_using_batch_norm for Real-NVP normalizing flow model N_f
#     vae_para      = [256,8,'silu']           # num_of_neuron, num_of_layer, type_of_act fun for VAE N_v
#     decoder_para  = [256,8,'silu']           # num_of_neuron, num_of_layer, type_of_act fun for decoder N_d
#     penalty       = [1, 300, 10]             # penalty for KL div and decoder reconstruction loss, and the encoder re-constraint loss 
#     lr_min        = [1e-5, 1e-4, 5e-5]      # minimal learning rate of N_e, N_f, N_v+N_d
#     decay         = [0.997, 0.993, 0.997]   # learning rate decay rate of N_e, N_f, N_v+N_d
#     weight_decay  = [0,0,0]                 # L2 regularization of N_e, N_f, N_v+N_d    
#     latent_plus    = 20*181 - 6 +1                     # how many additional dimensionality needed
#     #--------------------------------------------------------------------------------#
    
    en_model = torch.load('../../_Models/ACB/InVAErt_TurbTowers_Encoder_model.pt').to(device) # takes the parameters as input, returns velocities
    # nf_model = torch.load('NF_model.pt').to(device)      # takes the velocities as input, returns velocities
    de_model = torch.load('../../_Models/ACB/InVAErt_TurbTowers_Decoder_model.pt').to(device) # takes the velocities as input, returns parameters

    
    l1_errors = torch.zeros(1000)     # errors from the true parameters
    l2_errors = torch.zeros(1000)     # errors from the true parameters
    
    crps = np.zeros((1000,6)) # errors from the average of the normalizing flow predictions
    
    l1_errors_from_param = torch.zeros(1000)     # errors from the true parameters
    l2_errors_from_param = torch.zeros(1000)     # errors from the true parameters
    
    n_samples = 1000
    iter = 0
    with torch.no_grad():
        en_model.eval()
        # nf_model.eval()
        de_model.eval()
            
        # Evaluate the errors over the test set
        # for test_param, test_in, test_out in test_loader:
        for test_param, test_in, test_out in test_loader:
            test_param = test_param.to(device)
            test_in = test_in.reshape(test_in.shape[0], test_in.shape[1]*test_in.shape[2])
            test_in = test_in.to(device)
            test_out = test_out.to(device)

            # predict the function measurements from the true parameters
            y_pred = en_model(test_param)
            y_pred = y_pred.reshape(1, 20, 181)
            l1_errors[iter] = torch.mean(relative_error(y_pred, test_out)).item()
            l2_errors[iter] = torch.mean(relative_error(y_pred, test_out, p=2)).item()
            

            # predict the parameters from the time-series inputs
            w_samples = de_model.VAE_sampling(n_samples).to(device)
            decode_input = test_in.repeat(n_samples,1)
            decode_input = torch.cat((decode_input, w_samples), 1)
            param_pred = de_model.Decoder(decode_input)
            
            crps[iter] = crps_per_parameter(param_pred, test_param)
            
            
            # predict the time-series from the parameters
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
    
    
