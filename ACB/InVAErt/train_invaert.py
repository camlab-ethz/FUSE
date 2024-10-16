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
    
    n_train = config['parameters']['n_train']['value']
    n_val = config['parameters']['n_val']['value']
    n_test = config['parameters']['n_test']['value']
    batch_size = config['parameters']['batch_size']['value']
    data_path = config['parameters']['data_path']
    
    loader = LoadTurbTowers()
    val_in, test_in, train_in, val_param, test_param, train_param, input_mins, input_maxs, p_min, p_max, n_points = loader.get_datasets(n_train, n_val, n_test, batch_size, data_path)
    
    # Define the model
    n_parameters = p_min.shape[0]    
    channels_in = input_mins.shape[1]
        
    if torch.cuda.is_available():
        device='cuda:0'
    else:
        device='cpu'
        
    # Define hyperparameters, initiate trainer and define models 
    #-------------------------------------------------------------------------------#

    # Define hyperparameters, see, e.g. appendix of the paper
    # Note: grid search for more optimal parameters TBD
    para_dim    =   6 #n_parameters            # dim(V). In this case, we do not have auxillary data
    # para_dim    =   20*181 #n_parameters       # testing out parameter<>function data swap
    
    nB          =   [64,64,64]             # mini-batch size of N_e, N_f, N_v+N_d
    learning_rate = [1e-3, 1e-3, 1e-3]      # initial learning rate for N_e, N_f, N_v+N_d
    encoder_para  = [256,6,'silu']           # num_of_neuron, num_of_layer, type_of_act fun for emulator N_e
    nf_para       = [128,4,4, False]          # num_of_neuron, num_of_layer_of_each_block, num_of_affine_coupling_blocks, \
                                            # if_using_batch_norm for Real-NVP normalizing flow model N_f
    vae_para      = [256,8,'silu']           # num_of_neuron, num_of_layer, type_of_act fun for VAE N_v
    decoder_para  = [256,8,'silu']           # num_of_neuron, num_of_layer, type_of_act fun for decoder N_d
    penalty       = [1, 300, 10]             # penalty for KL div and decoder reconstruction loss, and the encoder re-constraint loss 
    lr_min        = [1e-5, 1e-4, 5e-5]      # minimal learning rate of N_e, N_f, N_v+N_d
    decay         = [0.997, 0.993, 0.997]   # learning rate decay rate of N_e, N_f, N_v+N_d
    weight_decay  = [0,0,0]                 # L2 regularization of N_e, N_f, N_v+N_d    
    latent_plus    = 20*181 - 6 +1                     # how many additional dimensionality needed
    # latent_plus = 1
    #--------------------------------------------------------------------------------#

    # Reshape train_in, test_in to remove the channels, the encoder only accepts a vector, not channels of vectors
    train_in = train_in.reshape(train_in.shape[0], train_in.shape[1]*train_in.shape[2])
    val_in = val_in.reshape(val_in.shape[0], val_in.shape[1]*val_in.shape[2])
    
    X = torch.cat((train_param, val_param), 0)    
    Y = torch.cat((train_in, val_in), 0)

    
    # Y = torch.cat((train_param, val_param), 0)    # testing out parameter<>function data swap
    # X = torch.cat((train_in, val_in), 0)     # testing out parameter<>function data swap
    
    train_in = train_in.to(device)
    train_param = train_param.to(device)
    val_in = val_in.to(device)
    val_param = val_param.to(device)
    
    # Initiate trainer class
    Trainer = Training(X, Y, para_dim, train_tensor = train_param, train_truth_tensor = train_in,\
                                         test_tensor = val_param, test_truth_tensor = val_in, latent_plus = latent_plus)
    
    # Trainer = Training(X, Y, para_dim, train_tensor = train_in, train_truth_tensor = train_param,\
    #                                      test_tensor = val_in, test_truth_tensor = val_param, latent_plus = latent_plus) # testing out parameter<>function data swap

    # Define models, i.e. N_e, N_f, N_v+N_d
    Encoder_model, NF_model, Decoder_model = Trainer.Define_Models(device, encoder_para, nf_para, vae_para, decoder_para)
    
    folder_name = './'
    # Training and testing step of the encoder, i.e. the emulator N_e
    Trainer.Encoder_train_test(folder_name, Encoder_model, lr_min[0], learning_rate[0], decay[0], nB[0])

    # Trainer.NF_train_test(folder_name, NF_model, lr_min[1], learning_rate[1], decay[1], nB[1])

    Trainer.Decoder_train_test(folder_name, Decoder_model, lr_min[2], learning_rate[2], decay[2], nB[2], penalty,\
                                               l2_decay = weight_decay[2], EN = Encoder_model, residual = False)
    
    
if __name__ == "__main__":
    main()
    
    
