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
from dataset import LoadFullBody


seed = 0
torch.manual_seed(seed)            

def main():
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
        
    # Get the data using a given normalization, select whether to use 
    # the parameters from Alastruey paper
    norm = config['parameters']['data_norm']
    parameter_subset = config['parameters']['params']
    n_train = config['parameters']['n_train']['value']
    n_val = config['parameters']['n_val']['value']
    n_test = config['parameters']['n_test']['value']
    batch_size = config['parameters']['batch_size']['value']
    
    # Load the data from the configs
    loader = LoadFullBody()
    waveforms, parameters, input_mins, input_maxs, p_min, p_max, n_points = loader.get_datasets(n_train, n_val, n_test, batch_size)
    
    # Define the model
    n_parameters = parameters.shape[1]    
    channels_in = waveforms.shape[1]
    n_points = waveforms.shape[2]
        
    if torch.cuda.is_available():
        device='cuda:0'
    else:
        device='cpu'
        
    # Define hyperparameters, initiate trainer and define models 
    #-------------------------------------------------------------------------------#

    # Define hyperparameters, see, e.g. appendix of the paper
    # Note: grid search for more optimal parameters TBD
    para_dim    =   n_parameters #n_parameters            # dim(V). In this case, we do not have auxillary data
    nB          =   [64,64,64]             # mini-batch size of N_e, N_f, N_v+N_d
    learning_rate = [1e-3, 1e-3, 1e-3]      # initial learning rate for N_e, N_f, N_v+N_d
    encoder_para  = [256,6,'silu']           # num_of_neuron, num_of_layer, type_of_act fun for emulator N_e
    nf_para       = [128,4,4, False]          # num_of_neuron, num_of_layer_of_each_block, num_of_affine_coupling_blocks, \
                                            # if_using_batch_norm for Real-NVP normalizing flow model N_f
    vae_para      = [16,4,'silu']           # num_of_neuron, num_of_layer, type_of_act fun for VAE N_v
    decoder_para  = [64,4,'silu']           # num_of_neuron, num_of_layer, type_of_act fun for decoder N_d
    penalty       = [1, 300, 5]             # penalty for KL div and decoder reconstruction loss, and the encoder re-constraint loss 
    lr_min        = [1e-5, 1e-4, 5e-5]      # minimal learning rate of N_e, N_f, N_v+N_d
    decay         = [0.997, 0.993, 0.997]   # learning rate decay rate of N_e, N_f, N_v+N_d
    weight_decay  = [.0001,.0001,.0001]                 # L2 regularization of N_e, N_f, N_v+N_d    
    latent_plus    = n_points*channels_in - n_parameters +100                     # how many additional dimensionality needed
    #--------------------------------------------------------------------------------#

    # Reshape train_in, test_in to remove the channels, the encoder only accepts a vector, not channels of vectors
    train_in = waveforms[:n_train]
    val_in = waveforms[n_train:n_train+n_val]
    
    train_param = parameters[:n_train]
    val_param = parameters[n_train:n_train+n_val]
    
    train_in = train_in.reshape(train_in.shape[0], n_points*channels_in)
    val_in = val_in.reshape(val_in.shape[0], n_points*channels_in)
    
    X = torch.cat((train_param, val_param), 0)    
    Y = torch.cat((train_in, val_in), 0)
    
    train_in = train_in.to(device)
    train_param = train_param.to(device)
    val_in = val_in.to(device)
    val_param = val_param.to(device)
    
    # Initiate trainer class
    Trainer = Training(X, Y, para_dim, train_tensor = train_param, train_truth_tensor = train_in, test_tensor = val_param, test_truth_tensor = val_in, latent_plus = latent_plus)

    # Define models, i.e. N_e, N_f, N_v+N_d
    Encoder_model, NF_model, Decoder_model = Trainer.Define_Models(device, encoder_para, nf_para, vae_para, decoder_para)
    
    folder_name = './'
    
    # Training and testing step of the encoder, i.e. the emulator N_e
    Trainer.Encoder_train_test(folder_name, Encoder_model, lr_min[0], learning_rate[0], decay[0], nB[0])

    # Must use early stopping before KL collapses    
    Trainer.Decoder_train_test(folder_name, Decoder_model, lr_min[2], learning_rate[2], decay[2], nB[2], penalty,\
                                               l2_decay = weight_decay[2], EN = Encoder_model, residual = False)
    
    
if __name__ == "__main__":
    main()
    
    
