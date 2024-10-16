import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from timeit import default_timer
import pdb
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from garom import GAROM
import yaml

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
    
    # Model setup
    nz = config['parameters']['noise_dim']['value']
    latent_dim = config['parameters']['encoding_dim']['value']
    plotting = config['parameters']['plot']['value']
    printing = config['parameters']['print']['value']
    print_every = config['parameters']['print_every']['value']
    epochs = config['parameters']['epochs']['value']
    gamma = config['parameters']['factor']['value']
    lambda_k = config['parameters']['lambda_k']['value']
    lr = config['parameters']['learning_rate']['value']
    wd = config['parameters']['weight_decay']['value']
    regular = config['parameters']['regular']['value']
    input_dim = 181
    param_dim = 6
    print(
        f"simulation: seed {seed}, encoding dim {latent_dim}, noise dim {nz}, gamma {gamma}, regularizer {regular}")
    # optimizer and scheduler
    optimizers = {'generator': torch.optim.Adam,
                  'discriminator': torch.optim.Adam}
    optimizers_kwds = {'generator': {"lr": lr, "weight_decay": wd},
                       'discriminator': {"lr": lr, "weight_decay": wd}}
    schedulers = None
    schedulers_kwds = None
    
    garom = GAROM(input_dimension=input_dim,
                  hidden_dimension=latent_dim,
                  parameters_dimension=param_dim,
                  noise_dimension=nz,
                  regularizer=regular,
                  optimizers=optimizers,
                  optimizer_kwargs=optimizers_kwds,
                  schedulers=schedulers,
                  scheduler_kwargs=schedulers_kwds)
        
    if torch.cuda.is_available():
        device='cuda:0'
    else:
        device='cpu'
    
    print(count_params(garom._generator)+count_params(garom._discriminator))
    garom.to(device)  
    
    start = default_timer()
    garom.train(train_loader, val_loader, epochs=epochs,
                gamma=gamma, lambda_k=lambda_k,
                every=print_every,
                save_csv=f'gaussian_{latent_dim}_{seed}')

    print(f"total time {default_timer()-start}")
    garom.save(f'generator_gaussian_{latent_dim}', f'discriminator_gaussian_{latent_dim}')

    
    
if __name__ == "__main__":
    main()
    
    
