import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from timeit import default_timer
import pdb
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import garom

import sys
sys.path.append('../')
from utilities import *
from dataset import LoadFullBody


seed = 0
torch.manual_seed(seed)

l1_loss = torch.nn.L1Loss(reduction='none')
    

def eval_test_cases(model, test_loader, device, p_max, p_min, param_max, param_min, point_reduction, n_test):
    '''
    This function will generate several plots for each test case. These are:
    1. average C>C predictions
    2. worst case C>C predictions
    3. average C>D predictions for interesting parameters
    4. worst case C>D predictions ''
    5. average C>D predictions for all parameters
    6. worst case C>D predictions ''
    7. Violin plots of the error for C>D each test case
    8. Error correlations between CRPS and L1 error
    9. Fingerprints for several parameters: Age, PP, LVET, DBP_a, SBP_a, MAP_a 
    '''
    
    # channels: ['AbdAorta' 'AntTibial' 'AorticRoot' 'Brachial' 'Carotid' 'CommonIliac'
             # 'Digital' 'Femoral' 'IliacBif' 'Radial' 'SupMidCerebral' 'SupTemporal'
             # 'ThorAorta']
    evaluate_true_case = True


    crps_scores = np.zeros((n_test,))
    crps_per_case = np.zeros((n_test,3))
    l1_errors = torch.zeros((n_test,13))
    l2_errors = torch.zeros((n_test,13))
    l1_errors_per_case = torch.zeros((n_test,4))
    
    n_samples = 100
    y_preds = torch.zeros((n_samples, 13, 487))
    
    if evaluate_true_case:
        # EVALUATE THE ERRORS OVER THE WAVEFORMS GIVEN THE TRUE PARAMETERS
        with torch.no_grad(): 
            model.eval()
            iter = 0
            for theta, x, y in test_loader:
                theta = theta.to(device)
                y_pred = model(theta)
                
                y_pred = y_pred.cpu()
                

                    
                l1_errors[iter] = relative_error(y_pred[0], y[0], p=1)
                l2_errors[iter] = relative_error(y_pred[0], y[0], p=2)
                iter+=1
                
            # find the location of the median and maximum
            l1_error_means = torch.mean(l1_errors, 1)
            l1_errors_per_case[:,0] = l1_error_means
            l2_error_means = torch.mean(l2_errors, 1)
            error_median = torch.median(l1_error_means)
            error_median_index = torch.abs(l1_error_means-error_median).argmin()
            error_max_index = l1_error_means.argmax()
            
                
            print(f'Using True Parameters')
            print(f'median L1 loss: {torch.median(l1_error_means).item():.4f} average L1 loss: {torch.mean(l1_error_means).item():.4f} +/- {torch.std(l1_error_means).item():.4f} ')
            print(f'max L1 loss: {torch.max(l1_error_means).item():.4f} ')
            print(f'median L2 loss: {torch.median(l2_error_means).item():.4f} average L2 loss: {torch.mean(l2_error_means).item():.4f} +/- {torch.std(l2_error_means).item():.4f} ')
            print(f'max L2 loss: {torch.max(l2_error_means).item():.4f} ')
            print('\n')
            
        
        return 

        
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
    train_loader, val_loader, test_loader, wv_shape, pm_shape, tg_shape, p_max, p_min, param_max, param_min = loader.get_dataloaders(n_train, n_val, n_test, batch_size, params = parameter_subset, norm=norm)
    
    point_reduction = 1
    n_points = int(np.ceil(wv_shape[1]/point_reduction))
    if torch.cuda.is_available():
        device='cuda:0'
    else:
        device='cpu'
    
    # Load the model
    model = torch.load('../../_Models/PWP/GAROM_FullBody.pt')
    
    print(count_params(model))
        
    eval_test_cases(model, test_loader, device, p_max, p_min, param_max, param_min, point_reduction, n_test)

    
    
if __name__ == "__main__":
    main()
    

