import numpy as np
import torch
import pdb
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from utilities import *
    
    
def continuous_errors(pred, true):
    l1_errors = torch.zeros((1000,))
    l2_errors = torch.zeros((1000,))
    
    for sample in range(true.shape[0]):
        l1_errors[sample] = torch.mean(relative_error(pred[sample], true[sample], p=1)).item()
        l2_errors[sample] = torch.mean(relative_error(pred[sample], true[sample], p=2)).item()
        
    pred = pred.reshape(1000, 2, 10, 181)
    true = true.reshape(1000, 2, 10, 181)
    
    
    # plot_true_vs_predicted(true[3], pred[3][None], 'cvano_true.png')
    
    print(f"L1 error: {torch.mean(l1_errors).item()} +/- {torch.std(l1_errors).item()}")
    print(f"L2 error: {torch.mean(l2_errors).item()} +/- {torch.std(l2_errors).item()}")
    
    return


def discrete_errors(pred, true):
    crps_scores = np.zeros((1000,))
    
    for sample in range(true.shape[0]):
        crps_score = crps_per_parameter(pred[sample,:,:], true[sample])
        crps_scores[sample] = np.mean(crps_score)
        
    print(f"discrete CRPS: {np.mean(crps_scores).item()} +/- {np.std(crps_scores).item()}")
    
    return
          
          
          
          
    
    
def main():
    n_train = 0
    n_test = 1000
    n_val = 0
    batch_size = 1
    
    # True Parameters
    print('Using True Parameters')
    file_path = "./"
    # Load the compressed npz file
    data = np.load(file_path+'turb_towers_results_decoding_latent_vectors_of_true_parameters.npz')

    # Extract the arrays from the npz file
    true_pressures = data['true_pressures']
    pressures = data['pressures']
    parameters = data['parameters']
    
    
    true_pressures = torch.from_numpy(true_pressures)
    pressures = torch.from_numpy(pressures)
    true_parameters = torch.from_numpy(parameters)
    
    pressure_preds = pressures.permute(0,3,1,2)
    pressure_preds = pressure_preds[:,:,:,0]
    
    true_pressures = true_pressures.permute(0,2,1)
          
    continuous_errors(pressure_preds, true_pressures)
    
    # True Parameters
    print('Using Predicted Parameters')
    file_path = "./"
        # Load the compressed npz file
    data = np.load(file_path+'turb_towers_results_using_infered_parameters.npz')
    
        
    # Extract the arrays from the npz file
    true_pressures = data['true_pressures']
    pressures = data['pressures']
    parameters = data['parameters']
    
    true_pressures = torch.from_numpy(true_pressures)
    pressures = torch.from_numpy(pressures)
    parameters = torch.from_numpy(parameters)
    
    pressures = pressures.permute(0,2,3,1)
    pressure_preds = torch.mean(pressures,1)
    
    true_pressures = true_pressures.permute(0,2,1)
    
          
    parameters = parameters[:,:,0]
    
    
    discrete_errors(parameters, true_parameters)
    continuous_errors(pressure_preds, true_pressures)
          
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    main()