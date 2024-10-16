import numpy as np
import torch
import pdb
from functools import reduce
import operator

def count_params(model):
    """
    Print the number of parameters
    """
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

def range_norm(data):
    data -= torch.min(data)
    data /= torch.max(data)
    
    return data

def de_range_norm(data, max, min):
    # new_data = torch.clone(data)*(max)
    new_data = torch.clone(data)*(max-min)
    new_data = new_data+min 
    return new_data

def max_norm(data):
    data /= torch.max(torch.max(data,0)[0],0)[0]
    
    return data

def de_max_norm(data, max, min):
    new_data = torch.clone(data)*(max)
    return new_data

def gaussian_norm(data):
    mean = data.mean()
    std = data.std()

    normalized_tensor = (data - mean) / std
    
    return normalized_tensor


def random_channel_mask(batch_size, input_channels = 52):
    # generate a mask, each input gets masked to a different extent
    keep_probability = torch.rand([batch_size,1]).expand(-1,input_channels)
    mask = (torch.rand(batch_size, input_channels) < (keep_probability)).float()
    
    return mask
def bernoulli_channel_mask(batch_size, input_channels = 52):
    # generate a mask, each input gets masked to a different extent
    
    keep_probability = torch.ones([batch_size,1]).expand(-1,input_channels) / 2
    mask = torch.bernoulli(keep_probability)
    
    return mask

def parameter_error(samples, targets, error_allowance = 0.05):
    lower_bound = targets - error_allowance
    upper_bound = targets + error_allowance

    # calculate the total number of samples within the error allowance of the target
    within_range = (samples >= lower_bound) & (samples <= upper_bound)
    # sum them and divide by the total number of samples
    total_in = torch.sum(within_range, 0)
    percent_in = total_in / samples.shape[0]

    return torch.mean(percent_in) * 100



def crps(y_pred, y_true, sample_weight=None):
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    
    num_samples = y_pred.shape[0]
    absolute_error = np.mean(np.abs(y_pred - y_true), axis=0)

    if num_samples == 1:
        return np.average(absolute_error, weights=sample_weight)

    y_pred = np.sort(y_pred, axis=0)
    b0 = y_pred.mean(axis=0)
    b1_values = y_pred * np.arange(num_samples).reshape((num_samples, 1))
    b1 = b1_values.mean(axis=0) / num_samples

    per_obs_crps = absolute_error + b0 - 2 * b1
    return np.average(per_obs_crps, weights=sample_weight)
    
def crps_torch(y_pred, y_true):
    
    num_samples = y_pred.size(0)
    absolute_error = torch.mean(torch.abs(y_pred - y_true), dim=0)

    if num_samples == 1:
        return torch.mean(absolute_error, weights=sample_weight)

    y_pred, _ = torch.sort(y_pred, dim=0)
    b0 = torch.mean(y_pred, dim=0)
    b1_values = y_pred * torch.arange(num_samples).reshape((num_samples, 1)).to(y_pred.device)
    b1 = torch.mean(b1_values, dim=0) / num_samples

    per_obs_crps = absolute_error + b0 - 2 * b1
    return torch.mean(per_obs_crps)

def crps_per_parameter(y_pred, y_true, sample_weight=None):
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    
    num_samples = y_pred.shape[0]
    absolute_error = np.mean(np.abs(y_pred - y_true), axis=0)

    if num_samples == 1:
        return np.average(absolute_error, weights=sample_weight)

    y_pred = np.sort(y_pred, axis=0)
    b0 = y_pred.mean(axis=0)
    b1_values = y_pred * np.arange(num_samples).reshape((num_samples, 1))
    b1 = b1_values.mean(axis=0) / num_samples

    per_obs_crps = absolute_error + b0 - 2 * b1
    return per_obs_crps

def h1_loss(prediction, target):
    p_grad = torch.gradient(prediction, dim=-1, edge_order=2)[0]
    t_grad = torch.gradient(target, dim=-1, edge_order=2)[0]
    
    difference = torch.abs(p_grad-t_grad)
    loss = torch.mean(difference)
    return loss
    

def relative_error(predicted, target, p=1):
    """
    Compute the relative Lp error between predicted and target tensors.

    Args:
        predicted (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Target tensor.

    Returns:
        torch.Tensor: Relative Lp error.
    """
    numerator = torch.norm(predicted - target, p=p, dim=1)
    denominator = torch.norm(target, p=p, dim=1)
    error = (numerator / denominator)*100
    return error

def align_waveforms(y, argmin_indices):
    # Iterate over each row
    for i in range(y.size(1)):
        # Check if argmin index is non-zero
        if argmin_indices[0, i] != 0:
            # Calculate the shift amount
            shift_amount = -argmin_indices[0, i].item()

            # Roll the row along the last dimension using the calculated shift amount
            shifted_row = torch.roll(y[0, i], shifts=shift_amount, dims=-1)

            # Update the original tensor with the shifted row
            y[0, i] = shifted_row

    return y
