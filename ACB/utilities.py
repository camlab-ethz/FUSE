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

def de_range_norm(data, min, max):
    new_data = torch.clone(data)*(max-min)
    new_data = new_data+min
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

def crps_per_continuous(y_pred, y_true, sample_weight=None):
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    
    y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1]*y_pred.shape[2])
    y_true = y_true.reshape(y_true.shape[-1]*y_true.shape[-2])

    num_samples = y_pred.shape[0]
    absolute_error = np.mean(np.abs(y_pred - y_true), axis=0)

    if num_samples == 1:
        return np.average(absolute_error, weights=sample_weight)

    y_pred = np.sort(y_pred, axis=0)
    b0 = y_pred.mean(axis=0)
    b1_values = y_pred * np.arange(num_samples).reshape((num_samples, 1))
    b1 = b1_values.mean(axis=0) / num_samples

    per_obs_crps = absolute_error + b0 - 2 * b1
    return np.mean(per_obs_crps)


def lp_loss(predictions, targets, p=1):
    """
    Compute the Lp loss between predictions and targets.

    Args:
        predictions (torch.Tensor): Predicted values.
        targets (torch.Tensor): Target values.
        p (float): Value of p for Lp loss.

    Returns:
        torch.Tensor: Lp loss.
    """
    assert predictions.size() == targets.size(), "predictions and targets must have the same shape"
    
    # Calculate the absolute difference
    abs_diff = torch.abs(predictions - targets)
    
    # Calculate the Lp loss
    lp_loss = torch.pow(abs_diff, p).mean() #/ torch.pow(torch.abs(targets), p).mean()

    return lp_loss

def relative_l1_error(predictions, targets, epsilon=1e-8):
    """
    Compute the relative L1 error between predictions and targets.

    Args:
        predictions (torch.Tensor): Predicted values.
        targets (torch.Tensor): Target values.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        torch.Tensor: Relative L1 error.
    """
    abs_diff = torch.abs(predictions - targets)
    abs_sum = torch.abs(targets) + epsilon
    relative_error = torch.mean(abs_diff / abs_sum)*100

    return relative_error

def extrapolate_linear(tensor, num_points=10, dim=-1):
    """
    Extrapolate linearly from the beginning and end of a PyTorch tensor along a specified dimension.

    Args:
        tensor (torch.Tensor): Input PyTorch tensor.
        num_points (int): Number of points to extrapolate at each end.
        dim (int): Dimension along which to extrapolate.

    Returns:
        torch.Tensor: Extrapolated tensor with additional points at the beginning and end along the specified dimension.
    """
    # Get the number of points in the original tensor along the specified dimension
    batch_size = tensor.shape[0]
    channels = tensor.shape[1]
    num_orig_points = tensor.size(dim)

    gradients = torch.gradient(tensor, dim=-1, edge_order=2)[0]

    start_extrap = torch.ones((batch_size, channels, num_points)) * tensor[:,:,0].unsqueeze(-1)
    start_grad = gradients[:,:,0].unsqueeze(-1).repeat(1,1,num_points)
    start = torch.cumsum(start_grad, dim=-1)
    start = torch.flip(start, [-1])
    start_extrap += start
    
    end_extrap = torch.ones((batch_size, channels, num_points)) * tensor[:,:,-1].unsqueeze(-1)
    end_grad = gradients[:,:,-1].unsqueeze(-1).repeat(1,1,num_points)
    end = torch.cumsum(end_grad, dim=-1)
    end_extrap += end
                              
    # Concatenate extrapolated segments with original tensor along the specified dimension
    extrap_tensor = torch.cat((start_extrap, tensor, end_extrap), dim=dim)

    return extrap_tensor

def extrapolate_constant(tensor, num_points=10, dim=-1):
    """
    Pad with a constant from the beginning and end of a PyTorch tensor along a specified dimension.

    Args:
        tensor (torch.Tensor): Input PyTorch tensor.
        num_points (int): Number of points to extrapolate at each end.
        dim (int): Dimension along which to extrapolate.

    Returns:
        torch.Tensor: Padded tensor with additional points at the beginning and end along the specified dimension.
    """
    # Get the number of points in the original tensor along the specified dimension
    batch_size = tensor.shape[0]
    channels = tensor.shape[1]
    num_orig_points = tensor.size(dim)

    start_extrap = torch.ones((batch_size, channels, num_points)) * tensor[:,:,0].unsqueeze(-1)
    end_extrap = torch.ones((batch_size, channels, num_points)) * tensor[:,:,-1].unsqueeze(-1)
    # Concatenate extrapolated segments with original tensor along the specified dimension
    extrap_tensor = torch.cat((start_extrap, tensor, end_extrap), dim=dim)

    return extrap_tensor

def extrapolate_zeros(tensor, num_points=10, dim=-1):
    """
    Pad with zeros from the beginning and end of a PyTorch tensor along a specified dimension.

    Args:
        tensor (torch.Tensor): Input PyTorch tensor.
        num_points (int): Number of points to extrapolate at each end.
        dim (int): Dimension along which to extrapolate.

    Returns:
        torch.Tensor: Padded tensor with additional points at the beginning and end along the specified dimension.
    """
    # Get the number of points in the original tensor along the specified dimension
    batch_size = tensor.shape[0]
    channels = tensor.shape[1]
    num_orig_points = tensor.size(dim)

    start_extrap = torch.zeros((batch_size, channels, num_points)) 
    end_extrap = torch.zeros((batch_size, channels, num_points))
    # Concatenate extrapolated segments with original tensor along the specified dimension
    extrap_tensor = torch.cat((start_extrap, tensor, end_extrap), dim=dim)

    return extrap_tensor

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
    numerator = torch.norm(predicted - target, p=p, dim=-1)
    denominator = torch.norm(target, p=p, dim=-1)
    error = (numerator / denominator)*100
    return error