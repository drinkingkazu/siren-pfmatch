import torch

def rel_bias(target : torch.Tensor, pred : torch.Tensor):
    '''
    Function to compute the relative bias (the mean of 2 * |target - pred| / (target + pred))

    Parameters
    ----------
    target : torch.Tensor
        The reference based on which the bias is calculated.
    pred : torch.Tensor
        The subject for which the bias is calculated.
    Returns
    -------
    torch.Tensor
        The model visibility bias.
    '''
    if target.shape != pred.shape:
        raise ValueError(f'target and pred must have the same shape {(*target.shape,)} != {(*pred.shape,)}')

    bias = (2 * torch.abs(pred-target) / (pred+target)).mean()
    return bias

def abs_bias(target: torch.Tensor, pred : torch.Tensor):
    '''
    Function to compute the absolute bias (the mean of |target - pred|)
    
    Parameters
    ----------
    target : torch.Tensor
        Some reference target based on which the bias is calculated.
    pred : torch.Tensor
        Prediction for target on which which the bias is calculated.
        
    Returns
    -------
    torch.Tensor
        The model absolute bias.

    '''
    if target.shape != pred.shape:
        raise ValueError(f'target and pred must have the same shape {(*target.shape,)} != {(*pred.shape,)}')
    
    return torch.abs(target - pred).mean()