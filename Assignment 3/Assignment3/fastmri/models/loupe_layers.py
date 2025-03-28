"""
    nn.Modules for LOUPE
    
    For more details, please read:
    
    Bahadir, Cagla Deniz, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Learning-based Optimization of the Under-sampling Pattern in MRI." 
    IPMI 2019. https://arxiv.org/abs/1901.01960.
"""


# third party
# from keras.layers import nn.Module
# import keras.backend as K
# import tensorflow as tf
# from keras.initializers import RandomUniform, RandomNormal
import torch
import torch.nn as nn
import pytorch_lightning as pl


class RescaleProbMap(nn.Module):
    """
    Rescale Probability Map
    given a prob map x, rescales it so that it obtains the desired sparsity
    """
    
    def __init__(self, sparsity):
        super(RescaleProbMap, self).__init__()
        self.sparsity = sparsity

    def forward(self, x):
        xbar = torch.mean(x)
        r = self.sparsity / xbar
        beta = (1 - self.sparsity) / (1 - xbar)
        le = (r <= 1).float()
        return le * x * r + (1 - le) * (1 - (1 - x) * beta)


class ProbMask(nn.Module):
    """ 
    Probability mask layer
    Contains a layer of weights, that is then passed through a sigmoid.

    Modified from Local Linear Layer code in https://github.com/adalca/neuron
    """
    
    def __init__(self, slope=1,
                 initializer=None, shape=None,
                 **kwargs):
        super(ProbMask, self).__init__()

        if initializer == None:
            self.initializer = self._logit_slope_random_uniform
        else:
            self.initializer = initializer

        # Higher slope means a more step-function-like logistic function
        # note: slope is converted to a tensor so that we can update it 
        #   during training if necessary
        self.slope = nn.Parameter(torch.abs(torch.tensor(slope, dtype=torch.float32)))

        # Initialize self.mult with a placeholder shape
        if shape is not None:
            self.mult = nn.Parameter(self.initializer(shape).to(self.device))
        else:
            self.mult = nn.Parameter(torch.empty(1))

        
    def _logit_slope_random_uniform(self, shape, eps=0.0001):
        # eps could be very small, or something like eps = 1e-6
        # The idea is how far from the tails to have your initialization.
        x = torch.rand(shape).uniform_(eps, 1.0 - eps).to(self.device)  # [0, 1]
        
        # Logit with slope factor
        return -torch.log(1.0 / x - 1.0) / self.slope

    
    def forward(self, x):
        if self.mult.numel() == 1:
            input_shape_h = list(x.shape)
            input_shape_h[-1] = 1
            self.mult = nn.Parameter(self.initializer(input_shape_h[1:])).to(self.device)

        logit_weights = torch.zeros_like(x[..., 0:1]) + self.mult

        # print("slope in probmask: ", self.slope)
    
        # print("10 logit weights:", logit_weights.flatten()[:10])

        return torch.sigmoid(self.slope * logit_weights)

    @property
    def device(self):
        return next(self.parameters()).device
    
class ThresholdRandomMask(nn.Module):
    """ 
    Local thresholding layer
    Takes as input the input to be thresholded, and the threshold, the threshold can be the random mask
    """
    
    def __init__(self, slope):
        super(ThresholdRandomMask, self).__init__()
        # self.slope = nn.Parameter(torch.tensor(slope, dtype=torch.float32)) if slope is not None else None
        self.slope = slope if slope is not None else None

    def forward(self, inputs, thresh):
        print("slope in  threshold: ", self.slope)
        if self.slope is not None:
            return torch.sigmoid(self.slope * (inputs - thresh))
        else:
            return (inputs > thresh).bool()

class RandomMask(nn.Module):
    """ 
    Create a random binary mask of the same size as the input shape
    """
    
    def __init__(self):
        super(RandomMask, self).__init__()

    def forward(self, x):
        input_shape = x.shape
        with torch.random.fork_rng():
            torch.manual_seed(torch.randint(0, 2**32 - 1, (1,)).item())
        threshs = torch.rand(input_shape, dtype=torch.float32).to(x.device)

        return (0 * x) + threshs


class UnderSample(nn.Module):
    """
    Under-sampling by multiplication of k-space with the mask
    """
    
    def __init__(self):
        super(UnderSample, self).__init__()

    def forward(self, kspace, mask):
        k_space_r = kspace[..., 0] * mask[..., 0]
        k_space_i = kspace[..., 1] * mask[..., 0]
        k_space = torch.stack([k_space_r, k_space_i], dim=-1)
        return k_space