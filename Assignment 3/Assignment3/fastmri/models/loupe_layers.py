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
        """
        note that in v1 the initial initializer was uniform in [-A, +A] where A is some scalar.
        e.g. was RandomUniform(minval=-2.0, maxval=2.0, seed=None),
        But this is uniform *in the logit space* (since we take sigmoid of this), so probabilities
        were concentrated a lot in the edges, which led to very slow convergence, I think.

        IN v2, the default initializer is a logit of the uniform [0, 1] distribution,
        which fixes this issue
        """
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
            
        # print("slope: ", self.slope)
        logit_weights = torch.zeros_like(x[..., 0:1]) + self.mult
        # if not hasattr(self, 'prev_logit_weights'):
        #     self.prev_logit_weights = logit_weights.clone()
        
        # any_value_changed = not torch.equal(logit_weights, self.prev_logit_weights)
        # self.prev_logit_weights = logit_weights.clone()
        
        # print("Any value changed:", any_value_changed)
    
        print("10 logit weights:", logit_weights.flatten()[:10])

        return torch.sigmoid(self.slope * logit_weights)

    @property
    def device(self):
        return next(self.parameters()).device
    
class ThresholdRandomMask(nn.Module):
    """ 
    Local thresholding layer
    Takes as input the input to be thresholded, and the threshold
    """
    
    def __init__(self, slope):
        super(ThresholdRandomMask, self).__init__()
        # self.slope = nn.Parameter(torch.tensor(slope, dtype=torch.float32)) if slope is not None else None
        self.slope = slope if slope is not None else None

    def forward(self, inputs, thresh):
        if self.slope is not None:
            ######################################## test ########################################
            # mask = torch.sigmoid(self.slope * (inputs - thresh))
            # mask = torch.sigmoid(self.slope * (inputs))
            return torch.sigmoid(self.slope * (inputs - thresh))
            # return (mask > 0.5).bool()
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
    

# class ComplexAbs(nn.Module):
#     """
#     Complex Absolute

#     Inputs: [kspace, mask]
#     """

#     def __init__(self, **kwargs):
#         super(ComplexAbs, self).__init__(**kwargs)

#     def build(self, input_shape):
#         super(ComplexAbs, self).build(input_shape)

#     def call(self, inputs):
#         two_channel = tf.complex(inputs[..., 0], inputs[..., 1])
#         two_channel = tf.expand_dims(two_channel, -1)
        
#         two_channel = tf.abs(two_channel)
#         two_channel = tf.cast(two_channel, tf.float32)
#         return two_channel

#     def compute_output_shape(self, input_shape):
#         list_input_shape = list(input_shape)
#         list_input_shape[-1] = 1
#         return tuple(list_input_shape)


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


# class ConcatenateZero(nn.Module):
#     """
#     Concatenate input with a zero'ed version of itself

#     Input: tf.float32 of size [batch_size, ..., n]
#     Output: tf.float32 of size [batch_size, ..., n*2]
#     """

#     def __init__(self, **kwargs):
#         super(ConcatenateZero, self).__init__(**kwargs)

#     def build(self, input_shape):
#         super(ConcatenateZero, self).build(input_shape)

#     def call(self, inputx):
#         return tf.concat([inputx, inputx*0], -1)


#     def compute_output_shape(self, input_shape):
#         input_shape_list = list(input_shape)
#         input_shape_list[-1] *= 2
#         return tuple(input_shape_list)


# class FFT(nn.Module):
#     """
#     fft layer, assuming the real/imag are input/output via two features

#     Input: tf.float32 of size [batch_size, ..., 2]
#     Output: tf.float32 of size [batch_size, ..., 2]
#     """

#     def __init__(self, **kwargs):
#         super(FFT, self).__init__(**kwargs)

#     def build(self, input_shape):
#         # some input checking
#         assert input_shape[-1] == 2, 'input has to have two features'
#         self.ndims = len(input_shape) - 2
#         assert self.ndims in [1,2,3], 'only 1D, 2D or 3D supported'

#         # super
#         super(FFT, self).build(input_shape)

    # def call(self, inputx):
    #     assert inputx.shape.as_list()[-1] == 2, 'input has to have two features'

    #     # get the right fft
    #     if self.ndims == 1:
    #         fft = tf.fft
    #     elif self.ndims == 2:
    #         fft = tf.fft2d
    #     else:
    #         fft = tf.fft3d

    #     # get fft complex image
    #     fft_im = fft(tf.complex(inputx[..., 0], inputx[..., 1]))

    #     # go back to two-feature representation
    #     fft_im = tf.stack([tf.real(fft_im), tf.imag(fft_im)], axis=-1)
    #     return tf.cast(fft_im, tf.float32)

    # def compute_output_shape(self, input_shape):
    #     return input_shape


# class IFFT(nn.Module):
#     """
#     ifft layer, assuming the real/imag are input/output via two features

#     Input: tf.float32 of size [batch_size, ..., 2]
#     Output: tf.float32 of size [batch_size, ..., 2]
#     """

#     def __init__(self, **kwargs):
#         super(IFFT, self).__init__(**kwargs)

#     def build(self, input_shape):
#         # some input checking
#         assert input_shape[-1] == 2, 'input has to have two features'
#         self.ndims = len(input_shape) - 2
#         assert self.ndims in [1,2,3], 'only 1D, 2D or 3D supported'

#         # super
#         super(IFFT, self).build(input_shape)

    # def call(self, inputx):
    #     assert inputx.shape.as_list()[-1] == 2, 'input has to have two features'

    #     # get the right fft
    #     if self.ndims == 1:
    #         ifft = tf.ifft
    #     elif self.ndims == 2:
    #         ifft = tf.ifft2d
    #     else:
    #         ifft = tf.ifft3d

    #     # get ifft complex image
    #     ifft_im = ifft(tf.complex(inputx[..., 0], inputx[..., 1]))

    #     # go back to two-feature representation
    #     ifft_im = tf.stack([tf.real(ifft_im), tf.imag(ifft_im)], axis=-1)
    #     return tf.cast(ifft_im, tf.float32)

    # def compute_output_shape(self, input_shape):
    #     return input_shape
