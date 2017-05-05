import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvLayerConfig(object):
  """
  The configuration class contains these parameters:
  1. filter size
  2. number of filters
  3. conv parameter (conv_param): stride, pad
  4. pool parameter (pool_param): pool_height, pool_width, stride
  5. Use spatial batch norm or not
  """
  def __init__(self, num_filters=32, filter_size=3, stride=1, pad=-1, 
               pool_height=2, pool_width=2, pool_stride=2, use_batch_norm=False):
    pad = pad if pad >= 0 else (filter_size - 1) / 2
    self.num_filters = num_filters
    self.filter_size = filter_size
    self.conv_param = {
      'stride': stride,
      'pad': pad  
    }
    self.pool_param = {
      'pool_height': pool_height,
      'pool_widht': pool_width,
      'stride': pool_stride
    }
    self.use_batch_norm = use_batch_norm


class MultipleLayerConvNet(object):
  """
  A multiple layer convolutional network with the following architecture:
  
  # [conv-relu-pool]xN - conv - relu - [affine]xM - [softmax or SVM]
  # [conv-relu-pool]XN - [affine]XM - [softmax or SVM]
  I think the first could be one special case of the second, with the pool {'pool_height': 1, 'pool_width': 1, 'stride': 1}, so we can only realize the model of 
  [conv-relu-pool]XN - [affine]XM - [softmax or SVM]

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  
  Just assume the stride and maxpool params are all the same. But still we 
  can pass them in.
  """
    
  def __init__(self, input_dim=(3, 32, 32), conv_layer_configs, 
               num_classes=10, weight_scale=1e-3, reg=0.0, verbose=False
               dtype=np.float32)
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'Wi' #
    # and 'bi', where 1<=i<=conv_layer_count; use keys 'W(conv_layer_count+1)' #
    # and 'b(conv_layer_count+1)' for the weights and biases of the output     #    
    # affine layer.                                                            #
    ############################################################################
    C, H, W = input_dim
    conv_layer_count = len(conv_layer_configs)
    
    for i in xrange(conv_layer_count):
      layer_index = i + 1
      conv_config = conv_layer_configs[i]
      channel_count = C if i == 0 else conv_layer_configs[i-1].num_filters
      
      self.params['W%d'%layer_index] = weight_scale * np.random.randn(conv_config.num_filters, channel_count, conv_config.filter_size, conv_config.filter_size)
      self.params['b%d'%layer_index] = np.zeros(conv_config.num_filters)
      # We still need an iteration to get the final output size here  
      # After conv
      padding = conv_config.conv_param["pad"]
      conv_stride = conv_config.conv_param["stride"]
      H = 1 + (H + 2 * padding - filter_size)/conv_stride
      W = 1 + (W + 2 * padding - filter_size)/conv_stride
      # After pooling
      pool_height = conv_config.pool_param["pool_height"]
      pool_width = conv_config.pool_param["pool_width"]
      pool_stride = conv_config.pool_param["stride"]
      H = 1 + (H - pool_height)/pool_stride
      W = 1 + (W - pool_width)/pool_stride  
      if H<=1 or W<=1:
         raise ValueError('Invalid output size: H=%d, W=%d"' % (H, W))  
    
    output_dim = H * W * conv_layer_configs[conv_layer_count-1].num_filters
    output_layer_idx = conv_layer_count + 1
    self.params['W%d'%output_layer_idx] = weight_scale * np.random.randn(output_dim, num_classes)
    self.params['b%d'%output_layer_idx] = np.zeros(num_classes)
    
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
    
    
  def loss(self, X, y=None):
    pass

