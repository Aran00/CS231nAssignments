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
    if use_batch_norm:
      self.bn_param = {'mode': 'train', 'gamma': np.ones(num_filters), 'beta': np.zeros(num_filters)}
    

class HiddenLayerConfig(object):
  """
  The configuration class contains these parameters:
  1. hidden layer dim
  2. use batch norm or not
  """   
  def __init__(self, hidden_layer_dim=100, use_batch_norm=False):
    self.hidden_layer_dim = hidden_layer_dim
    self.use_batch_norm = use_batch_norm
    if use_batch_norm: 
      self.bn_param = {'mode': 'train', 'gamma': np.ones(hidden_layer_dim), 'beta': np.zeros(hidden_layer_dim)}  
               
    
class MultipleLayerConvNet(object):
  """
  A multiple layer convolutional network with the following architecture:
  
  # [conv-relu-pool]xN - conv - relu - [affine]xM - [softmax or SVM]
  # [conv-relu-pool]XN - [affine]XM - [softmax or SVM]
  I think the first could be one special case of the second, with the pool {'pool_height': 1, 'pool_width': 1, 'stride': 1}, so we can only realize the model of 
  [conv-relu-pool]XN - [affine-relu]XM - [softmax or SVM] (Not sure why the tutor doesn't add relu behind affine...)

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  
  Just assume the stride and maxpool params are all the same. But still we 
  can pass them in.
  """
    
  def __init__(self, conv_layer_configs, hidden_layer_configs, input_dim=(3, 32, 32),
               num_classes=10, weight_scale=1e-3, reg=0.0, verbose=False,
               dtype=np.float32):
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
    self.conv_layer_configs = conv_layer_configs
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
      filter_size = conv_config.filter_size
      H = 1 + (H + 2 * padding - filter_size)/conv_stride
      W = 1 + (W + 2 * padding - filter_size)/conv_stride
      # After pooling
      pool_height = conv_config.pool_param["pool_height"]
      pool_width = conv_config.pool_param["pool_width"]
      pool_stride = conv_config.pool_param["stride"]
      H = 1 + (H - pool_height)/pool_stride
      W = 1 + (W - pool_width)/pool_stride  
      if H<=0 or W<=0:
         raise ValueError('Invalid output size: H=%d, W=%d"' % (H, W))  
    
    conv_output_dim = H * W * conv_layer_configs[conv_layer_count-1].num_filters
    
    # Because conv layers have an output, the remain layers count is len(hidden_layers) + 1, and the last is output layer
    self.hidden_layer_configs = hidden_layer_configs
    hidden_layer_count = len(hidden_layer_configs)
    for i in xrange(hidden_layer_count):
      hidden_layer_dim = hidden_layer_configs[i].hidden_layer_dim
      idx = conv_layer_count + i
      layer_index = idx + 1
      input_dim = conv_output_dim if i == 0 else hidden_layer_configs[i-1].hidden_layer_dim
      self.params['W%d'%layer_index] = weight_scale * np.random.randn(input_dim, hidden_layer_dim)
      self.params['b%d'%layer_index] = np.zeros(hidden_layer_dim)

    output_layer_idx = conv_layer_count + self.hidden_layer_count + 1
    self.params['W%d'%output_layer_idx] = weight_scale * np.random.randn(hidden_layer_configs[hidden_layer_count-1], num_classes)
    self.params['b%d'%output_layer_idx] = np.zeros(num_classes)
  
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the multi-layer convolutional-affine network.
    """
    mode = 'test' if y is None else 'train'
    
    conv_layer_count = len(self.conv_layer_configs)
    hidden_layer_count = len(self.hidden_layer_configs)
    next_layer_input = X
    layer_caches = []
    # Forward to get output and loss
    for i in xrange(conv_layer_count + hidden_layer_count):
      layer_idx = i + 1
      W_cur = self.params['W%d'%layer_idx]
      b_cur = self.params['b%d'%layer_idx]
      if i < conv_layer_count:
        config = self.conv_layer_configs[i]
        if config.use_batch_norm:
          config.bn_param["mode"] = mode
          current_layer_out, current_layer_cache = conv_bn_relu_pool_forward(next_layer_input, W_cur, b_cur,
                                                                          config.conv_param, config.bn_param, config.pool_param)
        else:
          current_layer_out, current_layer_cache = conv_relu_pool_forward(next_layer_input, W_cur, b_cur, config.conv_param, config.pool_param)
      else:
        config = self.hidden_layer_configs[i-conv_layer_count]
        if config.use_batch_norm:
          config.bn_param["mode"] = mode
          current_layer_out, current_layer_cache = affine_bn_relu_forward(next_layer_input, W_cur, b_cur, config.bn_param)
        else:
          current_layer_out, current_layer_cache = affine_relu_forward(next_layer_input, W_cur, b_cur)
      layer_caches.append(current_layer_cache)
      next_layer_input = current_layer_out

    output_layer_idx = conv_layer_count + hidden_layer_count + 1
    W_output = self.params['W%d'%output_layer_idx]
    b_output = self.params['b%d'%output_layer_idx]
    scores, final_layer_cache = affine_forward(next_layer_input, W_output, b_output)

    if y is None:
      return scores

    loss, grads = 0, {}
    # Backward to get gradient
    loss, dscores = softmax_loss(scores, y)  
    d_out, grads['W%d'%output_layer_idx], grads['b%d'%output_layer_idx] = affine_backward(dscores, final_layer_cache)
    
    for i in xrange(conv_layer_count + hidden_layer_count - 1, -1, -1):
      layer_idx = i + 1
      if i >= conv_layer_count:
        config = self.hidden_layer_configs[i - conv_layer_count]
        backward_func = affine_bn_relu_backward if config.use_batch_norm else affine_relu_backward
      else:
        config = self.conv_layer_configs[i]
        backward_func = conv_bn_relu_pool_backward if config.use_batch_norm else conv_relu_pool_backward
      d_out, grads['W%d'%layer_idx], grads['b%d'%layer_idx] = backward_func(d_out, layer_caches[i])
    
    # Add regulation
    for i in xrange(conv_layer_count + hidden_layer_count + 1):
        layer_idx = i + 1
        W = self.params['W%d'%layer_idx]
        loss += 0.5 * self.reg * np.sum(np.power(W, 2))
        grads['W%d'%layer_idx] += self.reg * W
    
    return loss, grads
