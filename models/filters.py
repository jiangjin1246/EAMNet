import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.util_filters import rgb2lum, tanh_range, lerp
from models.util_filters import *

import cv2
import math
import pdb

# device = torch.device("cuda")

class Filter(nn.Module):

  def __init__(self, net, cfg):
    super(Filter, self).__init__()

    self.cfg = cfg
    # self.height, self.width, self.channels = list(map(int, net.get_shape()[1:]))

    # Specified in child classes
    self.num_filter_parameters = None
    self.short_name = None
    self.filter_parameters = None

  def get_short_name(self):
    assert self.short_name
    return self.short_name

  def get_num_filter_parameters(self):
    assert self.num_filter_parameters
    return self.num_filter_parameters

  def get_begin_filter_parameter(self):
    return self.begin_filter_parameter

  def extract_parameters(self, features):
    # output_dim = self.get_num_filter_parameters(
    # ) + self.get_num_mask_parameters()
    # features = ly.fully_connected(
    #     features,
    #     self.cfg.fc1_size,
    #     scope='fc1',
    #     activation_fn=lrelu,
    #     weights_initializer=tf.contrib.layers.xavier_initializer())
    # features = ly.fully_connected(
    #     features,
    #     output_dim,
    #     scope='fc2',
    #     activation_fn=None,
    #     weights_initializer=tf.contrib.layers.xavier_initializer())
    return features[:, self.get_begin_filter_parameter():(self.get_begin_filter_parameter() + self.get_num_filter_parameters())], \
           features[:, self.get_begin_filter_parameter():(self.get_begin_filter_parameter() + self.get_num_filter_parameters())]

  # Should be implemented in child classes
  def filter_param_regressor(self, features):
    assert False

  # Process the whole image, without masking
  # Should be implemented in child classes
  def process(self, img, param, defog, IcA):
    assert False

  def debug_info_batched(self):
    return False

  def no_high_res(self):
    return False

  # Apply the whole filter with masking
  def apply(self,
            img,
            img_features=None,
            defog_A=None,
            IcA=None,
            specified_parameter=None,
            high_res=None):
    assert (img_features is None) ^ (specified_parameter is None)
    if img_features is not None:
      filter_features, mask_parameters = self.extract_parameters(img_features)
      filter_parameters = self.filter_param_regressor(filter_features)
    else:
      assert not self.use_masking()
      filter_parameters = specified_parameter

    if high_res is not None:
      # working on high res...
      pass
    debug_info = {}
    # We only debug the first image of this batch
    if self.debug_info_batched():
      debug_info['filter_parameters'] = filter_parameters
    else:
      debug_info['filter_parameters'] = filter_parameters[0]
    # self.mask_parameters = mask_parameters
    # self.mask = self.get_mask(img, mask_parameters)
    # debug_info['mask'] = self.mask[0]
    #low_res_output = lerp(img, self.process(img, filter_parameters), self.mask)
    low_res_output = self.process(img, filter_parameters, defog_A, IcA)

    if high_res is not None:
      if self.no_high_res():
        high_res_output = high_res
      else:
        self.high_res_mask = self.get_mask(high_res, mask_parameters)
        # high_res_output = lerp(high_res,
        #                        self.process(high_res, filter_parameters, defog, IcA),
        #                        self.high_res_mask)
    else:
      high_res_output = None
    #return low_res_output, high_res_output, debug_info
    return low_res_output, filter_parameters

  def use_masking(self):
    return self.cfg.masking

  def get_num_mask_parameters(self):
    return 6

  # Input: no need for tanh or sigmoid
  # Closer to 1 values are applied by filter more strongly
  # no additional TF variables inside
  def get_mask(self, img, mask_parameters):
    if not self.use_masking():
      print('* Masking Disabled')
      return tf.ones(shape=(1, 1, 1, 1), dtype=tf.float32)
    else:
      print('* Masking Enabled')
    with tf.name_scope(name='mask'):
      # Six parameters for one filter
      filter_input_range = 5
      assert mask_parameters.shape[1] == self.get_num_mask_parameters()
      mask_parameters = tanh_range(
          l=-filter_input_range, r=filter_input_range,
          initial=0)(mask_parameters)
      size = list(map(int, img.shape[1:3]))
      grid = np.zeros(shape=[1] + size + [2], dtype=np.float32)

      shorter_edge = min(size[0], size[1])
      for i in range(size[0]):
        for j in range(size[1]):
          grid[0, i, j,
               0] = (i + (shorter_edge - size[0]) / 2.0) / shorter_edge - 0.5
          grid[0, i, j,
               1] = (j + (shorter_edge - size[1]) / 2.0) / shorter_edge - 0.5
      grid = tf.constant(grid)
      # Ax + By + C * L + D
      inp = grid[:, :, :, 0, None] * mask_parameters[:, None, None, 0, None] + \
            grid[:, :, :, 1, None] * mask_parameters[:, None, None, 1, None] + \
            mask_parameters[:, None, None, 2, None] * (rgb2lum(img) - 0.5) + \
            mask_parameters[:, None, None, 3, None] * 2
      # Sharpness and inversion
      inp *= self.cfg.maximum_sharpness * mask_parameters[:, None, None, 4,
                                                          None] / filter_input_range
      mask = tf.sigmoid(inp)
      # Strength
      mask = mask * (
          mask_parameters[:, None, None, 5, None] / filter_input_range * 0.5 +
          0.5) * (1 - self.cfg.minimum_strength) + self.cfg.minimum_strength
      print('mask', mask.shape)
    return mask

  # def visualize_filter(self, debug_info, canvas):
  #   # Visualize only the filter information
  #   assert False

  def visualize_mask(self, debug_info, res):
    return cv2.resize(
        debug_info['mask'] * np.ones((1, 1, 3), dtype=np.float32),
        dsize=res,
        interpolation=cv2.cv2.INTER_NEAREST)

  def draw_high_res_text(self, text, canvas):
    cv2.putText(
        canvas,
        text, (30, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 0, 0),
        thickness=5)
    return canvas




class UsmFilter(Filter):#Usm_param is in [Defog_range]

  def __init__(self, net, cfg):

    Filter.__init__(self, net, cfg)
    self.short_name = 'UF'
    self.begin_filter_parameter = cfg.usm_begin_param
    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):
    return tanh_range(*self.cfg.usm_range)(features)

  def process(self, img, param, defog_A, IcA):


    self.channels = 3
    kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
              [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
              [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
              [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
              [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = np.repeat(kernel, self.channels, axis=0)

    # print('      param:', param)

    kernel = kernel.to(img.device)
    # self.weight = nn.Parameter(data=kernel, requires_grad=False)
    # self.weight.to(device)

    output = F.conv2d(img, kernel, padding=2, groups=self.channels)


    img_out = (img - output) * param + img
    # img_out = (img - output) * torch.tensor(0.043).cuda() + img

    return img_out

class ContrastFilter(Filter):

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.short_name = 'Ct'
    self.begin_filter_parameter = cfg.contrast_begin_param

    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):
    # return tf.sigmoid(features)
    # return torch.tanh(features)
    return tanh_range(*self.cfg.cont_range)(features)

  def process(self, img, param, defog, IcA):
    # print('      param.shape:', param.shape)

    # luminance = torch.minimum(torch.maximum(rgb2lum(img), 0.0), 1.0)
    luminance = rgb2lum(img)
    zero = torch.zeros_like(luminance)
    one = torch.ones_like(luminance)

    luminance = torch.where(luminance < 0, zero, luminance)
    luminance = torch.where(luminance > 1, one, luminance)

    contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
    contrast_image = img / (luminance + 1e-6) * contrast_lum
    return lerp(img, contrast_image, param)
    # return lerp(img, contrast_image, torch.tensor(0.015).cuda())




class GammaFilter(Filter):  #gamma_param is in [1/gamma_range, gamma_range]

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.short_name = 'G'
    self.begin_filter_parameter = cfg.gamma_begin_param
    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):
    log_gamma_range = np.log(self.cfg.gamma_range)
    # return tf.exp(tanh_range(-log_gamma_range, log_gamma_range)(features))
    return torch.exp(tanh_range(-log_gamma_range, log_gamma_range)(features))

  def process(self, img, param, defog_A, IcA):
    # print('      param:', param)

    # param_1 = param.repeat(1, 3)
    zero = torch.zeros_like(img) + 0.00001
    img = torch.where(img <= 0, zero, img)
    # print("GAMMMA", param)
    return torch.pow(img, param)
    # return torch.pow(img, torch.tensor(0.51).cuda())


