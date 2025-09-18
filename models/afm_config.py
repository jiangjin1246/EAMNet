import argparse
from easydict import EasyDict as edict
from models.filters import *


__C                             = edict()
# Consumers can get config by: from config import cfg

cfg                             = __C

###########################################################################
# Filter Parameters
###########################################################################


cfg.filters = [
    GammaFilter, ContrastFilter, UsmFilter
]
cfg.num_filter_parameters = 3

#cfg.wb_begin_param = 0
cfg.gamma_begin_param = 0
cfg.contrast_begin_param = 1
cfg.usm_begin_param = 2

cfg.gamma_range = 3
cfg.usm_range = (0.0, 5)
cfg.cont_range = (0.0, 1.0)