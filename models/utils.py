import torch
import numpy as np

def fanin_init(size, fanin=None):
  fanin = fanin or size[0]
  v = 1. / np.sqrt(fanin)
  return torch.Tensor(size).uniform_(-v, v)