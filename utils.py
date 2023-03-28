import json

import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable

#cuda_available = torch.cuda.is_available()
cuda_available = False
f = torch.cuda.FloatTensor if cuda_available else torch.FloatTensor

def read_json(path):
  config = []
  with open(path) as f:
    config = json.load(f)
  return config

def read_csv(path):
  data = pd.read_csv(path)
  return data

def to_numpy(tensor):
  return tensor.cpu().data.numpy() if cuda_available else tensor.data.numpy()

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=f):
  return Variable(torch.from_numpy(ndarray.astype(np.float32)), volatile=volatile, requires_grad=requires_grad).type(dtype)