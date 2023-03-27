import json

import numpy as np
import torch
import pandas as pd

cuda_available = torch.cuda.is_available()

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

def to_tensor(ndarray):
  return torch.from_numpy(ndarray.astype(np.float32))
