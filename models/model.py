import torch
from torch import nn

class Model(nn.Module):
  def __init__(self, in_dims, out_dims):
    super().__init__()
    self.fc1 = nn.Linear(in_dims, 256)
    self.fc2 = nn.Linear(256, 256)
    self.fc3 = nn.Linear(256, 256)
    self.fc4 = nn.Linear(256, out_dims)

  def _init_weights(self):
    pass

  def forward(self, x):
    x = self.fc1(x)
    x = nn.ReLU(self.fc2(x))
    x = nn.ReLU(self.fc3(x))
    x = nn.Tanh(self.fc4(x))
    return x