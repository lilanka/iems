import torch
from torch import nn

class Critic(nn.Module):
  def __init__(self, n_obs, n_actions):
    super(Critic, self).__init__()
    self.fc1 = nn.Linear(n_obs, 256)
    self.fc2 = nn.Linear(256 + n_actions, 256)
    self.fc3 = nn.Linear(256, 1)
    self.relu = nn.ReLU()
    self.fc3.weight.data.mul_(0.1)
    self.fc3.bias.data.mul_(0.0)

  def forward(self, xs):
    x, a = xs
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(torch.cat([out, a], 1))
    out = self.relu(out)
    out = self.fc3(out)
    return out