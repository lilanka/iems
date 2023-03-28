from torch import nn

from .utils import fanin_init

class Actor(nn.Module):
  def __init__(self, in_dims, out_dims):
    super(Actor, self).__init__()
    self.fc1 = nn.Linear(in_dims, 256)
    self.fc2 = nn.Linear(256, 256)
    self.fc3 = nn.Linear(256, 256)
    self.fc4 = nn.Linear(256, out_dims)
    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()
    self._init_weights()

  def _init_weights(self):
    self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
    self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
    self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
    self.fc4.weight.data = fanin_init(self.fc4.weight.data.size())

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(self.fc2(x))
    x = self.relu(self.fc3(x))
    x = self.tanh(self.fc4(x))
    return x