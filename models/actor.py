import torch
from torch import nn

from .utils import fanin_init

class Actor(nn.Module):
  def __init__(self, in_dims, out_dims):
    super(Actor, self).__init__()
    self.is_disc_action = True

    self.fc1 = nn.Linear(in_dims, 256)
    self.fc2 = nn.Linear(256, 256)
    self.fc3 = nn.Linear(256, 256)

    self.ind_layers = nn.ModuleList()
    for i in range(len(out_dims)):
      self.ind_layers.append(nn.Linear(256, out_dims[i]))

    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()
    #self._init_weights()
    self.fc3.weight.data.mul_(0.1)
    self.fc3.bias.data.mul_(0.0)

  def _init_weights(self):
    self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
    self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
    self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
    self.fc4.weight.data = fanin_init(self.fc4.weight.data.size())

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(self.fc2(x))
    x = self.relu(self.fc3(x))

    out = [] 
    for layer in self.ind_layers:
      out.append(torch.softmax(layer(x), dim=0))
    return torch.cat(out, dim=-1)  

  def get_log_prob(self, x, actions):
    action_prob = self.forward(x)
    return torch.log(action_prob.gather(1, actions.long()))

  def get_fim(self, x):
    action_prob = self.forward(x)
    M = action_prob.pow(-1).view(-1).detach()
    return M, action_prob, {}
  
  def get_kl(self, x):
    prob1 = self.forward(x)
    prob0 = prob1.detach()
    kl = prob0 * (torch.log(prob0) - torch.log(prob1))
    return kl.sum(1, keepdim=True)