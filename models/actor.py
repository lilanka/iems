import torch
from torch import nn

from .utils import * 

class Actor(nn.Module):
  def __init__(self, in_dims, out_dims, log_std=0):
    super(Actor, self).__init__()
    self.is_disc_action = False 

    self.fc1 = nn.Linear(in_dims, 256)
    self.fc2 = nn.Linear(256, 256)
    self.fc3 = nn.Linear(256, 256)
    self.action_mean = nn.Linear(256, out_dims)

    self.action_mean.weight.data.mul_(0.1)
    self.action_mean.bias.data.mul_(0.0)

    self.action_log_std = nn.Parameter(torch.ones(out_dims) * log_std)

    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(self.fc2(x))
    x = self.relu(self.fc3(x))
    action_mean = self.action_mean(x)

    action_mean[:4] = torch.sigmoid(action_mean[:4])
    action_mean[4:6] = self.tanh(action_mean[4:6])
    action_mean[6:] = torch.sigmoid(action_mean[6:])

    action_log_std = self.action_log_std.expand_as(action_mean)
    action_std = torch.exp(action_log_std)
    return action_mean, action_log_std, action_std

  def select_action(self, x):
    action_mean, _, action_std = self.forward(x)
    action = torch.normal(action_mean, action_std)
    return action

  def get_kl(self, x):
    mean1, log_std1, std1 = self.forward(x)

    mean0 = mean1.detach()
    log_std0 = log_std1.detach()
    std0 = std1.detach()
    #pdb.set_trace()
    kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
    return kl.sum(1, keepdim=True)

  def get_log_prob(self, x, actions):
    #pdb.set_trace()
    action_mean, action_log_std, action_std = self.forward(x)
    #print(normal_log_density(actions, action_mean, action_log_std, action_std))
    return normal_log_density(actions, action_mean, action_log_std, action_std)

  def get_fim(self, x):
    #pdb.set_trace()
    mean, _, _ = self.forward(x)
    #vec of len = No. of states*size of action e.g. cov_inv.shape = 2085*6
    cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0)) 
    param_count = 0
    std_index = 0
    id = 0
    for name, param in self.named_parameters():
      if name == "action_log_std":
        std_id = id
        std_index = param_count
      param_count += param.view(-1).shape[0]
      id += 1
    #pdb.set_trace()
    return cov_inv.detach(), mean, {'std_id': std_id, 'std_index': std_index}