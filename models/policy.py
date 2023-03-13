import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
  """ Policy network
  Args:
    in_dimensions: int - Input dimensions
    out_dimensions: int - Out dimensions
  """
  def __init__(self, in_dimensions, out_dimensions):
    super(PolicyNet, self).__init__()
    self.fc1 = nn.Linear(in_dimensions, 256)
    self.fc2 = nn.Linear(256, 256)
    self.fc3 = nn.Linear(256, 256)
    self.fc4 = nn.Linear(256, out_dimensions)

  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc4(x)
    return x