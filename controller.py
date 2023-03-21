from battery import Battery
from energy_network import EnergyNetwork

class Controller:
  """
  Controller logic
  """
  def __init__(self, config):
    # initialize components in the system 
    self.battery = Battery(config["battery"]) 
    self.energy_network = EnergyNetwork()