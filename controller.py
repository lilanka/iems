from system.battery import Battery
from system.energy_network import EnergyNetwork
from models.agent import Agent

class Controller:
  """
  Controller logic
  """
  def __init__(self, config):
    # initialize components in the system 
    self.battery1 = Battery(config["battery1"]) 
    self.battery2 = Battery(config["battery2"]) 
    self.energy_network = EnergyNetwork(config)
    self.agent = Agent()

  def run_system(self, data):
    self.energy_network.run_energy_network(0.5, 0.5, 0.4)
