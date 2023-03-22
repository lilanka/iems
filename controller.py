from system.battery import Battery
from system.energy_network import EnergyNetwork

class Controller:
  """
  Controller logic
  """
  def __init__(self, config):
    # initialize components in the system 
    self.battery = Battery(config["battery"]) 
    self.energy_network = EnergyNetwork(config["energy_network"])

  def run_system(self, data):
    self.energy_network.run_energy_network(5, 10)
    #node_voltages = self.energy_network.get_node_voltages()
    #print(node_voltages)
    line_currents = self.energy_network.get_line_currents()
    #grid_powers_p, grid_powers_q = self.energy_network.get_grid_powers()