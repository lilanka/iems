import numpy as np

from system.battery import Battery
from system.energy_network import EnergyNetwork
from models.agent import Agent

class Controller:
  """
  Controller logic
  """
  def __init__(self, config):
    self.config = config

    # initialize components in the system 
    self.battery1 = Battery(config["battery1"]) 
    self.battery2 = Battery(config["battery2"]) 
    self.energy_network = EnergyNetwork(config)
    self.agent = Agent()

  def run_system(self, data):
    self.energy_network.run_energy_network(0.5, 0.5, 0.4)
    self._get_reward(0.7)

  def _get_reward(self, rate):
    cfg = self.config["cost_coeff"]
    p = self.energy_network.get_dg_p()
    grid_p, _ = self.energy_network.get_grid_powers()

    a_d = np.array([cfg["res_fuel_cell_1"]["a"], cfg["chp_diesel"]["a"], cfg["fuel_cell"]["a"], cfg["res_fuel_cell_2"]["a"]])       
    b_d = np.array([cfg["res_fuel_cell_1"]["b"], cfg["chp_diesel"]["b"], cfg["fuel_cell"]["b"], cfg["res_fuel_cell_2"]["b"]])       
    c_d = np.array([cfg["res_fuel_cell_1"]["c"], cfg["chp_diesel"]["c"], cfg["fuel_cell"]["c"], cfg["res_fuel_cell_2"]["c"]])

    cost_dg = np.sum(a_d * p**2 + b_d * p + c_d)

    energy_cost = rate * grid_p / 4 
    cost_grid = energy_cost if grid_p >= 0 else energy_cost * cfg["sell_beta"] 

    # note: at validation stage, r has to be modified with a penalty
    r = -(cost_dg + cost_grid) 
    return r