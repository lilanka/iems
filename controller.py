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
    self.energy_network.run_energy_network(5, 5, 0.4)
    self._get_auxiliary_cost()

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
  
  def _get_auxiliary_cost(self):
    # equivalent penalty for CPO 
    node_voltages = self.energy_network.get_node_voltages()
    line_currents = self.energy_network.get_line_currents()
    grid_p, grid_q = self.energy_network.get_grid_powers()
    battery_soc = self.energy_network.get_soc()
    p, q, s = self.energy_network.get_dg_p(), self.energy_network.get_dg_q(), self.energy_network.get_dg_s() 

    aux_cost_voltage = np.sum(np.maximum(np.maximum(0, node_voltages - 1.05), 0.95 - node_voltages))
    aux_cost_current = np.sum(np.maximum(0, (line_currents / 0.145) - 1))

    s_grid = np.sqrt(grid_p**2 + grid_q**2)
    aux_cost_ext_grid = np.maximum(0, (s_grid / 25) - 1)
    aux_cost_battery = np.sum(np.maximum(np.maximum(0, battery_soc - 80), 20 - battery_soc))
    aux_cost_dg = np.sum(np.maximum(0, (np.sqrt(p**2 + q**2) / s) - 1))

    aux_cost = aux_cost_voltage + aux_cost_current + aux_cost_ext_grid + aux_cost_battery + aux_cost_dg
    return aux_cost