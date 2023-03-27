import numpy as np
import torch

from system.battery import Battery
from system.energy_network import EnergyNetwork
from models.model import Model 
from memory import SequentialMemory
from utils import *

def define_action_space(cfg):
  # define action space
  p_rfc1 = np.arange(cfg["action_space"]["res_fuel_cell_1"]['p_min'], cfg["action_space"]["res_fuel_cell_1"]['p_max'] + cfg["action_space"]["res_fuel_cell_1"]['step_size'], cfg["action_space"]["res_fuel_cell_1"]['step_size'])
  p_chp = np.arange(cfg["action_space"]["chp_diesel"]['p_min'], cfg["action_space"]["chp_diesel"]['p_max'] + cfg["action_space"]["chp_diesel"]['step_size'], cfg["action_space"]["chp_diesel"]['step_size'])
  p_fc = np.arange(cfg["action_space"]["fuel_cell"]['p_min'], cfg["action_space"]["fuel_cell"]['p_max'] + cfg["action_space"]["fuel_cell"]['step_size'], cfg["action_space"]["fuel_cell"]['step_size'])
  p_rfc2 = np.arange(cfg["action_space"]["res_fuel_cell_2"]['p_min'], cfg["action_space"]["res_fuel_cell_2"]['p_max'] + cfg["action_space"]["res_fuel_cell_2"]['step_size'], cfg["action_space"]["res_fuel_cell_2"]['step_size'])
  p_bat1 = np.arange(cfg["action_space"]["battery_1"]["p_min"], cfg["action_space"]["battery_1"]["p_max"] + cfg["action_space"]["battery_1"]['step_size'], cfg["action_space"]["battery_1"]['step_size'])
  p_bat2 = np.arange(cfg["action_space"]["battery_2"]["p_min"], cfg["action_space"]["battery_2"]["p_max"] + cfg["action_space"]["battery_2"]['step_size'], cfg["action_space"]["battery_2"]['step_size'])
  q_rfc1 = np.arange(cfg["action_space"]["res_fuel_cell_1"]['q_min'], cfg["action_space"]["res_fuel_cell_1"]['q_max'] + cfg["action_space"]["res_fuel_cell_1"]['step_size'], cfg["action_space"]["res_fuel_cell_1"]['step_size'])
  q_chp = np.arange(cfg["action_space"]["chp_diesel"]['q_min'], cfg["action_space"]["chp_diesel"]['q_max'] + cfg["action_space"]["chp_diesel"]['step_size'], cfg["action_space"]["chp_diesel"]['step_size'])
  q_fc = np.arange(cfg["action_space"]["fuel_cell"]['q_min'], cfg["action_space"]["fuel_cell"]['q_max'] + cfg["action_space"]["fuel_cell"]['step_size'], cfg["action_space"]["fuel_cell"]['step_size'])
  q_rfc2 = np.arange(cfg["action_space"]["res_fuel_cell_2"]['q_min'], cfg["action_space"]["res_fuel_cell_2"]['q_max'] + cfg["action_space"]["res_fuel_cell_2"]['step_size'], cfg["action_space"]["res_fuel_cell_2"]['step_size'])

  for i in range(len(p_bat1)):
    if abs(p_bat1[i]) < 1e-10:
      p_bat1[i] = 0

  for i in range(len(p_bat2)):
    if abs(p_bat2[i]) < 1e-10:
      p_bat2[i] = 0

  return [q_rfc2, p_chp, p_fc, p_rfc2, p_bat1, p_bat2, q_rfc1, q_chp, q_fc, q_rfc2]

def define_cost_coefficients(cfg):
  a_d = np.array([cfg["res_fuel_cell_1"]["a"], cfg["chp_diesel"]["a"], cfg["fuel_cell"]["a"], cfg["res_fuel_cell_2"]["a"]])       
  b_d = np.array([cfg["res_fuel_cell_1"]["b"], cfg["chp_diesel"]["b"], cfg["fuel_cell"]["b"], cfg["res_fuel_cell_2"]["b"]])       
  c_d = np.array([cfg["res_fuel_cell_1"]["c"], cfg["chp_diesel"]["c"], cfg["fuel_cell"]["c"], cfg["res_fuel_cell_2"]["c"]])
  return a_d, b_d, c_d

class Controller:
  """
  Controller logic
  """
  def __init__(self, config, is_training=True):
    self.is_training = is_training
    self.config = config
    #self.device = "cuda" if torch.cuda.is_available() else "cpu"
    # todo: find a way to put things on gpu
    self.device = "cpu"

    # initialize action observation space 
    self.n_actions = config["action_space"]["n_actions"]
    self.n_obs = config["obs_space"]["n_obs"]
    self.action_space = define_action_space(config)

    # initialize cost coefficients
    self.a_d, self.b_d, self.c_d = define_cost_coefficients(config["cost_coeff"])

    # initialize the system 
    # microgrid
    self.battery1 = Battery(config["battery1"]) 
    self.battery2 = Battery(config["battery2"]) 
    self.energy_network = EnergyNetwork(config)

    # agent network
    self.agent = Model(self.n_obs, self.n_actions).to(self.device)

    # replay buffer
    self.memory = SequentialMemory(limit=config["Memory"]["mem_size"], window_length=config["Memory"]["window_length"])
    self.s1 = self.a1 = None # most recent state and action

  def select_action(self, obs):
    # todo: find a way to select actions from action space
    action = to_numpy(self.agent(to_tensor(obs))) # wrong
    self.a1 = action
    return action

  def observe(self, r, obs2, done):
    if self.is_training:
      self.memory.append(self.s1, self.a1, r, done) 
      self.s1 = obs2

  def update_policy(self):
    s1_b, a1_b, r_b, s2_b, t_b = self.memory.sample_and_split(self.config["batch_size"])
    print(s1_b)

  def get_observations(self, swd, price, action=None, is_reset=False):
    if is_reset:
      self._reset_system() 
      soc = self.energy_network.get_soc()
      battery1_soc, battery2_soc = self.battery1.get_next_soc(soc[0], is_percentage=True), self.battery2.get_next_soc(soc[1], is_percentage=True)
    else:
      battery1_soc, battery2_soc = self.battery1.get_next_soc(action[4]), self.battery2.get_next_soc(action[5])

    self.energy_network.run_energy_network(swd, action, [battery1_soc, battery2_soc])
    # update soc for next power flow
    self.energy_network.update_soc([battery1_soc, battery2_soc])
    pq = self.energy_network.get_pq()
    return np.concatenate((pq, [battery1_soc, battery2_soc, price]))

  def random_action(self):
    action = []
    for i in range(self.n_actions):
      action.append(np.random.choice(self.action_space[i]))
    self.a1 = action
    return action
  
  def run_system(self, data):
    self.energy_network.run_energy_network(data)

  def get_reward(self, price):
    p = self.energy_network.get_dg_p()
    grid_p, _ = self.energy_network.get_grid_powers()

    cost_dg = np.sum(self.a_d * p**2 + self.b_d * p + self.c_d)

    energy_cost = price * grid_p / 4 
    cost_grid = energy_cost if grid_p >= 0 else energy_cost * self.config["cost_coeff"]["sell_beta"] 

    # note: at validation stage, r has to be modified with a penalty
    r = -(cost_dg + cost_grid) 
    return r

  def _reset_system(self):
    self.energy_network.reset()
    self.battery1.reset()
    self.battery2.reset()
  
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