import numpy as np
import pandapower as pp
import pandapower.networks as pn

class EnergyNetwork:
  """
  Microgrid model
  """
  def __init__(self, config):
    self.config = config
    self.network = pn.create_cigre_network_mv(with_der="all")

    self.pf = None 
    self._init_pfs()

    # set solar capacities
    self.network.sgen["sn_mva"][:6] = 0.020
    
    # remove unwanted branches of the network  
    self.network.switch["closed"][7] = False
    self.network.load["sn_mva"][8:10] = [0, 0]
    self.network.load["sn_mva"][15:18] = [0, 0, 0]
    
    # initialize energy storage system
    self.network.storage["sn_mva"][0] = 0.6
    self.network.storage["soc_percent"][0] = config["battery1"]["soc"]
    self.network.storage["soc_percent"][1] = config["battery1"]["soc"]

  def update_soc(self, soc):
    self.network.storage["soc_percent"] = soc

  def run_energy_network(self, swd, pq=None, soc=None):
    # pq (1xn_actions): 
    #   0-4: p solar  
    #   4-6: p battery
    #   6-n_actions: q solar
    self._insert_solar_wind_demand_pq_soc(swd, pq, soc)
    pp.runpp(self.network, max_iteration=1000)

  def reset(self):
    # reset energy network
    self.network.sgen["p_mw"][9:] = 0 
    self.network.sgen["q_mvar"][9:] = 0
    self.network.storage["p_mw"] = self.network.storage["sn_mva"]
    self.network.storage["soc_percent"] = [self.config["battery1"]["soc"], self.config["battery2"]["soc"]]

  def get_dg_p(self):
    return self.network.sgen["p_mw"][9:]

  def get_dg_q(self):
    return self.network.sgen["q_mvar"][9:]

  def get_dg_s(self):
    return self.network.sgen["sn_mva"][9:]

  def get_node_voltages(self):
    #  from res_bus
    return self.network.res_bus["vm_pu"][1:12]

  def get_line_currents(self):
    # from res_line
    net = self.network.res_line["i_ka"]
    net = net.drop([net.index[10], net.index[11], net.index[14]]).reset_index().iloc[:,1:]
    return net["i_ka"] 

  def get_grid_powers(self):
    # from res_ext_grid
    return self.network.res_ext_grid["p_mw"][0], self.network.res_ext_grid["q_mvar"][0]

  def get_soc(self):
    return self.network.storage["soc_percent"]

  def get_pq(self):
    return np.concatenate((self.network.res_bus["p_mw"][:12], self.network.res_bus["q_mvar"][:12]))

  def _insert_solar_wind_demand_pq_soc(self, swd, pq, soc):
    
    # update solar 
    self.network.sgen["p_mw"][:8] =  swd[0] * self.network.sgen["sn_mva"][:8]
    self.network.sgen["p_mw"][8] =  swd[1] * self.network.sgen["sn_mva"][8]

    # update demand
    self.network.load["p_mw"] = swd[2] * self.network.load["sn_mva"] * self.pf
    self.network.load["q_mvar"] = swd[2] * self.network.load["sn_mva"] * np.sin(np.arccos(self.pf))

    if soc is not None:
      self.network.storage["soc_percent"] = soc

    # update p, q
    if pq is not None:
      self.network.sgen["p_mw"][9:] = pq[:4]
      self.network.storage["p_mw"] = pq[4:6]
      self.network.sgen["q_mvar"][9:] = pq[6:]

  def _init_pfs(self):
    # calculate fixed power factors of load
    self.pf = np.cos(np.arctan(self.network.load["q_mvar"] / self.network.load["p_mw"]))