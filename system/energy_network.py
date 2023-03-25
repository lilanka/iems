import numpy as np
import pandapower as pp
import pandapower.networks as pn

class EnergyNetwork:
  """
  Microgrid model
  """
  def __init__(self, config):
    self.network = pn.create_cigre_network_mv(with_der="all")

    self.pf = None 
    self._init_pfs()

    # initialize the energy network parameters
    """
    self.network.sgen["p_mw"][9], self.network.sgen["q_mvar"][9] = config["residential_fuell_cell_1_p"], config["residential_fuell_cell_1_q"] 
    self.network.sgen["p_mw"][12], self.network.sgen["q_mvar"][12] = config["residential_fuell_cell_2_p"], config["residential_fuell_cell_2_q"] 
    self.network.sgen["p_mw"][10], self.network.sgen["q_mvar"][10] = config["chp_diesel_p"], config["chp_diesel_q"] 
    self.network.sgen["p_mw"][11], self.network.sgen["q_mvar"][11] = config["fuell_cell_p"], config["fuell_cell_q"] 
    """
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

  def run_energy_network(self, swd, p=None, q=None):
    # p, q (1x4) array
    self._insert_solar_wind_demand(swd, p, q)
    pp.runpp(self.network)

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

  def _insert_solar_wind_demand(self, swd, p, q):
    # update solar 
    self.network.sgen["p_mw"][:8] =  swd[0] * self.network.sgen["sn_mva"][:8]
    self.network.sgen["p_mw"][8] =  swd[1] * self.network.sgen["sn_mva"][8]

    # update demand
    self.network.load["p_mw"] = swd[2] * self.network.load["sn_mva"] * self.pf
    self.network.load["q_mvar"] = swd[2] * self.network.load["sn_mva"] * np.sin(np.arccos(self.pf))

    # update p, q
    if p is not None and q is not None:
      self.network.sgen["p_mw"][9:] = p
      self.network.sgen["q_mvar"][9:] = q

  def _init_pfs(self):
    # calculate fixed power factors of load
    self.pf = np.cos(np.arctan(self.network.load["q_mvar"] / self.network.load["p_mw"]))