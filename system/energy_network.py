import pandapower as pp
import pandapower.networks as pn

class EnergyNetwork:
  def __init__(self, config):
    self.network = pn.create_cigre_network_mv(with_der="all")

    # initialize the energy network parameters
    """
    self.network.sgen["p_mw"][9], self.network.sgen["q_mvar"][9] = config["residential_fuell_cell_1_p"], config["residential_fuell_cell_1_q"] 
    self.network.sgen["p_mw"][12], self.network.sgen["q_mvar"][12] = config["residential_fuell_cell_2_p"], config["residential_fuell_cell_2_q"] 
    self.network.sgen["p_mw"][10], self.network.sgen["q_mvar"][10] = config["chp_diesel_p"], config["chp_diesel_q"] 
    self.network.sgen["p_mw"][11], self.network.sgen["q_mvar"][11] = config["fuell_cell_p"], config["fuell_cell_q"] 
    """
    self.network.sgen["sn_mva"][:6] = 0.020

    # remove unwanted branches of the network  
    self.network.switch["closed"][7] = False
    self.network.load["sn_mva"][8:10] = [0, 0]
    self.network.load["sn_mva"][15:18] = [0, 0, 0]

  def run_energy_network(self, solar_power, wind_power):
    self._insert_solar_wind_powers(solar_power, wind_power)
    # run the system and get active/reactive powers
    pp.runpp(self.network)

  def get_node_voltages(self):
    #  from res_bus
    return self.network.res_bus["vm_pu"][1:12]

  def get_line_currents(self):
    # from res_line
    net = self.network.res_line["i_ka"]
    net = net.drop([net.index[10], net.index[11], net.index[14]]).reset_index().iloc[:,1:]
    return net 

  def get_grid_powers(self):
    # from res_ext_grid
    return self.network.res_ext_grid["p_mw"][0], self.network.res_ext_grid["q_mvar"][0]

  def _insert_solar_wind_powers(self, solar_power, wind_power):
    self.network.sgen["p_mw"][:8] =  solar_power * self.network.sgen["sn_mva"][:8]
    self.network.sgen["p_mw"][8] =  wind_power * self.network.sgen["sn_mva"][8]