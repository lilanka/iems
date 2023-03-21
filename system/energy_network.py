import pandapower as pp
import pandapower.networks as pn

class EnergyNetwork:
  def __init__(self, config):
    self.network = pn.create_cigre_network_mv(with_der="all")

    # initialize the energy network parameters
    self.network.sgen["p_mw"][9], self.network.sgen["q_mvar"][9] = config["residential_fuell_cell_1_p"], config["residential_fuell_cell_1_q"] 
    self.network.sgen["p_mw"][12], self.network.sgen["q_mvar"][12] = config["residential_fuell_cell_2_p"], config["residential_fuell_cell_2_q"] 
    self.network.sgen["p_mw"][10], self.network.sgen["q_mvar"][10] = config["chp_diesel_p"], config["chp_diesel_q"] 
    self.network.sgen["p_mw"][11], self.network.sgen["q_mvar"][11] = config["fuell_cell_p"], config["fuell_cell_q"] 
    self.network.sgen["sn_mva"][:7] = 0.020

  def run_energy_network(self, solar_power, wind_power):
    self._insert_solar_wind_powers(solar_power, wind_power)
    # run the system and get active/reactive powers
    pp.runpp(self.network)

  def get_node_voltages(self):
    #  from res_bus
    return self.network.res_bus["vm_pu"]

  def get_line_currents(self):
    # from res_line
    return self.network.res_line["i_ka"]

  def get_grid_powers(self):
    # from res_ext_grid
    return self.network.res_ext_grid["p_mw"][0], self.network.res_ext_grid["q_mvar"][0]

  def _insert_solar_wind_powers(self, solar_power, wind_power):
    self.network.sgen["p_mw"][:8] =  solar_power * self.network.sgen["sn_mva"][:8]
    self.network.sgen["p_mw"][8] =  wind_power * self.network.sgen["sn_mva"][8]