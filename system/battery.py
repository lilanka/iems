class Battery:
  """
  Battery system
  """
  def __init__(self, config):
    self.soc = config["soc"]
    self.nc = config["charging_efficiency"] 
    self.nd = config["discharging_efficiency"] 
    self.dt = config["dt"]

  def get_soc_value(self, p):
    # p: battery chargning discharging power 
    p = abs(p)
    u = self._charging_state(p)
    # todo: check if the equation is correct
    self.soc += (p * self.dt * ((self.nc * self.nd + 1) * u - 1)) / self.nd 
    return self.soc

  def _charging_state(self, p):
    return 1 if p > 0 else 0