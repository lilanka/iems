class Battery:
  """
  Battery system
  """
  def __init__(self, config):
    self.cap = config["capacity"]
    self.energy = (config["soc"] * self.cap) / 100 # MWh value
    self.nc = config["charging_efficiency"] 
    self.nd = config["discharging_efficiency"] 
    self.dt = config["dt"]

  def get_soc_value(self, p):
    # p: battery chargning discharging power (MW)
    u = self._charging_state(p)
    self.energy += (abs(p) * self.dt * ((self.nc * self.nd + 1) * u - 1)) / self.nd 
    return (self.energy / self.cap) * 100

  def _charging_state(self, p):
    return 1 if p > 0 else 0