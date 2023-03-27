class Battery:
  """
  Battery system
  """
  def __init__(self, config):
    self.cap = config["capacity"]
    self.energy = (config["soc"] * self.cap) / 100 # MWh value
    self.energy_default = (config["soc"] * self.cap) / 100 # MWh value
    self.nc = config["charging_efficiency"] 
    self.nd = config["discharging_efficiency"] 
    self.dt = config["dt"]

  def get_next_soc(self, p, is_percentage=False):
    # p: battery chargning discharging power (MW)
    if is_percentage:
      p = (p * self.cap) / 100

    u = self._charging_state(p)
    self.energy += (abs(p) * self.dt * ((self.nc * self.nd + 1) * u - 1)) / self.nd 
    #print(f"battery {b}: action: {p}, self.energy: {self.energy}")
    return (self.energy / self.cap) * 100

  def reset(self):
    # reset battery
    self.energy = self.energy_default

  def _charging_state(self, p):
    return 1 if p > 0 else 0