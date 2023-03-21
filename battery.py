class Battery:
  """
  Args:
    soc: state of charge of battery
    ch_eff: charging efficiency 
    disch_eff: discharging efficiency
    dt: time duration
  """
  def __init__(self, soc, ch_eff, disch_eff, dt=1):
    self.soc = soc 
    self.nc = ch_eff
    self.nd = disch_eff
    self.dt = dt

  def get_soc_value(self, p):
    # p: battery chargning discharging power 
    p = abs(p)
    u = self._charging_state(p)
    self.e_prev += (p * self.dt (self.nc * self.nd + 1) * u - 1) / self.nd
    return self.e_prev

  def _charging_state(self, p):
    return 1 if p > 0 else 0