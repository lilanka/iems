import json
import pandas as pd

def read_json(path):
  config = []
  with open(path) as f:
    config = json.load(f)
  return config

def read_csv(path):
  data = pd.read_csv(path)
  return data
