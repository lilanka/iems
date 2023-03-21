import json

def read_json(path):
  config = []
  with open(path) as f:
    config = json.load(f)
  return config