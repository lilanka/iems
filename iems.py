#!/bin/python3

from controller import Controller
from process_data import prepare_data
from utils import read_json

def main():
  # get dataset
  train_data, test_data = prepare_data()
  print(train_data)

  # configs of the system
  config_path = "config.json"
  config = read_json(config_path)

  controller = Controller(config)

if __name__ == "__main__":
  main()