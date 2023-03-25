#!/bin/python3

from controller import Controller
from process_data import prepare_data
from utils import read_json

def train(warmup, num_iterations, day, controller, training_data):
  step = episode = 0
  observations = None
  tau = training_data["p"].shape[0]
  #observations = controller.reset_system(trining_data[""])
  while step < num_iterations:
    # todo: resetting if it's start of the episode
    if observations is None or tau % day == 0:
      observations = controller.reset_system(training_data["swd"][episode])
    print(observations)
    break

    """
    # pick actions
    if step <= warmup:
      # take random actions just to fill the memory buffer for certain period
      action = controller.random_action()
    else:
      action = controller.agent()
    """
    episode += 1

def main():
  # don't need to load testing and testing data on memory
  is_training = True

  # get dataset
  training_data = prepare_data(is_training)

  # configs of the system
  config_path = "config.json"
  config = read_json(config_path)

  controller = Controller(config)

  if is_training:
    train(config["warmup"], config["train_iter"], config["day"], controller, training_data)
  #controller.run_system(train_data)

if __name__ == "__main__":
  main()