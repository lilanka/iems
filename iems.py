#!/bin/python3

import torch
import numpy as np

from utils import read_json
from controller import Controller
from process_data import prepare_data


def train(warmup, num_iterations, day, controller, training_data):
  step = episode = 0
  observations = None
  tau = training_data["p"].shape[0]
  while step < num_iterations:
    # todo: resetting if it's start of the episode
    if observations is None or tau % day == 0:
      observations = controller.get_observations(training_data["swd"][episode], training_data["p"][episode], is_reset=True)

    # pick actions
    if step <= warmup:
      # take random actions just to fill the memory buffer for certain period
      action = controller.random_action()
    else:
      action = controller.agent(torch.from_numpy(observations.astype(np.float32))).cpu().detach().numpy()

    # next observations
    observations2 = controller.get_observations(training_data["swd"][episode], training_data["p"][episode], action)
    # reward
    r = controller.get_reward(training_data["p"][episode]) 
    print(f"Iter {step}, S: {observations}, a: {action}, St: {observations2}, r: {r}")
    step += 1
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