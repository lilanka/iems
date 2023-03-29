#!/bin/python3

import numpy as np

from utils import read_json
from controller import Controller
from process_data import prepare_data


def train(warmup, num_iterations, day, controller, training_data):
  done = False
  step = episode = 0
  s1 = None
  tau = training_data["p"].shape[0]
  while step < num_iterations:
    # todo: resetting if it's start of the episode
    if s1 is None or tau % day == 0:
      s1 = controller.get_observations(training_data["swd"][episode], training_data["p"][episode], is_reset=True)
    
    # pick actions
    if step <= warmup:
      # take random actions just to fill the memory buffer for certain period
      action = controller.random_action()
    else:
      action = controller.select_action(s1)

    # next observation 
    s2 = controller.get_observations(training_data["swd"][episode], training_data["p"][episode], action)
    # reward
    r = controller.get_reward(training_data["p"][episode]) 

    print(f"Iter {step}, S: {s1}, a: {action}, St: {s2}, r: {r}")

    # agent update policy
    controller.observe(r, s2, done)
    if step > warmup:
      controller.update_policy()

    step += 1
    episode += 1
    s1 = s2

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