#!/bin/python3

import pdb

import numpy as np

from utils import read_json
from controller import Controller
from process_data import prepare_data

from torch.utils.tensorboard import SummaryWriter

# models available: cpo, ddpg
writer = SummaryWriter()

def train(warmup, num_iterations, day, controller, training_data, model, batch_size, buffer_size):
  s1 = None

  step, n_iter, cpo_iter = 0, 0, 0
  v_loss, p_loss, cost_loss = 0, 0, 0
  reward_per_episode = 0
  v_loss_list, p_loss_list, cost_loss_list, reward_list = [], [], [], []

  cpo_max_iter = buffer_size / batch_size

  while n_iter < num_iterations:
    done = False
    # TODO (Lilanka): resetting if it's start of the episode
    if step >= training_data["p"].shape[0]:
      step = 0
    if s1 is None or (step + 1) % day == 0:
      s1 = controller.get_observations(training_data["swd"][step], training_data["p"][step], is_reset=True)
      done = True
      reward_list.append(reward_per_episode)
      reward_per_episode = 0
    
    action = controller.select_action(s1)
    # next observation 
    s2 = controller.get_observations(training_data["swd"][step], training_data["p"][step], action)
    # reward
    r = controller.get_reward(training_data["p"][step]) 
    reward_per_episode += r
    # agent update policy
    controller.observe(r, s2, 0 if done else 1)

    if n_iter > buffer_size:
      if model == "cpo":
        if cpo_iter >= cpo_max_iter:
          controller.reset_buffer()
          cpo_iter = 0
        v_loss, p_loss, cost_loss = controller.update_policy()
        cpo_iter += 1
      writer.add_scalar("reward", r, step)

    print(f"Iter: {n_iter}, v_loss: {v_loss}, p_loss: {p_loss}, cost_loss: {cost_loss}, r: {r}")

    v_loss_list.append(v_loss)
    p_loss_list.append(p_loss)
    cost_loss_list.append(cost_loss)

    # tensorboard update
    #writer.add_scalar("v_loss", v_loss, step)
    #writer.add_scalar("p_loss", p_loss, step)
    #writer.add_scalar("cost_loss", cost_loss, step)
    #writer.add_scalar("reward", r, step)

    step += 1
    s1 = s2
    n_iter += 1

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
    train(config["warmup"], config["train_iter"], config["day"], controller, training_data, config["model"], 
          config["batch_size"], config["Memory"]["mem_size"])

if __name__ == "__main__":
  main()
