import numpy as np
from utils import read_csv

def prepare_data(is_training=True):
  if is_training: 
    training_data_path = "datasets/full datasets/"

    # datasets
    demand_train_data = read_csv(f"{training_data_path}/Demand_train_full.csv").iloc[:,1:]
    price_train_data = read_csv(f"{training_data_path}/Price_train_full.csv").iloc[:,1:]
    solar_train_data = read_csv(f"{training_data_path}/Solar_train_full.csv").iloc[:,1:]
    wind_train_data = read_csv(f"{training_data_path}/Wind_train_full.csv").iloc[:,1:]

    # return a = [solar, wind, demand], b = [price]
    return {"swd": np.random.rand(70177, 3), "p": np.random.rand(70171, 1)}

    #return {"demand": demand_train_data, "price": price_train_data, "solar": solar_train_data, "wind": wind_train_data}

  testing_data_path = "datasets/testing"