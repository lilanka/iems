import numpy as np
import pandas as pd
from utils import read_csv

def prepare_data(is_training=True):
  # todo: simplify data processing. add automated operations instead of hand coding some parameters

  if is_training: 
    training_data_path = "datasets/full datasets/"

    # datasets
    demand_train_data = read_csv(f"{training_data_path}/Demand_train_full.csv").iloc[:,1:]
    price_train_data = read_csv(f"{training_data_path}/Price_train_full.csv").iloc[:,1:]
    solar_train_data = read_csv(f"{training_data_path}/Solar_train_full.csv").iloc[:,1:]
    wind_train_data = read_csv(f"{training_data_path}/Wind_train_full.csv").iloc[:,1:]

    demand_temp = []
    solar_temp = []
    wind_temp = []
    price_temp = []

    for i in range(731):
      demand_temp.append(demand_train_data.iloc[:,i])
      solar_temp.append(solar_train_data.iloc[:,i])
      wind_temp.append(wind_train_data.iloc[:,i])
      price_temp.append(price_train_data.iloc[:,i])

    demand_train_data = pd.concat(demand_temp, ignore_index=True)
    solar_train_data = pd.concat(solar_temp, ignore_index=True)
    wind_train_data = pd.concat(wind_temp, ignore_index=True)
    price_train_data = pd.concat(price_temp, ignore_index=True)

    swd = [solar_train_data, wind_train_data, demand_train_data]
    swd_train_data = pd.concat(swd, axis=1, ignore_index=True)

    swd_dict_values = []
    for i in range(0,70176):
      swd_dict_values.append(swd_train_data.iloc[i,:])

    swd_price_dict = {"swd": np.array(swd_dict_values), "p": np.array(price_train_data)}

    # return a = [solar, wind, demand], b = [price]
    return swd_price_dict 

    #return {"demand": demand_train_data, "price": price_train_data, "solar": solar_train_data, "wind": wind_train_data}

  testing_data_path = "datasets/testing"