import pandas as pd

def read_csv(path):
  data = pd.read_csv(path)
  #print(data.iloc[0])
  return data

def prepare_data():
  training_data_path = "datasets/full datasets/"
  testing_data_path = "datasets/testing"

  # datasets
  demand_train_data = read_csv(f"{training_data_path}/Demand_train_full.csv").iloc[:,1:]
  price_train_data = read_csv(f"{training_data_path}/Price_train_full.csv").iloc[:,1:]
  solar_train_data = read_csv(f"{training_data_path}/Solar_train_full.csv").iloc[:,1:]
  wind_train_data = read_csv(f"{training_data_path}/Wind_train_full.csv").iloc[:,1:]

  return {"demand": demand_train_data, "price": price_train_data, "solar": solar_train_data, "wind": wind_train_data}, 0 
