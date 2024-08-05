import os
import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from mlforecast import MLForecast
from window_ops.rolling import rolling_mean, rolling_max, rolling_min
from sklearn.preprocessing import LabelEncoder

#list all files ending with .csv
files = [f for f in os.listdir() if os.path.isfile(f)]
files = [f for f in files if str(f).endswith('.csv')]
files.sort()

#Read the csv files and drop unnecessary columns
def read_df(file):
    df = pd.read_csv(file, parse_dates=['DATE'])
    df = df.drop(columns=['Unnamed: 0', 'SERIES', 'PREV. CLOSE'])
    #df = df.set_index('DATE')
    return df

#set dynamic features and static features for MLForecast to forecast avaoiding static features
dynamic_features = ['OPEN', 'HIGH', 'LOW', 'LTP', 'VWAP', '52W H', '52W L', 'VOLUME', 'VALUE', 'NO OF TRADES']
static_features = ['SYMBOL']

#sklearn pipeline to combine two regression models for forecasting
models = [make_pipeline( RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)),
          XGBRegressor(random_state=0, n_estimators=100, n_jobs=-1)]

#model training function to fit on model and return the model and data
def model_training(data, time_col, target_col, static_features=static_features): 
    #If you have no static features, pass an empty list or MLForecast will consider that all your additional columns are static.
    model.fit(train, id_col = 'SYMBOL', time_col='ds', target_col='y', max_horizon=h, static_features=static_features)
    return model, data

#label encoder to encode static features
le = LabelEncoder()

#loop to run the data read, model training at bank level separately for each csv file   
for file in files:
    
    data = read_df(f'{file}')
    data = data.rename(columns={'CLOSE': 'y', 'DATE': 'ds'})
    data['SYMBOL'] = le.fit_transform(data.SYMBOL)
    train = data.loc[data['ds'] < '2024-06-01']
    valid = data.loc[data['ds'] >= '2024-06-01']
    h = 60
    print(f'{file} data read')
    model = MLForecast(models=models,
                       freq='D',
                       lags=[1,2,4],
                       lag_transforms={
                           1: [(rolling_mean, 4), (rolling_min, 4), (rolling_max, 4)], 
                       },
                       #date_features=['daily'],
                       num_threads=10)

    model, data = model_training(train, valid, h, static_features)
    print(f'{file} training complete')
    with open(f'./mlf_{file}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f'{file} model saved')