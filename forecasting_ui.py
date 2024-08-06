import streamlit as st
import streamlit.components.v1 as components
import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Forecasting Indian Banks SharePrice")

st.write("We are using MLForecast ensemble models (https://nixtlaverse.nixtla.io/mlforecast/index.html) to do prediction of indian bank stocks")
st.write("Training data used -  2000-01-01 to 2024-06-01")
st.write("Validation data used - 2024-06-01 till 2024-07-30")

files = [f for f in os.listdir() if os.path.isfile(f)]
files = [f for f in files if str(f).endswith('.csv')]
files.sort()

def read_df(file):
    df = pd.read_csv(file, parse_dates=['DATE'])
    df = df.drop(columns=['Unnamed: 0', 'SERIES', 'PREV. CLOSE'])
    #df = df.set_index('DATE')
    return df

# Create a select box
option = st.selectbox(
    'Choose an option:',
    ('AXISBANK','BANKBARODA','ICICIBANK', 'KOTAKBANK', 'SBIN', 'PNB','HDFC', 'PSB', 'IDFCFIRSTB')
)

data = read_df(f'{option}_df.csv')

dynamic_features = ['OPEN', 'HIGH', 'LOW', 'LTP', 'VWAP', '52W H', '52W L', 'VOLUME', 'VALUE', 'NO OF TRADES']
static_features = ['SYMBOL']

data = data.rename(columns={'CLOSE': 'y', 'DATE': 'ds'})
#train = df.loc[df['ds'] < '2024-06-01']
valid = data.loc[data['ds'] >= '2024-06-01']

h = st.selectbox(
    'Choose the days to forecast:',
    ('20', '30', '45', '60' )
)
st.write("NOTE: The higher the value for days selected, lesser the accuracy of predictions...")

h = int(h)

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from mlforecast import MLForecast


st.title(f"{option}")
st.write("The values in the graph indicate the real world values used in validation dataset..")
newvalid = valid.set_index('ds')
st.line_chart(newvalid[['y']])


mfiles = [f for f in os.listdir() if os.path.isfile(f)]
mfiles = [f for f in files if str(f).endswith('.pkl')]
mfiles.sort()

if option=='AXISBANK':
    model_file='mlf_AXISBANK_df.csv.pkl'
elif option=='BANKBARODA':
    model_file='mlf_BANKBARODA_df.csv.pkl'
elif option=='ICICIBANK':
    model_file='mlf_ICICIBANK_df.csv.pkl'
elif option=='KOTAKBANK':
    model_file='mlf_KOTAKBANK_df.csv.pkl'
elif option=='SBIN':
    model_file='mlf_SBIN_df.csv.pkl'
# elif option=='HDFC':
#     model_file='mlf_HDFC_df.csv.pkl'
elif option=='PNB':
    model_file='mlf_PNB_df.csv.pkl'
elif option=='PSB':
    model_file='mlf_PSB_df.csv.pkl'
else:
    model_file='mlf_IDFCFIRSTB_df.csv.pkl'

from mlforecast import MLForecast
import pickle

# load the model from disk
with open(model_file, 'rb') as f:
    model = pickle.load(f)

predictions = model.predict(h=h, new_df= valid)
st.subheader("Here are the predictions for selected value..")
st.table(predictions)

st.write('Graph for prediction of selected bank:')
newpreds = predictions.set_index('ds')
st.line_chart(newpreds[['RandomForestRegressor','XGBRegressor']])
