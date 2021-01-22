import sys
import os
import keras
from keras.models import Model
from keras.callbacks import EarlyStopping
import math
import warnings
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.models import load_model
from keras.utils.vis_utils import plot_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import  MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, SimpleRNN, GRU

def MAPE(y_true, y_pred):
    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape

def process_data(train, test, lags):

    scaler = MinMaxScaler(feature_range=(0, 1)).fit(train.reshape(-1, 1))
    flow1 = scaler.transform(train.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = scaler.transform(test.reshape(-1, 1)).reshape(1, -1)[0]
    
    train, test = [], []
    for i in range(lags, len(flow1)):
        train.append(flow1[i - lags: i + 1])
    for i in range(lags, len(flow2)):
        test.append(flow2[i - lags: i + 1])

    train = np.array(train)
    test = np.array(test)

    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]

    return X_train, y_train, X_test, y_test, scaler

def get_model(X_train, y_train, lag):
    model = tf.keras.Sequential()
    model.add(LSTM(4, input_shape=(1, lag)))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    hist=model.fit(X_train1, y_train, validation_split=0.2, epochs=100, batch_size=64, verbose=0)
    
    return model, hist

lag = 24
horizon = 24
df = pd.read_csv('../data/2018_1hourdata.csv')

cols = ['sites'] + list(map(str, list(range(1,lag + 1))))
df_mae = pd.DataFrame(columns=cols)
df_rmse = pd.DataFrame(columns=cols)
df_mse = pd.DataFrame(columns=cols)
df_r2 = pd.DataFrame(columns=cols)
df_mape = pd.DataFrame(columns=cols)

for site in df.columns[1:]:
    
    # Data preprocessing
    X = df[site].values
    size = int(len(X) * 0.8)
    train, test = X[0:size], X[size:len(X)]
    X_train, y_train, X_test, y_test, scaler = process_data(train, test, lag)
    X_train1 = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

    y = []
    i = 0
    while i + lag <= len(y_test):
        y_i = y_test[i: i + lag]
        y.append(y_i)
        i = i + 1
    y = np.stack(y, axis=0)

    # Model building and training
    model, history = get_model(X_train, y_train, lag)

    # Prediction
    list1 = list(map(str, list(range(1,horizon+1))))
    df_obs = pd.DataFrame(columns=list1)
    df_pred = pd.DataFrame(columns=list1)

    for i in range(X_test.shape[0] - horizon):    
        pred = []
        test = np.reshape(X_test[i, None], (1,1, lag))
        model.fit(test, y_test[i, None], batch_size=1, epochs=1, validation_split=0.0)
        for j in range(0,horizon):
            output = model.predict(test)
            test = np.reshape(np.append(test, output[0][0])[1:], (1, 1, lag))
            pred.append(output[0])
        yobs = y[i]
        df_pred = df_pred.append(dict( zip( list1, pred)), ignore_index=True)
        df_obs = df_obs.append(dict( zip( list1, yobs)), ignore_index=True)

    # Writing the results
    mae_list = []; rmse_list = []; mse_list = []; r2_list = []; mape_list = []
    for hor in range(24):
        yobs = df_obs.iloc[:,hor] 
        ypred = df_pred.iloc[:,hor]  
        mae_list.append(mean_absolute_error(yobs, ypred))
        rmse_list.append(np.sqrt(mean_squared_error(yobs, ypred)))
        mse_list.append(mean_squared_error(yobs, ypred))
        r2_list.append(r2_score(yobs, ypred))
        mape_list.append(MAPE(yobs, ypred)[0])
    mae_list = [site] + mae_list
    rmse_list = [site] + rmse_list
    mse_list = [site] + mse_list
    r2_list = [site] + r2_list
    mape_list = [site] + mape_list
    df_mae = df_mae.append(dict( zip( cols, mae_list)), ignore_index=True)
    df_rmse = df_rmse.append(dict( zip( cols, rmse_list)), ignore_index=True)
    df_mse = df_mse.append(dict( zip( cols, mse_list)), ignore_index=True)
    df_r2 = df_r2.append(dict( zip( cols, r2_list)), ignore_index=True)
    df_mape = df_mape.append(dict( zip( cols, mape_list)), ignore_index=True)
    

folder = 'Results/Results_vanillaLSTM_R'
if not os.path.exists(folder):
    os.makedirs(folder)   
df_mae.to_csv(folder + '/MAE_vanillaLSTM.csv')    
df_rmse.to_csv(folder + '/RMSE_vanillaLSTM.csv')   
df_mse.to_csv(folder + '/MSE_vanillaLSTM.csv')  
df_r2.to_csv(folder + '/R2_vanillaLSTM.csv') 
df_mape.to_csv(folder + '/MAPE_vanillaLSTM.csv')    
    