import pandas as pd
from finta import TA
from exception import CustomException
from logger import logging
import os
import numpy as np
import pickle
import sys

from tensorflow.keras.models import load_model

# data transformation
def save_file(file_path, obj):
    try:

        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        obj.save(file_path)

        # with open (file_path, "wb") as file_obj:
        #     dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException
    
def save_scaler(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        raise CustomException
    
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step),0:7]  ###i=0, 0,1,2,3-----99   100
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)


# data ingestion
def on_balance_volume_creation(stock_df):
    # Adding of on balance volume to dataframe
    
    new_df = pd.DataFrame({})

    new_df = stock_df[['Adj Close']].copy()


    new_balance_volume = [0]
    tally = 0

    #Adding the volume if the 
    for i in range(1, len(new_df)):
        if (stock_df['Adj Close'][i] > stock_df['Adj Close'][i - 1]):
            tally += stock_df['Volume'][i]
        elif (stock_df['Adj Close'][i] < stock_df['Adj Close'][i - 1]):
            tally -= stock_df['Volume'][i]
        new_balance_volume.append(tally)

    new_df['On_Balance_Volume'] = new_balance_volume
    minimum = min(new_df['On_Balance_Volume'])

    new_df['On_Balance_Volume'] = new_df['On_Balance_Volume'] - minimum
    new_df['On_Balance_Volume'] = (new_df['On_Balance_Volume']+1).transform(np.log)

    return new_df

def nan(new_df):
    for i in range(19):
        new_df['BB_MIDDLE'][i] = new_df.loc[i, 'Exponential_moving_average']

        if i != 0:
            higher = new_df.loc[i, 'BB_MIDDLE'] + 2 * new_df['Adj Close'].rolling(i + 1).std()[i]
            lower = new_df.loc[i, 'BB_MIDDLE'] - 2 * new_df['Adj Close'].rolling(i + 1).std()[i]
            new_df['BB_UPPER'][i] = higher
            new_df['BB_LOWER'][i] = lower
        else:
            new_df['BB_UPPER'][i] = new_df.loc[i, 'BB_MIDDLE']
            new_df['BB_LOWER'][i] = new_df.loc[i, 'BB_MIDDLE']
    return new_df

def indicators(df, new_df):
    edited_df = pd.DataFrame()
    #edited_df is made in order to generate the order needed for the finta library
    edited_df['open'] = df['Open']
    edited_df['high'] = df['High']
    edited_df['low'] = df['Low']
    edited_df['close'] = df['Close']
    edited_df['volume'] = df['Volume']
    edited_df.head()

    ema = TA.EMA(edited_df)
    bb = TA.BBANDS(edited_df)
    new_df['Exponential_moving_average'] = ema
    z = pd.concat([new_df,bb], axis = 1)
    new_df_reset = z.reset_index(drop=True)
    x = nan(new_df_reset)
    return x

def set_index(x, df):
    data = pd.DataFrame()
    df = df.reset_index()
    data['Date'] = df["Date"]
    dx = pd.concat([data, x], axis= 1)
    dx = dx.set_index('Date', drop = True)
    # dx.drop('Date',axis = 1, inplace = True)
    return dx

def load_obj(file_path):
    try:
        logging.info("Model loading started")
        with open(file_path, "rb")as file_obj:
            loaded_model = pickle.load(file_obj)
            return loaded_model
        logging.info("Model load succesfuly")
    except Exception as e:
        raise CustomException(e,sys)
    

# prediction
def pred(dl,forcast_days):
    data = np.array(dl.tail(30))
    result = []
    cols = ['Adj Close',
     'Exponential_moving_average',
     'BB_UPPER',
     'BB_MIDDLE',
     'BB_LOWER']
    with open('/Users/harshalrajput/Desktop/Projects/Stock_prediction/artifacts/preprocess.pkl', 'rb') as file:
            scaler = pickle.load(file)
    model = load_model('/Users/harshalrajput/Desktop/Projects/Stock_prediction/artifacts/mag_model.h5')
    
    temp = []
    i = 0
    while (i<forcast_days):
        if (len(temp)>30):
            data = np.array(temp[-30:])
    #         print(x_input.shape)
            x_input = scaler.transform(np.array(temp[-30:]))
            x_input_1 = x_input.reshape(1,30,5)

            y_pred = model.predict(x_input_1)
            tr_pred = np.repeat(y_pred, 5, axis = -1)
            y_pred_new = scaler.inverse_transform(tr_pred)[:,0]
            result.append(y_pred_new[0])
            
            #calculate ems
            last_ema = data[-2][1]
            close_price = data[-1][0]
            ema = (close_price*0.0952) + last_ema*(1-0.0952)
      
            df = pd.DataFrame(data, columns=cols)

            # calculate boillinger bands
            middle = df["Adj Close"].iloc[-20:].sum()/20
            higher = middle + 2 * list(df['Adj Close'].rolling(20).std())[-1]
            lower = middle - 2 * list(df['Adj Close'].rolling(20).std())[-1]

            l = np.array([y_pred_new[0],ema,higher, middle, lower]).reshape(1, -1)
            print('****')
            print(l)
            print('****')

            temp.append(l[0])
            temp = temp[1:]
            i = i+1


        else:
            x_input = scaler.transform(data)
            x_input_1 = x_input.reshape(1,30,5)
            y_pred = model.predict(x_input_1)
    #         print(y_pred)
            tr_pred = np.repeat(y_pred, 5, axis = -1)

            y_pred_new = scaler.inverse_transform(tr_pred)[:,0]
            result.append(y_pred_new[0])

            # calculate ema
            last_ema = data[-2][1]
            close_price = data[-1][0]
            ema = (close_price*0.0952) + (last_ema*(1-0.0952))
            
            # calculate boillinger band
            df = pd.DataFrame(data, columns=cols)
            middle = df["Adj Close"].iloc[-20:].sum()/20
            higher = middle + 2 * list(df['Adj Close'].rolling(20).std())[-1]
            lower = middle - 2 * list(df['Adj Close'].rolling(20).std())[-1]

            l = np.array([y_pred_new[0],ema,higher, middle, lower]).reshape(1, -1)
            print('****')
            print(l)
            print('****')

            temp.extend(data)
            temp.append(l[0])
            temp = temp[1:]
            i = i+1
            
    forcast_date = pd.date_range(list(dl.index)[-1],periods = forcast_days, freq = '1d').tolist()
    forcast_dates = []
    for i in forcast_date:
        forcast_dates.append(i.date())
    pred_df = pd.DataFrame({'Date':np.array(forcast_dates), 'Adj Close':result})
    orignal = dl.tail(100)
    orignal = orignal.reset_index()
            
            
    return result, pred_df, orignal
    
    

    

