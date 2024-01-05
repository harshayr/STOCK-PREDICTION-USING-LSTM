### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Activation
from tensorflow.keras.layers import LSTM
from keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback, CSVLogger, EarlyStopping, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.metrics import RootMeanSquaredError, BinaryAccuracy
import pickle
from exception import CustomException
from logger import logging
import os
from dataclasses import dataclass
from utils import save_file
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error




@dataclass
class ModelConfig:
    train_model_file_path = os.path.join("artifacts", 'mag_model.h5')
    metrics_file_path = os.path.join("artifacts", 'metrics')

class ModelTrain:
    def __init__(self) -> None:
        self.train_model_file_path = ModelConfig()
    def model_train(self, X_train,y_train,X_test, ytest):

        # lstm_input = Input(shape=(30, 6), name='lstm_input')
        # inputs = LSTM(21, name='first_layer')(lstm_input)
        # inputs = Dense(16, name='first_dense_layer')(inputs)
        # inputs = Dense(1, name='dense_layer')(inputs)
        # output = Activation('linear', name='output')(inputs)

        # model = Model(inputs=lstm_input, outputs=output)
        # adam = optimizers.Adam(lr = 0.0008)

        # model.compile(optimizer=adam, loss='mse',metrics = 'accuracy')
        early_callback = EarlyStopping(monitor = 'val_loss', min_delta = 0, 
                                                   patience = 10, verbose = 1, mode = 'auto',baseline = None,
                                                 restore_best_weights = False )
        logging.info("Model training Started")
        model=Sequential()
        model.add(LSTM(50,return_sequences=True,input_shape=(30,5)))
        model.add(Dropout(0.2))
        model.add(LSTM(50,return_sequences=False))
        model.add(Dropout(0.2))
        # model.add(LSTM(50))
        model.add(Dense(16))
        model.add(Dense(1,activation = 'linear'))
        adam = optimizers.Adam(lr = 0.0008)
        model.compile(loss='mean_squared_error',optimizer=adam)
        history = model.fit(x=X_train, y=y_train, batch_size=15, epochs=100, shuffle=True, validation_split = 0.1,callbacks = [early_callback])
        logging.info('Model training completed')
        logging.info("Model saved succesfully")
        save_file(
                file_path=self.train_model_file_path.train_model_file_path,
                obj=model
                )
        
        
        df = pd.DataFrame(history.history)
        df.to_csv(self.train_model_file_path.metrics_file_path)

        ### Lets Do the prediction and check performance metrics
        train_predict=model.predict(X_train)
        # test_predict=model.predict(X_test)
        y_train = y_train.reshape([-1,1])
        # ytest = ytest.reshape([-1,1])

        tr_pred = np.repeat(train_predict, 5, axis = -1)
        # ts_pred = np.repeat(test_predict, 6, axis = -1)  # these are just a same things copied five times

        tr_y_actual = np.repeat(y_train, 5, axis = -1)
        # ts_y_actual = np.repeat(ytest, 6, axis = -1)

        # scaler = load_model('/Users/harshalrajput/Desktop/Projects/Stock_prediction/artifacts/preprocess.pkl')
        with open('/Users/harshalrajput/Desktop/Projects/Stock_prediction/artifacts/preprocess.pkl', 'rb') as file:
            scaler = pickle.load(file)

        ##Transformback to original form
        train_predict=scaler.inverse_transform(tr_pred)[:,0]
        # test_predict=scaler.inverse_transform(ts_pred)[:,0]
        y_train_actual = scaler.inverse_transform(tr_y_actual)[:,0]
        # y_test_actual = scaler.inverse_transform(ts_y_actual)[:,0]

        # actual_test = df_tesla["Open"][2210:]

        # result_test = pd.DataFrame({
        #     'actual': y_test_actual,
        #     'predicted': test_predict
        # })

        result_train = pd.DataFrame({
            'actual': y_train_actual,
            'predicted': train_predict
        })

        # train_mse = math.sqrt(mean_squared_error(y_train,train_predict))
        # test_mse = math.sqrt(mean_squared_error(ytest,test_predict))



        return df, result_train




