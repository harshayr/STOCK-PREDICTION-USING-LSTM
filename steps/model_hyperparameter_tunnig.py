import numpy as np
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from scikeras.wrappers import KerasRegressor


from tensorflow.keras.optimizers import Adam

from dataclasses import dataclass
import os


from exception import CustomException
from logger import logging

@ dataclass
class ModelTrainConfig:
    train_model_path = os.path.join("artifacts","model.pkl")

class ModelTunning:
    def __init__(self):
        self.model_train_config = ModelTrainConfig()

    def create_model(units_lstm1, units_lstm2, units_lstm3, dropout_rate, lr):
        model = Sequential()
        model.add(LSTM(units_lstm1, return_sequences=True, input_shape=(30, 6)))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units_lstm2, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units_lstm3))
        model.add(Dense(1))
        
        # Compile the model
        optimizer = Adam(learning_rate=lr)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model
    # Create KerasRegressor
    def initiate_tunning(self,X_train, y_train):
        model = KerasRegressor(build_fn=self.create_model, verbose=1)

        # Define the hyperparameters to tune
        param_grid = {
            'units_lstm1': [21, 50, 100],
            'units_lstm2': [50, 70, 100],
            'units_lstm3': [30, 50, 70],
            'dropout_rate': [0.2, 0.3],
            'lr': [0.001, 0.0008],
            'batch_size': [15, 30],
            'epochs': [100, 120, 170]
        }

        # Create GridSearchCV
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

        # Perform GridSearchCV
        grid_result = grid_search.fit(X_train, y_train)

        # Print results
        print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

        return model.set_params(**gs.best_params_)

