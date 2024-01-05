import numpy as np
import pandas as pd
import os
import sys
import pickle

from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler

from exception import CustomException
from logger import logging
from utils import save_file, create_dataset,save_scaler


@dataclass
class DataTransformConfig:
    preprocess_obj_file = os.path.join("artifacts", "preprocess.pkl")

class ScaleSplit:
    def __init__(self):
        self.datatransform_config = DataTransformConfig()
    def split_data(self,final_df):
        try:
            scaler=MinMaxScaler(feature_range=(0,1))
            df1=scaler.fit_transform(final_df)
            training_size=int(len(df1)*1.0)
            train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:]
            # save_scaler(self.datatransform_config.preprocess_obj_file, scaler)
            
            with open(self.datatransform_config.preprocess_obj_file, 'wb') as file:
                pickle.dump(scaler, file)
            logging.info("Scaler mode saved")
            return train_data,test_data
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, final_df):
        try:
            logging.info("Initiate data Transformtion")
            # final_df = pd.read_csv(final_df_path)
            # final_df = final_df.set_index('Date', drop = True)
            train_data,test_data = self.split_data(final_df)
            time_step = 30
            X_train, y_train = create_dataset(train_data, time_step)
            X_test, ytest = create_dataset(test_data, time_step)
            logging.info("Data Transformation Completed")
            return X_train, y_train, X_test, ytest
        
        except Exception as e:
            raise CustomException(e, sys)
        
        

        
