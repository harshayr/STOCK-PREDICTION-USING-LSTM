import pandas as pd
import sys
import datetime
import numpy as np
import yfinance as yf
import seaborn as sns

from logger import logging
from exception import CustomException
from logger import logging
from dataclasses import dataclass
import os
from utils import on_balance_volume_creation, indicators, set_index


# df = yf.download('TSLA')

@dataclass  # defifnig variable without using constructor
class DataIngestConfig:
    # train_path = os.path.join("artifacts", "train.csv")
    # test_path = os.path.join("artifacts",'test.csv')
    raw_path = os.path.join("artifacts", "raw.csv")


class IngestData:
    def __init__(self):
        self.ingest_config = DataIngestConfig()
        # self.scale_split = ScaleSplit()

    def initiate_data_ingestion(self,compony_name):
        logging.info("Data ingestion initiated")
        try:
            
            df = yf.download(compony_name)
            os.makedirs(os.path.dirname(self.ingest_config.raw_path), exist_ok=True)
            new_df = pd.DataFrame()
            new_df['Adj Close'] = df['Adj Close']

            # new_balance_volume = [0]
            # tally = 0
            # for i in range(1, len(df)):
            #     if (df['Adj Close'][i] > df['Adj Close'][i - 1]):
            #         tally += df['Volume'][i]
            #     elif (df['Adj Close'][i] < df['Adj Close'][i - 1]):
            #         tally -= df['Volume'][i]
            #     new_balance_volume.append(tally)
            
            # new_df = on_balance_volume_creation(df)
            x = indicators(df,new_df)

            final_df = set_index(x,df)
            final_df.to_csv(self.ingest_config.raw_path)

            # train_data,test_data = self.scale_split .split_data(final_df)

            logging.info("Data ingestion completed")

            return final_df

        except Exception as e:
            raise CustomException(e, sys)