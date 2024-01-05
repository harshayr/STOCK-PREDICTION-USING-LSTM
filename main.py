import streamlit as st
from steps.data_ingestion import IngestData
import pandas as pd
import sys
import datetime
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

from logger import logging
from exception import CustomException
from logger import logging
from dataclasses import dataclass
import os
from utils import on_balance_volume_creation, indicators, set_index,load_obj,pred

from steps.model_train import ModelTrain
from steps.data_ingestion import IngestData
from steps.data_transformatin import ScaleSplit

companies_dict = {
    'AAPL': 'Apple',
    'GOOGL': 'Alphabet',
    'MSFT': 'Microsoft Corporation',
    'AMZN': 'Amazon',
    'FB': 'Facebook',
    'BRK-A': 'Berkshire Hathaway',  # Note: BRK-A is for Class A shares
    'BABA': 'Alibaba Group',
    'JNJ': 'Johnson & Johnson',
    'JPM': 'JPMorgan Chase & Co.',
    'XOM': 'ExxonMobil',
    'BAC': 'Bank of America',
    'WMT': 'Wal-Mart Stores Inc.',
    'WFC': 'Wells Fargo & Co.',
    'RDS-A': 'Royal Dutch Shell plc',  # Note: RDS-A is for Class A shares
    'V': 'Visa Inc.',
    'PG': 'Procter & Gamble Co.',
    'BUD': 'Anheuser-Busch Inbev',
    'T': 'AT&T Inc.',
    'CVX': 'Chevron Corporation',
    'UNH': 'UnitedHealth Group Inc.'
}
def get_key_from_value(dictionary, search_value):
    for key, value in dictionary.items():
        if value == search_value:
            return key
    return None

st.sidebar.title("Stock chart predictor")
selected_user = st.sidebar.selectbox("Select compony", sorted(list(companies_dict.values())))

if selected_user is not None:
    compony_name = get_key_from_value(companies_dict,selected_user)
    # df = initiate_data_ingestion(compony_name)
    obj  = IngestData()
    df = obj.initiate_data_ingestion(compony_name)
    st.dataframe(df)
    col1, col2 = st.columns(2)
    with col1:
        st.title('Start date')
        st.header(df.index[0].date())
    with col2:
        st.title('Last date')
        st.header(df.index[-1].date())

    if st.sidebar.button("Train Model"):
        obj1 = ScaleSplit()
        obj2 = ModelTrain()
        X_train, y_train, X_test, ytest = obj1.initiate_data_transformation(df)
        history ,result_train= obj2.model_train(X_train,y_train,X_test, ytest)


        col2= st.columns(1) 
        with col2[0]:
            st.header('Losses')
            st.line_chart(history) 

        col3= st.columns(1) 
        with col3[0]:
            st.header('Train data pred')
            st.line_chart(result_train)
        
        col4 = st.columns(1)
        with col4[0]:
            st.header('30 Days Prediction')
            result, pred_df, orignal = pred(df,30)
            plt.figure(figsize=(14, 7))

            sns.lineplot(x='Date', y='Adj Close', data=orignal, label='Original Data')
            sns.lineplot(x='Date', y='Adj Close', data=pred_df, label='Forecast Data')

            plt.xlabel('Date')
            plt.ylabel('Adj Close')
            plt.legend()

            st.pyplot(plt)

        col5 = st.columns(1)
        with col5[0]:
            st.header('60 Days Prediction')
            result, pred_df, orignal = pred(df,60)
            plt.figure(figsize=(14, 7))

            sns.lineplot(x='Date', y='Adj Close', data=orignal, label='Original Data')
            sns.lineplot(x='Date', y='Adj Close', data=pred_df, label='Forecast Data')

            plt.xlabel('Date')
            plt.ylabel('Adj Close')
            plt.legend()

            st.pyplot(plt)


        col6 = st.columns(1)
        with col6[0]:
            st.header('90 Days Prediction')
            result, pred_df, orignal = pred(df,90)
            plt.figure(figsize=(14, 7))

            sns.lineplot(x='Date', y='Adj Close', data=orignal, label='Original Data')
            sns.lineplot(x='Date', y='Adj Close', data=pred_df, label='Forecast Data')

            plt.xlabel('Date')
            plt.ylabel('Adj Close')
            plt.legend()

            st.pyplot(plt)



        




                                                   

 









