from exception import CustomException
from logger import logging
from steps.model_train import ModelTrain
from steps.data_ingestion import IngestData
from steps.data_transformatin import ScaleSplit
from dataclasses import dataclass


class TrainPipeline:

    def __init__(self):

        self.ingestdata = IngestData()
        self.transformdata = ScaleSplit()
        self.modeltrain = ModelTrain()

    def training_pipeline(self, compony_name):
        final_df_path = self.ingestdata.initiate_data_ingestion(compony_name)
        X_train, y_train, X_test, ytest= self.transformdata.initiate_data_transformation(final_df_path)
        history = self.modeltrain.model_train(X_train, y_train)









