from pipelines.train_pipeline import TrainPipeline
from exception import CustomException
import sys


if __name__ == "__main__":
    train_pipeline = TrainPipeline()
    train_pipeline.training_pipeline()

