import logging
import os 
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# log_path = datetime.now()
log_path = os.path.join(os.getcwd(), "log",LOG_FILE)
os.makedirs(log_path,exist_ok=True)
LOG_FILE_PATH = os.path.join(log_path,LOG_FILE)


logging.basicConfig(

    filename=LOG_FILE_PATH,
    format= "[%(asctime)s] %(lineno)d %(name)s %(levelname)s %(message)s",  # formate for storing msg in log_file
    level=logging.INFO  # in the case of INFO onlly i am going to print the msg
)




