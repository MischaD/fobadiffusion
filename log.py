import logging
import sys
import os
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(filename)s-%(funcName)s-%(lineno)04d | %(levelname)s | %(message)s')

os.makedirs('./log/console_logs/', exist_ok=True)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler('./log/console_logs/' + datetime.now().strftime("%y_%m_%d_%H:%M:%S") + '.log')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)