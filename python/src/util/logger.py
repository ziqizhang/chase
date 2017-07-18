
import logging
import os

logger = logging.getLogger(__name__)
LOG_DIR=os.getcwd()+"/logs"
logging.basicConfig(filename=LOG_DIR+'/log.txt', level=logging.INFO, filemode='w')

