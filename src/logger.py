import logging
import os

def get_logger(name):
    os. makedirs("logs", exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # for terminal  
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    # for file 
    file_handler = logging.FileHandler("logs/app.log")
    file_handler.setFormatter(formatter)
    
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    
    return logger