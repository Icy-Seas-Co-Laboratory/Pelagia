import datetime
import cv2
import json
from PIL import Image
import csv
import numpy as np
import sys
from time import time, sleep
from logging.handlers import TimedRotatingFileHandler
import shutil
import os
import logging.config
import logging

from .utils.serialization import json_ready
from .utils.validation import validate_schema_name

def loadModel(config, logger):
    """
    Helper function to load model and sidecar. 
    """
    model_path = config['classification']['model_dir'] + \
        os.path.sep + config['classification']['model_name'] + ".keras"
    label_path = config['classification']['model_dir'] + \
        os.path.sep + config['classification']['model_name'] + ".json"
    logger.info(f"Loading model from {model_path}.")
    logger.info(f"Loading model sidecar from {label_path}.")
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)

    if config['classification']['feature_space']:
        # Modify model for feature extraction:
        # Remove final softmax activation to expose penultimate dense layer. Generate as new model object.
        x = model.layers[-3].output
        model = tf.keras.models.Model(inputs=model.input, outputs=x)

    with open(label_path, 'r') as file:
        sidecar = json.load(file)
        logger.debug('Sidecar Loaded.')

    logger.info(
        f"Loaded keras model {config['classification']['model_name']} and sidecar JSON file.")

    return model, sidecar
