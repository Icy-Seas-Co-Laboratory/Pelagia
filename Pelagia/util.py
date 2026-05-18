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


def calcThreshold(gray,
                  runCanny=True,
                  cannyParams=(30, 80),
                  dilateKernel=(3, 3),
                  min_contrast=120,
                  min_background_fraction=0.2):
    otsu_value, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    background = np.median(gray)

    # If Otsu is separating background noise, the threshold will be too close
    # to the background brightness.
    if background - otsu_value < min_contrast:
        safe_threshold = background - min_contrast
        _, thresh = cv2.threshold(
            gray, safe_threshold, 255, cv2.THRESH_BINARY_INV)

    # reject masks that are implausibly full.
    background_fraction = 1 - np.count_nonzero(thresh) / thresh.size
    if background_fraction < min_background_fraction:
        return np.zeros_like(gray, dtype=np.uint8)

    if runCanny:
        graySmooth = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(
            graySmooth, cannyParams[0], cannyParams[1], L2gradient=True)
        thresh = thresh | edges

    kernel = np.ones(dilateKernel, np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    return thresh


def calcStats(grayROI=None, cnt=None, x=None, y=None, w=None, h=None, rescale_factor=1.0):
    if cnt is None:
        return (['x', 'y', 'w', 'h', 'major_axis', 'minor_axis', 'area', 'perimeter', 'min_gray_value', 'mean_gray_value'])

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    area = area * rescale_factor**2
    perimeter = perimeter * rescale_factor

    if len(cnt) >= 5:  # Minimum number of points required to fit an ellipse
        ellipse = cv2.fitEllipse(cnt)
        center, axes, angle = ellipse
        major_axis_length = round(max(axes) * rescale_factor, 1)
        minor_axis_length = round(min(axes) * rescale_factor, 1)
    else:
        major_axis_length = -1
        minor_axis_length = -1
    mean_gray_value = np.mean(grayROI)
    min_gray_value = np.mean(grayROI)
    return ([x, y, w, h, major_axis_length, minor_axis_length, int(area), int(perimeter), int(min_gray_value), int(mean_gray_value)])


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


def is_file_above_minimum_size(file_path, min_size, logger):
    """
    Check if the file at file_path is larger than min_size bytes.

    :param file_path: Path to the file
    :param min_size: Minimum size in bytes
    :return: True if file size is above min_size, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    try:
        file_size = os.path.getsize(file_path)
        return file_size > min_size
    except OSError as e:
        logger.error(f"Error: {e}")
        return False


def delete_file(file_path, logger):
    """
    Delete the file at file_path.

    :param file_path: Path to the file to be deleted
    """

    try:
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)
            logger.debug(f"The folder '{file_path}' has been deleted.")
        else:
            os.remove(file_path)
            logger.debug(f"The file '{file_path}' has been deleted.")
    except FileNotFoundError:
        logger.debug(f"The file '{file_path}' does not exist.")
    except PermissionError:
        logger.warn(f"Permission denied: unable to delete '{file_path}'.")
    except OSError as e:
        logger.error(f"Error: {e}")


def setup_logger(name, config):
    """
    Helper function to construct a new logger.
    """
    if name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        return logger

    logger = logging.getLogger(name)
    # the level should be the lowest level set in handlers
    logger.setLevel(logging.DEBUG)

    log_format = logging.Formatter(
        '[%(levelname)s] (%(process)d) %(asctime)s - %(message)s')
    if not os.path.exists(config['general']['log_path']):
        try:
            os.makedirs(config['general']['log_path'])
        except PermissionError:
            print(
                f"Permission denied: Unable to create directory '{config['general']['log_path']}'.")
            print('Logging will not be performed and may crash the script.')
        except OSError as e:
            print(
                f"Error creating directory '{config['general']['log_path']}': {e}")
            print('Logging will not be performed and may crash the script.')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    debug_handler = TimedRotatingFileHandler(
        f"{config['general']['log_path']}{name} debug.log", interval=1, backupCount=14)
    debug_handler.setFormatter(log_format)
    debug_handler.setLevel(logging.DEBUG)
    logger.addHandler(debug_handler)

    info_handler = TimedRotatingFileHandler(
        f"{config['general']['log_path']}{name} info.log", interval=1, backupCount=14)
    info_handler.setFormatter(log_format)
    info_handler.setLevel(logging.INFO)
    logger.addHandler(info_handler)

    error_handler = TimedRotatingFileHandler(
        f"{config['general']['log_path']}{name} error.log", interval=1, backupCount=14)
    error_handler.setFormatter(log_format)
    error_handler.setLevel(logging.ERROR)
    logger.addHandler(error_handler)
    return logger


def is_file_finished_writing(file_path, wait_time=0.2):
    initial_size = os.path.getsize(file_path)
    sleep(wait_time)
    new_size = os.path.getsize(file_path)

    if initial_size == new_size:
        return True
    return False


def generate_quantile_image_field(
        avi_path,
        q,
        output_path=None,
        frame_step=1,
        max_frames=None,
        start_frame=0,
        grayscale=True,
        return_float=False,
        logger=None):
    """
    Generate a pixel-wise quantile image field from a video file.

    :param avi_path: Path to the source AVI/video file.
    :param q: quantile to calculate (e.g., 0.5 = median).
    :param output_path: Optional path where the quantile image should be saved.
    :param frame_step: Use every nth frame. Defaults to 1, which uses every frame.
    :param max_frames: Optional maximum number of sampled frames to use.
    :param start_frame: Zero-based frame index to start reading from.
    :param grayscale: Convert frames to grayscale before calculating the quantile.
    :param return_float: Return the raw floating point quantile instead of image dtype.
    :param logger: Optional logger for progress/debug messages.
    :return: Quantile image field as a numpy array.
    """
    avi_path = os.fspath(avi_path)
    if output_path is not None:
        output_path = os.fspath(output_path)

    if not os.path.exists(avi_path):
        raise FileNotFoundError(f"Video file does not exist: {avi_path}")

    if frame_step < 1:
        raise ValueError("frame_step must be >= 1.")

    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1 when provided.")

    if start_frame < 0:
        raise ValueError("start_frame must be >= 0.")

    video = cv2.VideoCapture(avi_path)
    if not video.isOpened():
        raise OSError(f"Issue opening video {avi_path}.")

    if start_frame > 0:
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    frame_index = start_frame
    sampled_frames = 0
    expected_shape = None
    image_dtype = None

    try:
        while video.isOpened():
            good_return, frame = video.read()
            if not good_return:
                break

            if (frame_index - start_frame) % frame_step == 0:
                if grayscale:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if expected_shape is None:
                    expected_shape = frame.shape
                    image_dtype = frame.dtype
                elif frame.shape != expected_shape:
                    raise ValueError(
                        f"Frame shape changed from {expected_shape} "
                        f"to {frame.shape} at frame {frame_index}.")

                frames.append(frame)
                sampled_frames += 1

                if max_frames is not None and sampled_frames >= max_frames:
                    break

            frame_index += 1
    finally:
        video.release()

    if not frames:
        raise ValueError(f"No frames were read from {avi_path}.")

    if logger is not None:
        logger.debug(
            f"Calculating quantile image field from {sampled_frames} frames in {avi_path}.")

    quantile_field = np.quantile(np.stack(frames, axis=0), q, axis=0)

    if np.issubdtype(image_dtype, np.integer):
        image_field = np.rint(quantile_field).clip(
            np.iinfo(image_dtype).min,
            np.iinfo(image_dtype).max
        ).astype(image_dtype)
    else:
        image_field = quantile_field.astype(image_dtype)

    if return_float:
        field = quantile_field
    else:
        field = image_field

    if output_path is not None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if not cv2.imwrite(output_path, image_field):
            raise OSError(
                f"Unable to write quantile image field to {output_path}.")

    return field
