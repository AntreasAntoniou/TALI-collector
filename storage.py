import numpy as np
import scipy.misc
import shutil
import torch
import scipy
import json
import os
from rich import print
import requests
import tqdm  # progress bar
import pathlib

def save_dict_in_json(filepath, metrics_dict, overwrite):
    """
    Saves a metrics .json file with the metrics
    :param log_dir: Directory of log
    :param metrics_file_name: Name of .csv file
    :param metrics_dict: A dict of metrics to add in the file
    :param overwrite: If True overwrites any existing files with the same filepath,
    if False adds metrics to existing
    """

    if isinstance(filepath, pathlib.Path):
        filepath = str(filepath)

    if ".json" not in filepath:
        filepath = f"{filepath}.json"

    metrics_file_path = filepath

    if overwrite and os.path.exists(metrics_file_path):
        os.remove(metrics_file_path)

    with open(metrics_file_path, "w+") as json_file:
        json.dump(metrics_dict, json_file, indent=4, sort_keys=True)


def load_dict_from_json(filepath):
    """
    Loads the metrics in a dictionary.
    :param log_dir: The directory in which the log is saved
    :param metrics_file_name: The name of the metrics file
    :return: A dict with the metrics
    """

    if isinstance(filepath, pathlib.Path):
        filepath = str(filepath)

    if ".json" not in filepath:
        filepath = f"{filepath}.json"

    with open(filepath) as json_file:
        metrics_dict = json.load(json_file)

    return metrics_dict
