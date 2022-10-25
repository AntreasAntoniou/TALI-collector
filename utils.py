import itertools
import json
import logging
import pathlib
from typing import Dict, Union

import defusedxml.ElementTree as ET
import numpy as np
from rich.logging import RichHandler


def get_logger():
    FORMAT = "%(message)s"
    logging.basicConfig(
        level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )

    logger = logging.getLogger("rich")
    return logger


logger = get_logger()


def save_json(
    filepath: Union[str, pathlib.Path],
    target_dict: Dict[str, str],
    overwrite: bool = True,
):

    if isinstance(filepath, str):
        filepath = pathlib.Path(filepath)

    if overwrite and filepath.exists():
        filepath.unlink(missing_ok=True)

    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True)

    with open(filepath, "w") as json_file:
        json_file.write(json.dumps(target_dict))


def load_json(filepath):
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

    with open(filepath, "rb") as json_file:
        metrics_dict = json.loads(json_file.read())

    return metrics_dict


def convert_keys_to_float(d: dict):
    new_dict = {}
    for k, v in d.items():
        try:
            new_key = float(k)
        except ValueError:
            new_key = k
        if type(v) == dict:
            v = convert_keys_to_float(v)
        new_dict[new_key] = v
    return new_dict


def convert_keys_to_str(d: dict):
    new_dict = {}
    for k, v in d.items():
        try:
            new_key = str(k)
        except ValueError:
            new_key = k
        if type(v) == dict:
            v = convert_keys_to_str(v)
        new_dict[new_key] = v
    return new_dict


def load_text_into_language_time_stamps(caption_dict: Dict):

    captions_matched = {
        key: value for key, value in caption_dict.items() if key in ["a.en", "en"]
    }

    if len(captions_matched) > 1:
        selected_key = "en"
    else:
        selected_key = list(captions_matched.keys())[0]

    selected_captions = captions_matched[selected_key]
    xml_tree = ET.fromstring(selected_captions)

    root = list(xml_tree.iter())
    timestamp_to_caption_dict = {}

    for item in root:
        if selected_key == "a.en":
            children_text = [
                child.text.replace("\n", " ")
                for child in item
                if child.text is not None
            ]
            if item.tag == "p" and children_text:
                timestamp_to_caption_dict[
                    float(item.attrib["t"]) / 1000
                ] = children_text

        elif selected_key == "en":
            if item.tag == "p" and len(item.items()) == 2:
                [(_, start), (_, dur)] = item.items()

                timestamp_to_caption_dict[float(start) / 1000] = (
                    item.text.replace("\n", " ") if item.text is not None else ""
                )

    return timestamp_to_caption_dict


def get_text_tokens(caption_dict, start_timestamp, end_timestamp):
    timestamp_to_caption_dict = load_text_into_language_time_stamps(
        caption_dict=caption_dict
    )
    start_timestamp = float(np.floor(start_timestamp))
    end_timestamp = float(np.floor(end_timestamp))

    if not timestamp_to_caption_dict:
        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.exception(f"No captions found for {caption_dict}")
        return None

    temp_timestamp_to_caption_dict = {}

    for current_start_timestamp in sorted(timestamp_to_caption_dict.keys()):
        current_start_timestamp_float = float(current_start_timestamp)
        if start_timestamp <= current_start_timestamp_float <= end_timestamp:
            temp_timestamp_to_caption_dict[
                current_start_timestamp_float
            ] = timestamp_to_caption_dict[current_start_timestamp]

        if current_start_timestamp_float > end_timestamp:
            break

    return temp_timestamp_to_caption_dict
