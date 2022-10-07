import argparse
import concurrent.futures
import logging
import os
import pathlib
import random
import re
import string
import time
import urllib.request
from collections import defaultdict
from random import shuffle
from typing import Dict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytube
import tqdm
from datasets import load_dataset
from rich import print
from rich.logging import RichHandler
from rich.traceback import install
from transformers import CLIPModel
from yelp_uri.encoding import recode_uri

from clip_helper import get_scores
from storage import load_dict_from_json, save_dict_in_json

install(show_locals=False, word_wrap=True, width=350)

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logging = logging.getLogger("rich")


def get_base_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23061991)
    parser.add_argument("--total_downloads_per_query", type=int, default=1)
    parser.add_argument("--total_results_per_query", type=int, default=100)
    parser.add_argument(
        "--target_dataset_dir", type=str, default="/mnt/nas/datasets/witty-tali"
    )
    parser.add_argument("--resolution_identifier", type=str, default="360p")
    parser.add_argument("--pool_type", type=str, default="Process")
    parser.add_argument("--sleep_duration", type=float, default=1)
    parser.add_argument("--starting_sample_idx", type=int, default=0)
    parser.add_argument("--ending_sample_idx", type=int, default=-1)
    parser = parser.parse_args()

    return parser


args = get_base_argument_parser()

score_table_folderpath = pathlib.Path(f"{args.target_dataset_dir}/score_table.parquet")

if not score_table_folderpath.exists():
    score_table_folderpath.mkdir(parents=True, exist_ok=True)


class PoolType:
    process: str = "Process"
    thread: str = "Thread"


useful_keys = [
    # "caption_reference_description",
    "context_page_description",
    "hierarchical_section_title",
]


def get_language_specific_wikipedia_data(sample, target_language="en"):
    multi_lingual_wikipedia_data = sample["wit_features"]

    if target_language in multi_lingual_wikipedia_data["language"]:
        language_idx = multi_lingual_wikipedia_data["language"].index(target_language)
        target_language_dict = {}
        for key, value in multi_lingual_wikipedia_data.items():
            target_language_dict[key] = value[language_idx]
        return target_language_dict
    else:
        return None


def filter_dict_by_keys(input_dict, filter_keys):
    output_dict = {}
    for key in filter_keys:
        output_dict[key] = input_dict[key]

    return output_dict


def download_video_meta_data_and_youtube_object(video_id: str, target_directory: str):
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    video_store_filepath = os.path.abspath(f"{target_directory}")
    video_store_filepath_object = pathlib.Path(video_store_filepath)

    try:
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        video_store_filepath = os.path.abspath(f"{target_directory}")
        video_store_filepath_object = pathlib.Path(video_store_filepath)

        video_store_filepath_object.mkdir(parents=True, exist_ok=True)

        youtube_object = pytube.YouTube(video_url)
        video_dict = {
            "captions": {},
            "age_restricted": youtube_object.age_restricted,
            "check_availability": youtube_object.check_availability(),
            "title": youtube_object.title,
            "rating": youtube_object.rating,
            "length": youtube_object.length,
            "views": youtube_object.views,
            "author": youtube_object.author,
            "meta_data": youtube_object.metadata.raw_metadata,
        }

        for caption_item in youtube_object.captions:
            video_dict["captions"][f"{caption_item.code}"] = caption_item.xml_captions

        if (
            video_dict["age_restricted"]
            or "en" not in video_dict["captions"]
            and "a.en" not in video_dict["captions"]
        ):
            return None

        save_dict_in_json(
            filepath=f"{video_store_filepath}/meta_data",
            metrics_dict=video_dict,
            overwrite=True,
        )
        return video_dict, youtube_object
    except Exception:

        logging.exception(
            f"Video {video_url}, {video_store_filepath_object} has gone boom, "
            f"will now delete this file"
        )

        return None


def download_video_and_meta_data(
    video_id: str,
    target_directory: str,
    resolution_identifier: str,
    sleep_duration: int,
):
    output = download_video_meta_data_and_youtube_object(
        video_id=video_id, target_directory=target_directory
    )

    if output is None:
        return video_id, False
    else:
        video_dict, youtube_object = output

    video_store_filepath = os.path.abspath(f"{target_directory}")
    video_low_def = youtube_object.streams.get_by_resolution(
        resolution=resolution_identifier
    )

    if video_low_def is None:
        logging.info(
            f"Can't find "
            f"{resolution_identifier} version of, "
            f"{video_id},"
            f"{youtube_object.streams}"
        )
    else:
        if not os.path.exists(
            f"{video_store_filepath}/full_video_{resolution_identifier}.mp4"
        ):
            logging.info(
                f"Download "
                f"{resolution_identifier} version of, "
                f"{video_id},"
                f"{video_store_filepath}/"
                f"full_video_{resolution_identifier}.mp4"
            )

            video_low_def.download(
                output_path=f"{video_store_filepath}/",
                filename=f"full_video_{resolution_identifier}.mp4",
                max_retries=1,
            )

        else:
            logging.info(
                f"Skipping "
                f"{resolution_identifier} version of, "
                f"{video_id}, as it already exists in "
                f"{video_store_filepath}/full_video_{resolution_identifier}.mp4"
            )

        time.sleep(sleep_duration)
        logging.info(f"Sleeping for {sleep_duration} seconds..")

    return video_id, True


def download_length(video_url_idx):
    url = f"https://www.youtube.com/watch?v={video_url_idx}"

    try:
        url = f"https://www.youtube.com/watch?v={video_url_idx}"
        youtube_object = pytube.YouTube(url)
        length = youtube_object.length
        return video_url_idx, length
    except Exception:
        logging.exception(f"Couldn't get length for {url}")
        return video_url_idx, 0


def search_for_video_ids(terms_string: str, sort_type: str = "relevance", n: int = 100):
    sort_key_to_code = {"relevance": "CAASAjAB", "view-counts": "CAMSAjAB"}
    try:
        url = f"https://www.youtube.com/results?search_query={terms_string}"
        f"&sp={sort_key_to_code[sort_type]}"
        url = recode_uri(url)
        html = urllib.request.urlopen(url)

        html = html.read().decode()

        terms = re.findall(pattern=r"watch\?v=(\S{11})", string=html)[:n]
    except Exception:
        logging.exception(f"Couldn't find any search results for terms {terms_string}")
        terms = []

    return terms


def extract_terms_dict_from_sample(sample: Dict) -> Dict:
    term_dict = defaultdict(list)
    output_dict = get_language_specific_wikipedia_data(
        sample=sample, target_language="en"
    )
    if output_dict is not None:
        output_dict = filter_dict_by_keys(
            input_dict=output_dict, filter_keys=useful_keys
        )
    else:
        output_dict = {}

    if sample["caption_attribution_description"] is not None:
        bits = sample["caption_attribution_description"].split(":")
        output_dict["caption"] = ":".join(bits[1:])

        for key, value in output_dict.items():
            if value is not None:
                terms = re.findall("[A-Z][^A-Z]*", value)

                term_string = "+".join(terms)
                term_dict[key].append(term_string)

    return term_dict


def search_and_return_url(search_idx, search_query, total_results_per_query):
    url_list = set()
    for query_type, queries in search_query.items():
        for sort_type in ["relevance", "view-counts"]:
            for query in queries:
                url_idxs = search_for_video_ids(
                    terms_string=query, sort_type=sort_type, n=total_results_per_query
                )
                url_list.add(url_idxs)
    search_query["url_list"] = list(url_list)
    return search_idx, search_query


def filter_video_ids_with_clip(
    reference_term, term_related_video_ids, dataset_directory, idx
):
    titles = []
    directory_path = os.path.join(dataset_directory, str(idx))
    valid_video_ids_related_to_term = []

    for video_id in term_related_video_ids:
        video_directory_path = os.path.join(directory_path, video_id)

        output = download_video_meta_data_and_youtube_object(
            video_id=video_id, target_directory=video_directory_path
        )

        if output is not None:
            video_dict, youtube_object = output
            titles.append(youtube_object.title)
            valid_video_ids_related_to_term.append(video_id)

    if len(titles) > 0:

        clip_scores = get_scores(
            reference_text=reference_term,
            query_text=titles,
            query_ids=valid_video_ids_related_to_term,
        )

        entry_filepath = score_table_folderpath / f"{idx}.parquet"

        table_entry = pa.table(
            [
                [idx],
                [clip_scores.reference_text],
                [clip_scores.sorted_query_texts],
                [clip_scores.sorted_scores],
                [clip_scores.sorted_query_ids],
                [clip_scores.sorted_args],
            ],
            names=[
                "wit_idx",
                "reference_text",
                "sorted_query_texts",
                "sorted_scores",
                "sorted_query_ids",
                "sorted_args",
            ],
        )

        table = table_entry

        pq.write_table(table, entry_filepath)

        return clip_scores.sorted_query_ids[: args.total_downloads_per_query]
    else:
        return []


def download_video_meta_data_given_sample(
    sample: Dict, idx: int, dataset_directory: str
):
    outputs = []
    search_term_dict = extract_terms_dict_from_sample(sample=sample)
    for term_name, term_values in search_term_dict.items():
        for sort_type in ["relevance", "view-counts"]:
            term_related_video_ids = search_for_video_ids(
                terms_string=term_values,
                n=args.total_results_per_query,
                sort_type=sort_type,
            )
            term_related_video_ids = list(set(term_related_video_ids))
            term_related_video_ids = filter_video_ids_with_clip(
                reference_term=term_values,
                term_related_video_ids=term_related_video_ids,
                dataset_directory=dataset_directory,
                idx=idx,
            )
            for video_id in term_related_video_ids:
                if pathlib.Path(os.path.join(dataset_directory, str(idx))).exists():
                    video_files_found = list(
                        pathlib.Path(os.path.join(dataset_directory, str(idx))).rglob(
                            "*.mp4"
                        )
                    )

                    if len(video_files_found) > 0:
                        return outputs

                directory_path = os.path.join(dataset_directory, str(idx), video_id)

                if not os.path.exists(directory_path):
                    pathlib.Path(directory_path).mkdir(parents=True, exist_ok=True)

                    output = download_video_and_meta_data(
                        video_id=video_id,
                        target_directory=directory_path,
                        resolution_identifier=args.resolution_identifier,
                        sleep_duration=args.sleep_duration,
                    )
                    outputs.append(output)
    return outputs


def download_video_meta_data_given_sample_wrapper(args_dict):
    return download_video_meta_data_given_sample(**args_dict)


def download_dataset_given_ids(set_ids, dataset, dataset_directory):
    arg_dict_list = [
        dict(sample=dataset[idx], idx=idx, dataset_directory=dataset_directory)
        for idx in set_ids
    ]
    pool_type = (
        concurrent.futures.ProcessPoolExecutor
        if args.pool_type.lower() == PoolType.process.lower()
        else concurrent.futures.ThreadPoolExecutor
        if args.pool_type.lower() == PoolType.thread.lower()
        else None
    )

    if pool_type is None:
        raise ValueError(
            f"Pool type {args.pool_type} is not supported, please use one of {PoolType}"
        )
    logging.info(f"Using {pool_type} for parallel processing")
    with tqdm.tqdm(total=len(set_ids), smoothing=0.0) as pbar:

        with pool_type(max_workers=args.num_threads) as executor:
            for video_idx, outputs_list in enumerate(
                executor.map(
                    download_video_meta_data_given_sample_wrapper, arg_dict_list
                ),
                start=1,
            ):

                pbar.update(1)
                pbar.set_description(f"Downloading video {outputs_list}")


if __name__ == "__main__":
    dataset = load_dataset(
        "wikimedia/wit_base", split="train", cache_dir="/mnt/nas/datasets/"
    )
    # random.seed(args.seed)
    dataset_ids = [
        i for i in range(args.starting_sample_idx, args.ending_sample_idx, 1)
    ]

    id_buckets = []
    num_buckets = (len(dataset_ids) // 20000) + 1

    for i in range(num_buckets):
        id_buckets.append(dataset_ids[i * 20000 : (i + 1) * 20000])

    set_ids = dataset_ids
    dataset_directory = pathlib.Path(f"{args.target_dataset_dir}/{'all'}")
    dataset_directory.mkdir(parents=True, exist_ok=True)

    with tqdm.tqdm(total=num_buckets, smoothing=0.0) as pbar:
        for id_bucket in id_buckets:
            download_dataset_given_ids(
                set_ids=id_bucket, dataset=dataset, dataset_directory=dataset_directory
            )
            pbar.update(1)
