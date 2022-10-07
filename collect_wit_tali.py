import argparse
import concurrent.futures
import logging
import os
import pathlib
import re
import time
import urllib.request
from collections import defaultdict
from random import shuffle
from typing import Dict

import numpy as np
import pytube
import tqdm
from datasets import load_dataset
from rich.logging import RichHandler
from yelp_uri.encoding import recode_uri

from storage import load_dict_from_json, save_dict_in_json

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logging = logging.getLogger("rich")


def get_base_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23061991)
    parser.add_argument("--total_results_per_query", type=int, default=1)
    parser.add_argument(
        "--target_dataset_dir", type=str, default="/mnt/nas/datasets/witty-tali"
    )
    parser.add_argument("--resolution_identifier", type=str, default="480p")
    parser.add_argument("--sleep_duration", type=int, default=1)
    # parser.add_argument("--rescan_dataset_files", default=False, action="store_true")
    parser = parser.parse_args()

    return parser


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


useful_keys = [
    # "caption_reference_description",
    "context_page_description",
    "hierarchical_section_title",
]


def download_video_and_meta_data_wrapper(arg_dict):
    return download_video_and_meta_data(**arg_dict)


def download_video_and_meta_data(
    url_idx,
    length,
    target_directory,
    num_threads,
    resolution_identifier,
    sleep_duration,
):
    """
    Downloads a youtube video and its meta data.
    :param resolution_identifier:
    :param num_threads:
    :param url_idx:
    :param length:
    :return:
    :param target_directory: Directory to save the video and meta data in
    :return: True is succesful and False if not
    """
    # init an HTML Session
    video_url = f"https://www.youtube.com/watch?v={url_idx}"
    video_store_filepath = os.path.abspath(f"{target_directory}/{url_idx}")
    video_store_filepath_object = pathlib.Path(video_store_filepath)

    try:
        video_url = f"https://www.youtube.com/watch?v={url_idx}"
        video_store_filepath = os.path.abspath(f"{target_directory}/{url_idx}")
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
            return url_idx, length, True

        save_dict_in_json(
            filepath=f"{video_store_filepath}/meta_data",
            metrics_dict=video_dict,
            overwrite=True,
        )

        video_low_def = youtube_object.streams.get_by_resolution(
            resolution=resolution_identifier
        )

        if video_low_def is None:
            logging.info(
                f"Can't find "
                f"{resolution_identifier} version of, "
                f"{url_idx},"
                f"{youtube_object.streams}"
            )
        else:
            if not os.path.exists(
                f"{video_store_filepath}/full_video_{resolution_identifier}.mp4"
            ):
                logging.info(
                    f"Download "
                    f"{resolution_identifier} version of, "
                    f"{url_idx},"
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
                    f"{url_idx}, as it already exists in "
                    f"{video_store_filepath}/full_video_{resolution_identifier}.mp4"
                )

            time.sleep(sleep_duration)
            logging.info(f"Sleeping for {sleep_duration} seconds..")
    except Exception:

        logging.exception(
            f"Video {video_url}, {video_store_filepath_object} has gone boom, "
            f"will now delete this file"
        )

        return url_idx, length, False

    return url_idx, length, True


def download_length(video_url_idx):
    """
    Downloads a youtube video and its meta data.
    :param video_url_idx:
    :rtype: object
    :return: True is successful and False if not
    """
    # init an HTML Session
    url = f"https://www.youtube.com/watch?v={video_url_idx}"

    try:
        url = f"https://www.youtube.com/watch?v={video_url_idx}"
        youtube_object = pytube.YouTube(url)
        length = youtube_object.length
        return video_url_idx, length
    except Exception:
        logging.exception(f"Couldn't get length for {url}")
        return video_url_idx, 0


def search_for_terms(terms, sort_type="relevance", n=3):
    """
    Search youtube with given terms
    :param terms: What to search youtube for
    :param sort_type: Method of sorting the search results either 'relevance'
                      or 'view-counts'
    :param n: Number of URLs to return
    :return: A list of youtube video identifier codes
    """
    sort_key_to_code = {"relevance": "CAASAjAB", "view-counts": "CAMSAjAB"}
    try:
        url = f"https://www.youtube.com/results?search_query={terms}"
        f"&sp={sort_key_to_code[sort_type]}"
        url = recode_uri(url)
        html = urllib.request.urlopen(url)

        html = html.read().decode()

        terms = re.findall(pattern=r"watch\?v=(\S{11})", string=html)[:n]
    except Exception:
        logging.exception(f"Couldn't find any search results for terms {terms}")
        terms = []

    return terms


def extract_terms_from_dataset(dataset, set_ids) -> Dict:
    """
    Read a txt file and extract a list of query terms
    :rtype: A set object containing query terms extracted from the input txt file
    """
    term_dict = defaultdict(dict)
    with tqdm.tqdm(total=len(set_ids)) as pbar:
        for idx in set_ids:
            output_dict = get_language_specific_wikipedia_data(
                sample=dataset[idx], target_language="en"
            )
            if output_dict is not None:
                output_dict = filter_dict_by_keys(
                    input_dict=output_dict, filter_keys=useful_keys
                )
            else:
                output_dict = {}

            if dataset[idx]["caption_attribution_description"] is not None:
                bits = dataset[idx]["caption_attribution_description"].split(":")
                output_dict["caption"] = ":".join(bits[1:])

                for key, value in output_dict.items():
                    if value is not None:
                        terms = re.findall("[A-Z][^A-Z]*", value)

                        term_string = "+".join(terms)
                        if key not in term_dict[idx]:
                            term_dict[idx][key] = [term_string]
                        else:
                            term_dict[idx][key].append(term_string)
            pbar.update(1)

    return term_dict


def search_and_return_url(search_idx, search_query, total_results_per_query):
    """
    :param total_results_per_query:
    :param search_query:
    :return:
    """

    url_list = set()
    for query_type, queries in search_query.items():
        for sort_type in ["relevance", "view-counts"]:
            for query in queries:
                url_idxs = search_for_terms(
                    terms=query, sort_type=sort_type, n=total_results_per_query
                )
                url_list.add(url_idxs)
    search_query["url_list"] = list(url_list)
    return search_idx, search_query


def search_and_return_url_wrapper(arg_dict):
    return search_and_return_url(**arg_dict)


def parallel_download_video_and_meta_data(
    seed,
    url_ids_to_length_dict,
    target_directory,
    url_to_status_dict_json_filepath,
    num_threads,
    resolution_identifier,
    sleep_duration,
    max_urls=-1,
):
    np.random.seed(seed)
    idx = np.arange(len(list(url_ids_to_length_dict.keys())))
    np.random.shuffle(idx)

    if max_urls == -1:
        max_urls = len(url_ids_to_length_dict)

    if url_to_status_dict_json_filepath.exists():
        url_idx_to_status_dict = load_dict_from_json(
            filepath=url_to_status_dict_json_filepath
        )
    else:
        url_idx_to_status_dict = {}

    temp_url_ids_to_length_dict = {}

    for idx, (url_idx, length) in enumerate(url_ids_to_length_dict.items()):
        temp_url_ids_to_length_dict[url_idx] = length
        if idx == max_urls:
            break
    url_ids_to_length_dict = temp_url_ids_to_length_dict

    for finished_url in url_idx_to_status_dict.keys():
        del url_ids_to_length_dict[finished_url]

    values = list(url_ids_to_length_dict.values())
    total_length_to_download = np.sum(values)

    arg_dicts = [
        dict(
            url_idx=url_idx,
            length=length,
            target_directory=target_directory,
            num_threads=num_threads,
            resolution_identifier=resolution_identifier,
            sleep_duration=sleep_duration,
        )
        for url_idx, length in url_ids_to_length_dict.items()
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        with tqdm.tqdm(total=total_length_to_download, smoothing=0.0) as pbar:
            for video_idx, (url_idx, length, result) in enumerate(
                executor.map(download_video_and_meta_data_wrapper, arg_dicts), start=1
            ):
                pbar.update(length)
                pbar.set_description(
                    f'Done downloading the "{url_idx}: {length} -> {result}" query'
                )
                url_idx_to_status_dict[url_idx] = result
                if video_idx % 1 == 0:
                    save_dict_in_json(
                        metrics_dict=url_idx_to_status_dict,
                        filepath=url_to_status_dict_json_filepath,
                        overwrite=True,
                    )


def parallel_search_return_url_dict(
    search_queries, seed, num_threads, total_results_per_query=3, max_queries=-1
):
    np.random.seed(seed)
    if max_queries != -1:
        search_queries = search_queries[:max_queries]

    arg_dicts = [
        dict(
            search_idx=search_idx,
            search_query=search_query,
            total_results_per_query=total_results_per_query,
        )
        for search_idx, search_query in search_queries.items()
    ]

    with tqdm.tqdm(total=len(arg_dicts), smoothing=0.0) as pbar:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_threads
        ) as executor:
            query_to_url_ids_dict = {}

            for search_idx, query_dict_with_urls in executor.map(
                search_and_return_url_wrapper, arg_dicts
            ):
                pbar.update(1)
                pbar.set_description(f"Done processing {query_dict_with_urls}")
                query_to_url_ids_dict[search_idx] = query_dict_with_urls

    return query_to_url_ids_dict


def parallel_extract_length_from_url(num_threads, url_ids):
    url_idx_to_length_dict = {}
    with tqdm.tqdm(total=len(url_ids), smoothing=0.0) as pbar:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_threads
        ) as executor:
            for video_url_idx, length in executor.map(download_length, url_ids):
                pbar.update(1)
                pbar.set_description(
                    f'Done processing the "{video_url_idx} -> {length}"'
                )
                if length > 0:
                    url_idx_to_length_dict[video_url_idx] = length

    return url_idx_to_length_dict


def download_dataset_given_ids(
    set_ids,
    dataset,
    dataset_directory,
    seed,
    total_results_per_query,
    num_threads,
    resolution_identifier,
    sleep_duration,
    max_queries=-1,
    max_downloads_in_set=-1,
):
    search_queries_dict = extract_terms_from_dataset(dataset=dataset, set_ids=set_ids)

    query_to_url_dict_json_filepath = pathlib.Path(
        f"{dataset_directory}/query_to_url_idx.json"
    )

    url_to_length_json_filepath = pathlib.Path(
        f"{dataset_directory}/url_to_length.json"
    )

    url_to_status_dict_json_filepath = pathlib.Path(
        f"{dataset_directory}/url_to_status.json"
    )

    if query_to_url_dict_json_filepath.exists():
        search_idx_to_query_dict = load_dict_from_json(
            filepath=query_to_url_dict_json_filepath
        )
    else:
        search_idx_to_query_dict = parallel_search_return_url_dict(
            search_queries=search_queries_dict,
            total_results_per_query=total_results_per_query,
            max_queries=max_queries,
            seed=seed,
            num_threads=num_threads,
        )

        save_dict_in_json(
            metrics_dict=search_idx_to_query_dict,
            filepath=query_to_url_dict_json_filepath,
            overwrite=True,
        )

    if url_to_length_json_filepath.exists():
        url_idx_to_length = load_dict_from_json(filepath=url_to_length_json_filepath)
    else:
        urls_extended = [
            url
            for value in search_idx_to_query_dict.values()
            for url in value["url_list"]
        ]
        url_idx_to_length = parallel_extract_length_from_url(
            url_ids=urls_extended, num_threads=num_threads
        )

        save_dict_in_json(
            metrics_dict=url_idx_to_length,
            filepath=url_to_length_json_filepath,
            overwrite=True,
        )

    parallel_download_video_and_meta_data(
        url_ids_to_length_dict=url_idx_to_length,
        target_directory=pathlib.Path(f"{dataset_directory}/"),
        url_to_status_dict_json_filepath=url_to_status_dict_json_filepath,
        max_urls=max_downloads_in_set,
        seed=seed,
        num_threads=num_threads,
        resolution_identifier=resolution_identifier,
        sleep_duration=sleep_duration,
    )


if __name__ == "__main__":
    args = get_base_argument_parser()
    dataset = load_dataset(
        "wikimedia/wit_base", split="train", cache_dir="/mnt/nas/datasets/"
    )
    dataset_ids = [i for i in range(len(dataset))]
    shuffle(dataset_ids)
    split_dict = {"train": 0.6, "val": 0.2, "test": 0.2}
    train_ids = dataset_ids[: int(split_dict["train"] * len(dataset_ids))]
    val_ids = dataset_ids[
        int(split_dict["train"] * len(dataset_ids)) : int(
            split_dict["train"] * len(dataset_ids)
        )
        + int(split_dict["val"] * len(dataset_ids))
    ]
    test_ids = dataset_ids[
        int(split_dict["train"] * len(dataset_ids))
        + int(split_dict["val"] * len(dataset_ids)) :
    ]
    id_dict = {"train": train_ids, "val": val_ids, "test": test_ids}

    for set_name in ["train", "val", "test"]:
        set_ids = id_dict[set_name]
        dataset_directory = pathlib.Path(f"{args.target_dataset_dir}/{set_name}")
        dataset_directory.mkdir(parents=True, exist_ok=True)

        download_dataset_given_ids(
            set_ids=set_ids,
            dataset=dataset,
            seed=args.seed,
            total_results_per_query=args.total_results_per_query,
            dataset_directory=str(dataset_directory),
            num_threads=args.num_threads,
            resolution_identifier=args.resolution_identifier,
            sleep_duration=args.sleep_duration,
        )
