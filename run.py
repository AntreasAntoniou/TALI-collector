import argparse
import concurrent.futures
import logging
import multiprocessing as mp
import os
import pathlib
import re
import string
import time
import urllib.request

import numpy as np
import pytube
import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip
from rich.logging import RichHandler

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
    parser.add_argument("--total_results_per_query", type=int, default=3)
    parser.add_argument("--target_dataset_dir", type=str, default='dataset/')

    # parser.add_argument("--rescan_dataset_files", default=False, action="store_true")
    parser = parser.parse_args()

    return parser


def download_video_and_meta_data_wrapper(arg_dict):
    return download_video_and_meta_data(**arg_dict)


def download_video_and_meta_data(url_idx, length, target_directory, num_threads):
    """
    Downloads a youtube video and its meta data.
    :param num_threads:
    :param url_idx:
    :param length:
    :return:
    :param target_directory: Directory to save the video and meta data in
    :return: True is succesful and False if not
    """
    # init an HTML Session
    input_video_low_def_path = None
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

        video_low_def = youtube_object.streams.get_by_resolution(resolution="480p")

        if video_low_def is None:
            logging.info(f"Can't find low def version of, {url_idx},"
                         f" {youtube_object.streams}")
        else:
            video_low_def.download(
                output_path=f"{video_store_filepath}/",
                filename="full_video_360p.mp4",
                max_retries=1,
            )

            time.sleep(1)

        input_video_low_def_path = f"{video_store_filepath}/full_video_360p.mp4"

        clip_count = 0
        with VideoFileClip(input_video_low_def_path) as video_low_def:
            try:
                output_video_low_def_path = (
                    f"{video_store_filepath}/video.mp4"
                )

                output_audio_low_def_path = (
                    f"{video_store_filepath}/audio.mp3"
                )

                video_low_def.write_videofile(
                    filename=output_video_low_def_path,
                    fps=30,
                    codec="libx264",
                    bitrate=None,
                    audio=True,
                    audio_fps=44100,
                    preset="medium",
                    audio_nbytes=4,
                    audio_codec="mp3",
                    audio_bitrate=None,
                    audio_bufsize=2000,
                    temp_audiofile=output_audio_low_def_path,
                    rewrite_audio=True,
                    remove_temp=False,
                    write_logfile=False,
                    verbose=False,
                    threads=num_threads,
                    ffmpeg_params=None,
                    logger=None,
                )

                clip_count += 1
                if os.path.exists(f"{video_store_filepath}/full_video_360p.mp4"):
                    os.remove(f"{video_store_filepath}/full_video_360p.mp4")

            except Exception:
                logging.exception('Gone boom 😼')
    except Exception:

        # Just print(e) is cleaner and more likely what you want,

        # but if you insist on printing message specifically whenever possible...

        logging.exception(f'Video {input_video_low_def_path} has gone boom, '
                          f'will now delete this file')

        if input_video_low_def_path is not None:
            os.remove(input_video_low_def_path)

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
    try:
        url = f"https://www.youtube.com/watch?v={video_url_idx}"
        youtube_object = pytube.YouTube(url)
        length = youtube_object.length
        return video_url_idx, length
    except:
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
        html = urllib.request.urlopen(
            f"https://www.youtube.com/results?search_query="
            f"{terms}"
            f"&sp={sort_key_to_code[sort_type]}"
        )

        html = html.read().decode()

        terms = re.findall(pattern=r"watch\?v=(\S{11})", string=html)[:n]
    except:
        terms = []

    return terms


def extract_terms_from_txt(filepath: string) -> set:
    """
    Read a txt file and extract a list of query terms
    :rtype: A set object containing query terms extracted from the input txt file
    """
    term_set = set()
    with open(file=filepath, mode="r") as file_reader:
        lines = file_reader.readlines()

    for line in lines:
        line = line[5:].replace("\n", "")
        last_character_digit_match = re.search(r"\d$", line)
        # if the string ends in digits m will be a Match object, or None otherwise.
        if last_character_digit_match is not None:
            line = line[:1]

        terms = re.findall("[A-Z][^A-Z]*", line)

        term_string = "+".join(terms)
        term_set.add(term_string)

    return term_set


def search_and_return_url(search_query, total_results_per_query):
    """
    :param total_results_per_query:
    :param search_query:
    :return:
    """
    search_query = f"How+to+{search_query}"
    url_list = []
    for sort_type in ["relevance", "view-counts"]:
        url_idxs = search_for_terms(
            terms=search_query, sort_type=sort_type, n=total_results_per_query
        )
        url_list.extend(url_idxs)

    return search_query, url_list


def search_and_return_url_wrapper(arg_dict):
    return search_and_return_url(**arg_dict)


def parallel_download_video_and_meta_data(
        seed,
        url_ids_to_length_dict,
        target_directory,
        url_to_status_dict_json_filepath,
        num_threads,
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
        dict(url_idx=url_idx, length=length, target_directory=target_directory,
             num_threads=num_threads)
        for url_idx, length in url_ids_to_length_dict.items()
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        with tqdm.tqdm(total=total_length_to_download, smoothing=0.0) as pbar:
            for video_idx, (url_idx, length, result) in enumerate(
                    executor.map(download_video_and_meta_data_wrapper, arg_dicts),
                    start=1
            ):
                pbar.update(length)
                pbar.set_description(
                    f'Done processing the "{url_idx}: {length} -> {result}" query'
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
    idx = np.arange(len(search_queries))
    np.random.shuffle(idx)
    search_queries = [search_queries[i] for i in idx]
    if max_queries != -1:
        search_queries = search_queries[:max_queries]

    arg_dicts = [
        dict(search_query=search_query, total_results_per_query=total_results_per_query)
        for search_query in search_queries
    ]

    with tqdm.tqdm(total=len(arg_dicts), smoothing=0.0) as pbar:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_threads
        ) as executor:
            query_to_url_ids_dict = {}

            for query_string, video_url_ids in executor.map(
                    search_and_return_url_wrapper, arg_dicts
            ):
                pbar.update(1)
                pbar.set_description(
                    f'Done processing the "{query_string} ->' f' {video_url_ids}" query'
                )
                query_to_url_ids_dict[query_string] = video_url_ids

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
                    f'Done processing the "{video_url_idx} -> {length}"' f" query"
                )
                if length > 0:
                    url_idx_to_length_dict[video_url_idx] = length

    return url_idx_to_length_dict


def download_dataset_given_txt_file(
        txt_filepath,
        dataset_directory,
        seed,
        total_results_per_query,
        num_threads,
        max_queries=-1,
        max_downloads_in_set=-1,
):
    search_queries = list(extract_terms_from_txt(filepath=txt_filepath))

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
        query_to_url_dict = load_dict_from_json(
            filepath=query_to_url_dict_json_filepath
        )
    else:
        query_to_url_dict = parallel_search_return_url_dict(
            search_queries=search_queries,
            total_results_per_query=total_results_per_query,
            max_queries=max_queries,
            seed=seed, num_threads=num_threads
        )

        save_dict_in_json(
            metrics_dict=query_to_url_dict,
            filepath=query_to_url_dict_json_filepath,
            overwrite=True,
        )

    if url_to_length_json_filepath.exists():
        url_idx_to_length = load_dict_from_json(filepath=url_to_length_json_filepath)
    else:
        urls_extended = [url for urls in query_to_url_dict.values() for url in urls]
        url_idx_to_length = parallel_extract_length_from_url(url_ids=urls_extended,
                                                             num_threads=num_threads)

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
        seed=seed, num_threads=num_threads
    )


if __name__ == "__main__":
    args = get_base_argument_parser()
    for set_name in ['debug', 'train', 'val', 'test']:
        txt_file = pathlib.Path(f"wikihow_queries/all_{set_name}.txt")

        dataset_directory = pathlib.Path(f"{args.target_dataset_dir}/{set_name}")
        dataset_directory.mkdir(parents=True, exist_ok=True)

        download_dataset_given_txt_file(
            txt_filepath=txt_file,
            seed=args.seed,
            total_results_per_query=args.total_results_per_query,
            dataset_directory=str(dataset_directory),
            num_threads=args.num_threads
        )
