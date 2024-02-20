import logging
import pathlib
import random
import re
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Union

import fire
import pytube
import yaml
from rich import print
from rich.logging import RichHandler
from rich.traceback import install
from tqdm.auto import tqdm
from yelp_uri.encoding import recode_uri

from utils import (
    convert_keys_to_str,
    load_text_into_language_time_stamps,
    save_json,
)

install()

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.DEBUG,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)

logging = logging.getLogger("rich")


class PoolType:
    process: str = "Process"
    thread: str = "Thread"


@dataclass
class TargetLanguageOutput:
    target_language_dict: Dict


def filter_dict_by_keys(input_dict: Dict, filter_keys: List[str]) -> Dict:
    output_dict = {}
    for key in filter_keys:
        output_dict[key] = input_dict[key]

    return output_dict


@dataclass
class VideoDataOutput:
    watch_url: Optional[str] = None
    embed_url: Optional[str] = None
    video_id: Optional[str] = None
    title: Optional[str] = None
    length: Optional[int] = None
    age_restricted: Optional[bool] = None
    views: Optional[int] = None
    author: Optional[str] = None
    thumbnail_url: Optional[str] = None
    channel_id: Optional[str] = None
    channel_url: Optional[str] = None
    description: Optional[str] = None
    keywords: Optional[List[str]] = None
    publish_date: Optional[Any] = None
    video_store_filepath: Optional[str] = None


@dataclass
class CaptionDataOutput:
    captions_dict: Dict


@dataclass
class VideoDownloaderObject:
    success: bool
    video_id: str


@dataclass
class SortType:
    name: str
    youtube_code: str


@dataclass
class DownloadLength:
    video_id: str
    length: int


@dataclass
class SearchOutput:
    search_idx: int
    search_query: Dict[str, Any]


@dataclass
class SortKeyStringToCode:
    relevance = SortType(name="relevance", youtube_code="CAASAjAB")
    view_counts = SortType(name="view_counts", youtube_code="CAMSAjAB")


def fetch_video_meta_data_and_youtube_object(
    video_id: str,
) -> Optional[Tuple[VideoDataOutput, CaptionDataOutput, pytube.YouTube]]:

    video_url = f"https://www.youtube.com/watch?v={video_id}"

    youtube_object = pytube.YouTube(video_url)
    youtube_object.streams.all()

    caption_dict = dict()

    for caption_item in youtube_object.captions:
        caption_dict[f"{caption_item.code}"] = caption_item.xml_captions

    if (
        youtube_object.age_restricted is True
        or "en" not in caption_dict
        and "a.en" not in caption_dict
    ):

        raise Exception(
            f"Video {video_id} is age restricted or has no captions"
        )

    caption_dict = load_text_into_language_time_stamps(
        caption_dict=caption_dict
    )

    metadata_output = VideoDataOutput(
        watch_url=youtube_object.watch_url,
        embed_url=youtube_object.embed_url,
        age_restricted=youtube_object.age_restricted,
        title=youtube_object.title,
        length=youtube_object.length,
        views=youtube_object.views,
        author=youtube_object.author,
        channel_id=youtube_object.channel_id,
        channel_url=youtube_object.channel_url,
        description=youtube_object.description,
        keywords=youtube_object.keywords,
        thumbnail_url=youtube_object.thumbnail_url,
        publish_date=youtube_object.publish_date,
        video_id=video_id,
    )

    captions = CaptionDataOutput(captions_dict=caption_dict)

    return metadata_output, captions, youtube_object


def download_video_and_meta_data(
    video_id: str,
    target_directory: Union[str, pathlib.Path],
    resolution_identifier: str,
    sleep_duration: int = 1,
) -> VideoDownloaderObject:
    target_directory = (
        pathlib.Path(target_directory)
        if isinstance(target_directory, str)
        else target_directory
    )
    time.sleep(sleep_duration)
    output = fetch_video_meta_data_and_youtube_object(video_id=video_id)

    if output is None:
        return VideoDownloaderObject(success=False, video_id=video_id)
    else:
        metadata_output, caption_data, youtube_object = output

    requested_video_resolution_stream = (
        youtube_object.streams.get_by_resolution(
            resolution=resolution_identifier
        )
    )

    if requested_video_resolution_stream is None:
        logging.info(
            f"Can't find "
            f"{resolution_identifier} version of, "
            f"{video_id},"
            f"{youtube_object.streams}"
        )
        return VideoDownloaderObject(success=False, video_id=video_id)
    else:
        video_filepath = (
            target_directory / str(video_id) / f"{resolution_identifier}.mp4"
        )

        logging.info(
            f"Download "
            f"{resolution_identifier} version of, "
            f"{video_id},"
            f"{target_directory.as_posix()}/"
            f"{resolution_identifier}.mp4"
        )

        try:
            if not target_directory.exists():
                target_directory.mkdir(parents=True, exist_ok=True)

            if not video_filepath.exists():
                requested_video_resolution_stream.download(
                    output_path=video_filepath.parent.absolute().as_posix(),
                    filename=video_filepath.name,
                    max_retries=1,
                )

        except Exception:
            shutil.rmtree(target_directory.as_posix())
            # logging.exception(
            #     f"Video {video_id}, {target_directory.as_posix()} has gone boom, "
            #     f"will now delete this file"
            # )
            return VideoDownloaderObject(success=False, video_id=video_id)

        metadata_output.video_store_filepath = video_filepath.as_posix()

        meta_data_table = target_directory / str(video_id) / "meta.yaml"

        caption_table = target_directory / str(video_id) / "captions.json"

        if not meta_data_table.parent.exists():
            meta_data_table.parent.mkdir(parents=True, exist_ok=True)

        try:
            captions_dict = convert_keys_to_str(caption_data.captions_dict)
            save_json(filepath=caption_table, target_dict=captions_dict)
            # store meta data as yaml
            with open(meta_data_table, "w") as yaml_file:
                yaml.dump(metadata_output.__dict__, yaml_file)

            time.sleep(sleep_duration)

        except Exception:
            # logging.exception(
            #     f"Video {video_id}, {target_directory.as_posix()} has gone boom, "
            #     f"will now delete this file. Exception was {sys.exc_info()[0]}"
            # )
            return VideoDownloaderObject(success=False, video_id=video_id)

    return VideoDownloaderObject(success=True, video_id=video_id)


def search_for_video_ids(
    terms_string: str,
    sort_type: SortType,
    n: int = 100,
    sleep_duration: int = 1,
) -> List[str]:

    time.sleep(sleep_duration)
    try:
        url = f"https://www.youtube.com/results?search_query={terms_string}"
        f"&sp={sort_type.youtube_code}"
        url = recode_uri(url)
        html = urllib.request.urlopen(url)

        html = html.read().decode()

        terms = re.findall(pattern=r"watch\?v=(\S{11})", string=html)[:n]
    except Exception:
        # logging.exception(
        #     f"Couldn't find any search results for terms {terms_string}"
        # )
        terms = []

    return terms


def search_and_return_url(
    query_term: str,
    total_results_per_query: int,
) -> List[str]:

    video_ids = search_for_video_ids(
        terms_string=query_term,
        sort_type=SortKeyStringToCode.relevance,
        n=total_results_per_query,
    )

    return video_ids


def search_and_download(
    query_term,
    total_results_per_query,
    sleep_duration,
    target_dataset_dir,
    total_downloads_per_query,
    resolution_identifier,
):
    video_ids = search_and_return_url(
        query_term=query_term,
        total_results_per_query=total_results_per_query,
    )

    completed = 0
    for video_id in video_ids:

        try:
            download_video_and_meta_data(
                video_id=video_id,
                target_directory=target_dataset_dir,
                resolution_identifier=resolution_identifier,
                sleep_duration=sleep_duration,
            )
            completed += 1
        except Exception as e:
            logging.exception(
                f"Couldn't download video {video_id} for query {query_term}"
            )

        if completed >= total_downloads_per_query:
            break


def main(
    target_dataset_dir: str,
    resolution_identifier: str = "360p",
    num_workers: int = 8,
    seed: int = 2306,
    pool_type: str = PoolType.thread,
    total_downloads_per_query: int = 1,
    total_results_per_query: int = 200,
    sleep_duration: float = 1.0,
    starting_sample_idx: int = 0,
    ending_sample_idx: int = -1,
    query_terms: Optional[List[str]] = None,
):
    # check if text is being piped and if so, read it and assume its query terms
    if not sys.stdin.isatty() and query_terms is None:
        query_terms = sys.stdin.read().splitlines()

    target_dataset_dir = pathlib.Path(target_dataset_dir)

    if not target_dataset_dir.exists():
        target_dataset_dir.mkdir(parents=True, exist_ok=True)

    query_terms = query_terms[starting_sample_idx:ending_sample_idx]
    random.seed(seed)

    processor = (
        ThreadPoolExecutor
        if pool_type == PoolType.thread
        else ProcessPoolExecutor
    )

    with processor(max_workers=num_workers) as executor:
        with tqdm(total=len(query_terms)) as pbar:
            for output in executor.map(
                search_and_download,
                query_terms,
                [total_results_per_query] * len(query_terms),
                [sleep_duration] * len(query_terms),
                [target_dataset_dir] * len(query_terms),
                [total_downloads_per_query] * len(query_terms),
                [resolution_identifier] * len(query_terms),
            ):
                pbar.update(1)


if __name__ == "__main__":
    fire.Fire(main)
