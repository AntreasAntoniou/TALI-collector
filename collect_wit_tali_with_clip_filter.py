import argparse
import concurrent.futures
import logging
import pathlib
import re
import shutil
import sys
import time
import traceback
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import orjson as json
import pyarrow as pa
import pyarrow.parquet as pq
import pytube
import tqdm
from datasets import load_dataset
from rich import print
from rich.logging import RichHandler
from rich.traceback import install
from yelp_uri.encoding import recode_uri

from clip_helper import get_scores
from utils import convert_keys_to_str, load_text_into_language_time_stamps, save_json

install()

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
    parser.add_argument("--target_language", type=str, default="en")
    parser.add_argument("--samples_per_bucket", type=int, default=20000)
    parser.add_argument("--wit_cache_dir", type=str, default="/mnt/nas/datasets/")
    parser.add_argument("--clip_cutoff", type=float, default=0.5)

    parser = parser.parse_args()

    return parser


args = get_base_argument_parser()
print("Starting TALI-WIT collection 🦾")

score_table_folderpath = pathlib.Path(args.target_dataset_dir) / "score_table.parquet"
dataset_table_folderpath = pathlib.Path(args.target_dataset_dir) / "dataset.parquet"
captions_table_folderpath = pathlib.Path(args.target_dataset_dir) / "captions"


if not score_table_folderpath.exists():
    score_table_folderpath.mkdir(parents=True, exist_ok=True)

if not dataset_table_folderpath.exists():
    dataset_table_folderpath.mkdir(parents=True, exist_ok=True)

if not captions_table_folderpath.exists():
    captions_table_folderpath.mkdir(parents=True, exist_ok=True)


class PoolType:
    process: str = "Process"
    thread: str = "Thread"


useful_keys = [
    "caption_reference_description",
    "page_title",
    # "context_page_description",
    # "hierarchical_section_title",
]


@dataclass
class TargetLanguageOutput:
    target_language_dict: Dict


def get_language_specific_wikipedia_data(
    sample: Dict, target_language: str = "en"
) -> TargetLanguageOutput:
    multi_lingual_wikipedia_data = sample["wit_features"]

    if target_language in multi_lingual_wikipedia_data["language"]:
        language_idx = multi_lingual_wikipedia_data["language"].index(target_language)
        target_language_dict = {}
        for key, value in multi_lingual_wikipedia_data.items():
            target_language_dict[key] = value[language_idx]
        return TargetLanguageOutput(target_language_dict=target_language_dict)
    else:
        return None


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


def download_video_meta_data_and_youtube_object(
    video_id: str, target_directory: Union[str, pathlib.Path]
) -> Tuple[VideoDataOutput, CaptionDataOutput, pytube.YouTube]:

    try:
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        youtube_object = pytube.YouTube(video_url)

        caption_dict = dict()

        for caption_item in youtube_object.captions:
            caption_dict[f"{caption_item.code}"] = caption_item.xml_captions

        if (
            youtube_object.age_restricted is True
            or "en" not in caption_dict
            and "a.en" not in caption_dict
        ):
            return None

        caption_dict = load_text_into_language_time_stamps(caption_dict=caption_dict)

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
    except Exception:

        # logging.exception(
        #     f"Video {video_url}, {target_directory.as_posix()} has gone boom, "
        #     f"will now delete this file"
        # )

        return None


@dataclass
class VideoDownloaderObject:
    success: bool
    video_id: str


@dataclass
class SortType:
    name: str
    youtube_code: str


def download_video_and_meta_data(
    video_id: str,
    wit_idx: int,
    term_idx: int,
    sort_type: SortType,
    target_directory: Union[str, pathlib.Path],
    resolution_identifier: str,
    sleep_duration: int,
) -> VideoDownloaderObject:
    target_directory = (
        pathlib.Path(target_directory)
        if isinstance(target_directory, str)
        else target_directory
    )

    output = download_video_meta_data_and_youtube_object(
        video_id=video_id, target_directory=target_directory
    )

    if output is None:
        return VideoDownloaderObject(success=False, video_id=video_id)
    else:
        metadata_output, caption_data, youtube_object = output

    requested_video_resolution_stream = youtube_object.streams.get_by_resolution(
        resolution=resolution_identifier
    )

    if requested_video_resolution_stream is None:
        # logging.info(
        #     f"Can't find "
        #     f"{resolution_identifier} version of, "
        #     f"{video_id},"
        #     f"{youtube_object.streams}"
        # )
        return VideoDownloaderObject(success=False, video_id=video_id)
    else:
        video_filepath = target_directory / f"{resolution_identifier}.mp4"

        # logging.info(
        #     f"Download "
        #     f"{resolution_identifier} version of, "
        #     f"{video_id},"
        #     f"{target_directory.as_posix()}/"
        #     f"{resolution_identifier}.mp4"
        # )

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

        meta_data_table = (
            dataset_table_folderpath
            / f"{sort_type.name}/{wit_idx}/{term_idx}/{video_id}/meta.parquet"
        )

        caption_table = (
            captions_table_folderpath
            / f"{sort_type.name}/{wit_idx}/{term_idx}/{video_id}/captions.json"
        )

        if not meta_data_table.parent.exists():
            meta_data_table.parent.mkdir(parents=True, exist_ok=True)
        try:
            keys = metadata_output.__dict__.keys()
            sorted_keys = sorted(keys)

            combined_values = [
                [wit_idx],
                [term_idx],
                [sort_type.name],
            ] + [[metadata_output.__dict__[key]] for key in sorted_keys]

            combined_keys = [
                "wit_idx",
                "term_idx",
                "sort_type",
            ] + sorted_keys

            table_entry = pa.table(
                combined_values,
                names=combined_keys,
            )

            pq.write_table(table_entry, meta_data_table)

            captions_dict = convert_keys_to_str(caption_data.captions_dict)
            save_json(filepath=caption_table, target_dict=captions_dict)

            time.sleep(sleep_duration)
            # logging.info(f" for {sleep_duration} seconds..")
        except Exception:
            # logging.exception(
            #     f"Video {video_id}, {target_directory.as_posix()} has gone boom, "
            #     f"will now delete this file. Exception was {sys.exc_info()[0]}"
            # )
            return VideoDownloaderObject(success=False, video_id=video_id)

    return VideoDownloaderObject(success=True, video_id=video_id)


@dataclass
class DownloadLength:
    video_id: str
    length: int


@dataclass
class SortKeyStringToCode:
    relevance = SortType(name="relevance", youtube_code="CAASAjAB")
    view_counts = SortType(name="view_counts", youtube_code="CAMSAjAB")


def search_for_video_ids(
    terms_string: str, sort_type: SortType, n: int = 100
) -> List[str]:

    try:
        url = f"https://www.youtube.com/results?search_query={terms_string}"
        f"&sp={sort_type.youtube_code}"
        url = recode_uri(url)
        html = urllib.request.urlopen(url)

        html = html.read().decode()

        terms = re.findall(pattern=r"watch\?v=(\S{11})", string=html)[:n]
    except Exception:
        # logging.exception(f"Couldn't find any search results for terms {terms_string}")
        terms = []

    return terms


def extract_terms_dict_from_sample(sample: Dict) -> Dict[str, list]:
    term_dict = defaultdict(list)
    output_dict = get_language_specific_wikipedia_data(
        sample=sample, target_language=args.target_language
    )
    if output_dict is not None:
        output_dict = filter_dict_by_keys(
            input_dict=output_dict.target_language_dict, filter_keys=useful_keys
        )
    else:
        output_dict = dict()

    for key, value in output_dict.items():
        if value is not None:
            terms = re.findall("[A-Z][^A-Z]*", value)

            term_string = "+".join(terms)
            term_dict[key].append(term_string)

    return term_dict


@dataclass
class SearchOutput:
    search_idx: int
    search_query: Dict[str, Any]


def search_and_return_url(
    search_idx: int, search_query: Dict[str, List], total_results_per_query: int
):
    url_list = set()
    for query_type, queries in search_query.items():
        for sort_type in [
            SortKeyStringToCode.relevance,
        ]:
            for query in queries:
                url_idxs = search_for_video_ids(
                    terms_string=query, sort_type=sort_type, n=total_results_per_query
                )
                url_list.add(url_idxs)
    search_query["url_list"] = list(url_list)

    return SearchOutput(search_idx=search_idx, search_query=search_query)


def filter_video_ids_with_clip(
    reference_term: str,
    term_related_video_ids: List[str],
    target_directory: Union[pathlib.Path, str],
    wit_idx: int,
    term_idx: int,
    sort_type: SortType,
):
    titles = []
    directory_path = (
        pathlib.Path(target_directory)
        if isinstance(target_directory, str)
        else target_directory
    )
    directory_path = directory_path / str(wit_idx)
    valid_video_ids_related_to_term = []

    for video_id in term_related_video_ids:
        video_directory_path = directory_path / video_id

        output = download_video_meta_data_and_youtube_object(
            video_id=video_id, target_directory=video_directory_path
        )

        if output is not None:
            video_dict, captions_dict, youtube_object = output
            titles.append(youtube_object.title)
            valid_video_ids_related_to_term.append(video_id)

    if len(titles) > 0:
        clip_scores = get_scores(
            reference_text=reference_term,
            query_text=titles,
            query_ids=valid_video_ids_related_to_term,
            cutoff=args.clip_cutoff,
        )

        if len(clip_scores.sorted_args) > 0:

            entry_filepath = (
                score_table_folderpath
                / f"{sort_type.name}/{wit_idx}/{term_idx}.parquet"
            )

            if not entry_filepath.parent.exists():
                entry_filepath.parent.mkdir(parents=True, exist_ok=True)

            table_entry = pa.table(
                [
                    [clip_scores.reference_text],
                    [wit_idx],
                    [term_idx],
                    [sort_type.name],
                    [clip_scores.sorted_query_texts],
                    [clip_scores.sorted_scores],
                    [clip_scores.sorted_query_ids],
                    [clip_scores.sorted_args],
                ],
                names=[
                    "reference_text",
                    "wit_idx",
                    "term_idx",
                    "sort_type",
                    "sorted_query_texts",
                    "sorted_scores",
                    "sorted_query_ids",
                    "sorted_args",
                ],
            )

            table = table_entry

            pq.write_table(table, entry_filepath)

            return clip_scores.sorted_query_ids[: args.total_downloads_per_query]

    return []


def download_video_meta_data_given_sample(
    sample: Dict, wit_idx: int, target_directory: Union[str, pathlib.Path]
):
    # try:
    target_directory = (
        pathlib.Path(target_directory)
        if isinstance(target_directory, str)
        else target_directory
    )
    outputs = []
    search_term_dict = extract_terms_dict_from_sample(sample=sample)
    target_directory = pathlib.Path(target_directory)
    for term_idx, (term_name, term_values) in enumerate(search_term_dict.items()):
        for sort_type in [
            SortKeyStringToCode.relevance,
        ]:
            term_related_video_ids = search_for_video_ids(
                terms_string=term_values,
                n=args.total_results_per_query,
                sort_type=sort_type,
            )
            term_related_video_ids = list(set(term_related_video_ids))

            term_related_video_ids = filter_video_ids_with_clip(
                reference_term=term_values,
                term_related_video_ids=term_related_video_ids,
                target_directory=target_directory,
                wit_idx=wit_idx,
                term_idx=term_idx,
                sort_type=sort_type,
            )

            for video_id in term_related_video_ids:
                video_directory_path = target_directory / str(wit_idx) / str(video_id)

                output = download_video_and_meta_data(
                    video_id=video_id,
                    term_idx=term_idx,
                    wit_idx=wit_idx,
                    sort_type=sort_type,
                    target_directory=video_directory_path,
                    resolution_identifier=args.resolution_identifier,
                    sleep_duration=args.sleep_duration,
                )
                outputs.append(output)
    # except Exception:
    #     outputs = []
    #     print(f"Error in {wit_idx}, with exception {traceback.format_exc()}")
    return outputs


def download_video_meta_data_given_sample_wrapper(args_dict):
    return download_video_meta_data_given_sample(**args_dict)


def download_dataset_given_ids(
    set_ids: List[int], dataset: Any, target_directory: Union[str, pathlib.Path]
):
    target_directory = (
        pathlib.Path(target_directory)
        if isinstance(target_directory, str)
        else target_directory
    )

    arg_dict_list = [
        dict(
            sample=dataset[wit_idx],
            wit_idx=wit_idx,
            target_directory=target_directory,
        )
        for wit_idx in set_ids
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
    # logging.info(f"Using {pool_type} for parallel processing")
    with tqdm.tqdm(total=len(set_ids), smoothing=0.0) as pbar:

        with pool_type(max_workers=args.num_threads) as executor:
            for video_idx, outputs_list in enumerate(
                executor.map(
                    download_video_meta_data_given_sample_wrapper, arg_dict_list
                ),
                start=1,
            ):

                pbar.update(1)
                pbar.set_description(f"Downloaded videos with ids {outputs_list}")


if __name__ == "__main__":
    print("Loading WIT dataset 👨🏻‍💻")
    dataset = load_dataset(
        "wikimedia/wit_base", split="train", cache_dir=args.wit_cache_dir
    )
    # random.seed(args.seed)
    dataset_ids = [
        i for i in range(args.starting_sample_idx, args.ending_sample_idx, 1)
    ]

    id_buckets = []
    num_buckets = (len(dataset_ids) // args.samples_per_bucket) + 1
    print("Splitting dataset into buckets 🪣")
    for i in range(num_buckets):
        id_buckets.append(
            dataset_ids[i * args.samples_per_bucket : (i + 1) * args.samples_per_bucket]
        )

    set_ids = dataset_ids
    dataset_directory = pathlib.Path(args.target_dataset_dir) / "all"
    dataset_directory.mkdir(parents=True, exist_ok=True)
    print("Beginning scraping 📔")
    with tqdm.tqdm(total=num_buckets, smoothing=0.0) as pbar:
        for id_bucket in id_buckets:
            download_dataset_given_ids(
                set_ids=id_bucket, dataset=dataset, target_directory=dataset_directory
            )
            pbar.update(1)
