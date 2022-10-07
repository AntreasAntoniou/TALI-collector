from dataclasses import dataclass
from typing import List

import numpy as np
import requests
import torch
from PIL import Image
from rich import print
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# model = model.to(torch.cuda.current_device())
model.eval()


@dataclass
class CLIPOutput:
    sorted_scores: List[float]
    sorted_query_texts: List[str]
    sorted_query_ids: List[int]
    sorted_args: List[int]
    reference_text: str


def get_scores(
    reference_text: str, query_text: List[str], query_ids: List[str]
) -> CLIPOutput:
    reference_inputs = processor(
        text=reference_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    query_inputs = processor(
        text=query_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    for key, value in reference_inputs.items():
        print(key, value.shape)
        # reference_inputs[key] = value[:, :77].to(model.device)

    for key, value in query_inputs.items():
        print(key, value.shape)
        # query_inputs[key] = value[:, :77].to(model.device)

    with torch.no_grad():
        reference_features = model.get_text_features(**reference_inputs)
        query_features = model.get_text_features(**query_inputs)
        reference_features = reference_features / reference_features.norm(
            p=2, dim=-1, keepdim=True
        )
        query_features = query_features / query_features.norm(p=2, dim=-1, keepdim=True)

    scores = reference_features @ query_features.T

    scores = scores[0].tolist()

    scores = [float(score) for score in scores if float(score) > 0.5]

    sorted_args = np.argsort(scores).tolist()

    sorted_args = [int(i) for i in reversed(sorted_args)]

    sorted_scores = [scores[i] for i in sorted_args]

    query_text_sorted = [query_text[i] for i in sorted_args]

    query_ids = [query_ids[i] for i in sorted_args]

    return CLIPOutput(
        sorted_scores=sorted_scores,
        sorted_args=sorted_args,
        reference_text=reference_text,
        sorted_query_texts=query_text_sorted,
        sorted_query_ids=query_ids,
    )
