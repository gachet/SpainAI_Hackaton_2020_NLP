import csv
import os
import pickle
import random

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from sentence_transformers import InputExample
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator

print(f"TORCH VERSION: {torch.__version__}")


def _read_textfile(file):
    with open(file, "r") as f:
        return f.readlines()


def _read_pickle(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def _get_dataloader(samples, batch_size):
    """Returns dataloader from samples list"""
    print("Cogiendo dataloader")
    return DataLoader(samples, shuffle=True, batch_size=batch_size)


def create_samples(descriptions, candidates_lists, labels):
    """Creates samples for training with CrossEncoder"""
    samples = []
    for description, candidate_list, label in tqdm(
        zip(descriptions, candidates_lists, labels)
    ):
        negative_examples = [
            candidate for candidate in candidate_list if candidate != label
        ]
        negative_examples = random.choices(negative_examples, k=4)
        if label in candidate_list:
            positive_example = [
                candidate for candidate in candidate_list if candidate == label
            ][0]
        else:
            positive_example = label
        samples.append(InputExample(texts=[description, positive_example], label=1))
        for neg_ex in negative_examples:
            samples.append(InputExample(texts=[description, neg_ex], label=0))
    return samples


def create_eval_samples(descriptions, candidates_lists, labels):
    dev_set = {}
    for i, candidates in tqdm(
        enumerate(candidates_lists), desc="Iterating over eval data"
    ):
        negative = [candidate for candidate in candidates if candidate != labels[i]]
        if labels[i] in candidates:
            positive = [candidate for candidate in candidates if candidate == labels[i]]
        else:
            positive = [labels[i]]
        dev_set[i] = {
            "query": descriptions[i],
            "positive": positive,
            "negative": negative,
        }
    return dev_set


def create_data(split="train", batchsize=32, checkpoint=None):
    descriptions = _read_textfile(f"./data/{split}.source")
    if checkpoint is not None:
        pkl_name = f"predictions_to_rank_{split}_checkpoint-{checkpoint}.pkl"
    else:
        pkl_name = f"predictions_to_rank_{split}.pkl"
    if pkl_name in os.listdir():
        candidates = _read_pickle(pkl_name)
        labels = _read_textfile(f"./data/{split}.target")
        if split == "train":
            samples = create_samples(descriptions, candidates, labels)
            return samples  # _get_dataloader(samples, batchsize)
        else:
            samples = create_eval_samples(descriptions, candidates, labels)
            return CERerankingEvaluator(samples, name="eval")
    else:
        pkl_name = f"predictions_to_rank_{split}_checkpoint-{checkpoint}.pkl"
        candidates = _read_pickle(pkl_name)
        labels = _read_textfile(f"./data/{split}.target")
        (
            candidates_tr,
            candidates_val,
            descriptions_tr,
            descriptions_val,
            labels_tr,
            labels_val,
        ) = train_test_split(candidates, descriptions, labels, test_size=0.1)
        samples_tr = create_samples(descriptions_tr, candidates_tr, labels_tr)
        samples_val = create_eval_samples(descriptions_val, candidates_val, labels_val)
        return samples_tr, CERerankingEvaluator(samples_val, name="eval")
