import json
import os
import pickle
import random
from argparse import ArgumentParser
from typing import Dict

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=10, early_stopping_threshold=0.005
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_PATH = "bert-base-cased"

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, use_fast=True, model_max_length=512
)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class RankerDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.encodings.update({"label": torch.tensor(labels)})

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def _read_textfile(file):
    with open(file, "r") as f:
        return f.readlines()


def _read_pickle(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def save_json(content, file, indent=4):
    with open(file, "w") as f:
        json.dump(content, f, indent=indent, sort_keys=True)


def encode(examples):
    return tokenizer(
        str(examples["text1"]),
        str(examples["text2"]),
        truncation=True,
        padding="max_length",
    )


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
        samples.append({"text1": description, "text2": positive_example, "labels": 1})
        for neg_ex in negative_examples:
            samples.append({"text1": description, "text2": neg_ex, "labels": 0})
    df = pd.DataFrame(samples)

    # samples = {k: list(v) for k, v in zip(df.columns, df._iter_column_arrays())}
    # samples["labels"] = [int(lab) for lab in samples["labels"]]
    # final_samples = {"data": samples}
    return df


def create_data(split="train", checkpoint=None):
    descriptions = _read_textfile(f"./data/{split}.source")
    if checkpoint is not None:
        pkl_name = f"predictions_to_rank_{split}_checkpoint-{checkpoint}.pkl"
    else:
        pkl_name = f"predictions_to_rank_{split}.pkl"
    if pkl_name in os.listdir():
        candidates = _read_pickle(pkl_name)
        labels = _read_textfile(f"./data/{split}.target")
        samples = create_samples(descriptions, candidates, labels)
        return samples
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
        samples_val = create_samples(descriptions_val, candidates_val, labels_val)
        return samples_tr, samples_val


def compute_metrics_function(pred: EvalPrediction) -> Dict:
    preds, labels = pred.predictions, pred.label_ids
    preds = np.argmax(preds, axis=1)
    # print(preds[:5])
    # print(labels[:5])
    # print(type(labels))
    # print(type(preds))
    metrics = {
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "accuracy": accuracy_score(labels, preds),
    }
    return metrics


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_dir", required=True, type=str, help="")
    parser.add_argument("--lr", required=False, default=5e-5, type=float, help="")
    parser.add_argument(
        "--grad_acc_steps", required=False, default=2, type=int, help=""
    )
    parser.add_argument(
        "--train_batch_size", required=False, default=12, type=int, help=""
    )
    parser.add_argument(
        "--eval_batch_size", required=False, default=128, type=int, help=""
    )
    parser.add_argument("--epochs", required=False, default=5, type=int, help="")
    parser.add_argument("--logdir", required=True, type=str, help="")
    parser.add_argument("--log_steps", required=False, default=500, type=int, help="")
    args = parser.parse_args()
    samples_tr = create_data(split="train")
    # samples_tr = samples_tr.sample(100).reset_index(drop=True)
    # samples_tr.to_csv("samples_train_ranker.csv", header=False, index=False)
    samples_val = create_data(split="val")
    # samples_val = samples_val.sample(20).reset_index(drop=True)
    descriptions_tr, descriptions_val = (
        samples_tr["text1"].tolist(),
        samples_val["text1"].tolist(),
    )
    names_tr, names_val = samples_tr["text2"].tolist(), samples_val["text2"].tolist()
    print("Encoding!!")
    train_encodings = tokenizer.batch_encode_plus(
        [(text1, text2) for text1, text2 in zip(descriptions_tr, names_tr)],
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    val_encodings = tokenizer.batch_encode_plus(
        [(text1, text2) for text1, text2 in zip(descriptions_val, names_val)],
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    # print(samples_val.shape)
    # samples_tr.to_csv("samples_val_ranker.csv", header=False, index=False)
    # _save_json(samples_tr, "samples_ranker_train.json")
    # _save_json(samples_val, "samples_ranker_val.json")
    # dataset_train = load_dataset(
    #    # "json", data_files="samples_ranker_train.json", field="data"
    #    "csv",
    #    data_files="samples_train_ranker.csv",
    # )
    # dataset_val = load_dataset("csv", data_files="samples_val_ranker.csv")
    # dataset_train = dataset_train.map(encode)
    # dataset_val = dataset_val.map(encode)
    # dataset_train.set_format(
    #    type="torch",
    #    columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
    # )
    # dataset_val.set_format(
    #    type="torch",
    #    columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
    # )
    print("Creating dataset")
    dataset_train = RankerDataset(train_encodings, samples_tr["labels"].tolist())
    dataset_val = RankerDataset(val_encodings, samples_val["labels"].tolist())
    trainargs = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_acc_steps,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        logging_dir=args.logdir,
        logging_steps=args.log_steps,
        save_steps=args.log_steps,
        fp16=True,
        fp16_backend="apex",
        # dataloader_num_workers=16,
        load_best_model_at_end=True,
    )  # rellenar con TrainingArguments
    trainer = Trainer(
        model=model,
        args=trainargs,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_function,
        callbacks=[early_stopping],
    )
    print("***************** TRAINING *********************")
    trainer.train()

    trainer.save_model()
    trainer.state.save_to_json(os.path.join(args.output_dir, "trainer_state.json"))

    metrics = trainer.evaluate()
    save_json(metrics, f"metrics_{args.output_dir}.json")
    print(metrics)
