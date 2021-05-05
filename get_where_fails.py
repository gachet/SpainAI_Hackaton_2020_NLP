import os
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm, trange

from transformers import PegasusForConditionalGeneration, PegasusTokenizer


def filter_predlist(predlist):
    newlist = []
    for pred in predlist:
        if pred not in newlist:
            newlist.append(pred)
    return newlist[:10]


def to_device(d, device):
    for k in d:
        d[k] = d[k].to(device)
    return d


class ErrorChecker:
    def __init__(self, predictions, targets, df):
        self.predictions = predictions
        self.targets = targets
        self.df = df
        self.scores = []
        self.data_dict = {}

    def __call__(self, save_name="val_with_scores.csv"):
        score = 0
        for pred, target in tqdm(
            zip(self.predictions, self.targets), desc="Computing scores"
        ):
            individual_dcg = self._individual_dcg(pred, target)
            self.scores.append(individual_dcg)
            score += individual_dcg
        total_score = score / len(self.predictions) * 100
        print(f"total score is {total_score}")
        self.df["score"] = self.scores
        self.df["predictions"] = self.predictions
        self.df.to_csv(save_name, index=False, header=True)

    def _individual_dcg(self, pred, target):
        if target in pred:
            idx = pred.index(target)
            return 1 / np.log2(idx + 2)
        else:
            return 0


def dcg(predictions, targets):
    """Computes Discounted Cumulative Gain for a list of predictions and targets"""
    targets = [t.replace("\n", "").lower() for t in targets]
    score = 0
    for pred, target in zip(predictions, targets):
        if target in pred:
            idx = pred.index(target)
            score += 1 / np.log2(idx + 2)
    return score / len(predictions) * 100


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_predictions(
    model, tokenizer, texts, batchsize=2, numseqs=30, device="cuda", maxlen=346
):
    model.to(device)
    model.eval()
    # texts = [f"summarize: {text}" for text in texts]
    predictions = []
    batches = chunks(texts, batchsize)
    for idx, batch in tqdm(enumerate(batches), desc="Getting predictions for DCG"):
        preds = []
        batch_encoded = tokenizer.batch_encode_plus(
            batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=maxlen,
        )
        batch_encoded = to_device(batch_encoded, device)
        with torch.no_grad():
            gen = model.generate(
                **batch_encoded, num_return_sequences=numseqs, num_beams=numseqs
            )
        for gen_out in gen:
            preds.append(
                tokenizer.decode(
                    gen_out.cpu().detach().numpy(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
        preds = chunks(preds, numseqs)
        preds = list(map(filter_predlist, preds))
        predictions.extend(preds)
        torch.cuda.empty_cache()
    return predictions


if __name__ == "__main__":
    val = pd.read_csv("val_0802.csv")
    texts = val["description"]
    tokenizer = PegasusTokenizer.from_pretrained("pegasus_0802_2")
    model = PegasusForConditionalGeneration.from_pretrained("pegasus_0802_2")
    predictions = get_predictions(model, tokenizer, texts)
    error_checker = ErrorChecker(predictions, val["name"], val)
    error_checker()
