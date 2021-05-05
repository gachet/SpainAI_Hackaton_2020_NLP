import csv
import os
import pickle

import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm

from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

MODELNAME = "t5_2401_2"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def to_device(d, device):
    for k in d:
        d[k] = d[k].to(device)
    return d


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


"""
def filter_predictions(predictions):
    newpreds = []
    for n in predictions:
        preds = []
        if n not in preds:
            preds.append(n)
        newpreds.append(preds[:10])
    return newpreds
"""


def filter_predlist(predlist):
    newlist = []
    for pred in predlist:
        if pred not in newlist:
            newlist.append(pred)
    return newlist[:10]


def create_dataloader(texts, batchsize, tokenizer):
    input_ids = [
        tokenizer.encode(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=346,
        )
        for text in tqdm(texts, desc="Tokenizing texts...")
    ]
    input_ids = torch.stack(input_ids)
    dataset = TensorDataset(input_ids)
    return DataLoader(
        dataset, batch_size=batchsize, shuffle=False, num_workers=40, drop_last=False
    )


def get_predictions(
    model,
    tokenizer,
    texts,
    batchsize=2,
    numseqs=30,
    device="cuda:0",
    checkpoint_freq=1000,
    split="train",
):
    model.to(device)
    predictions = []
    batches = chunks(texts, batchsize)
    for idx, batch in tqdm(enumerate(batches), desc="Getting predictions for DCG"):
        preds = []
        batch_encoded = tokenizer.batch_encode_plus(
            batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=346,
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
        preds = chunks(preds, numseqs)  # [preds[:numseqs], preds[numseqs:]]
        preds = list(map(filter_predlist, preds))
        predictions.extend(preds)
        torch.cuda.empty_cache()
        if idx % checkpoint_freq == 0:
            save_predictions(
                predictions,
                filename=f"t5_2401_predictions_to_rank_{split}_checkpoint-{idx}",
            )
    return predictions


"""
def get_predictions(model, tokenizer, texts, batchsize=8, numseqs=30, device="cuda"):
    model = torch.nn.DataParallel(model)
    model.to(device)
    predictions = []
    print("Creating dataloader")
    dataloader = create_dataloader(texts, batchsize, tokenizer)
    # batches = chunks(texts, batchsize)
    for batch in tqdm(dataloader, desc="Getting predictions"):
        preds = []
        # batch_encoded = tokenizer.batch_encode_plus(
        #    batch, return_tensors="pt", padding=True, truncation=True
        # )
        # batch_encoded = to_device(batch_encoded, device)
        # text_encoded = text_encoded.to(device)
        # print(batch)
        # input_ids = batch.input_ids.to(device)
        # batch.to(device)
        input_ids = batch[0]
        input_ids.to(device)
        with torch.no_grad():
            gen = model.generate(
                input_ids, num_return_sequences=numseqs, num_beams=numseqs
            )
        for gen_out in gen:
            preds.append(
                [
                    tokenizer.decode(
                        g.cpu().detach().numpy(),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    for g in gen_out
                ]
            )
        preds = filter_predictions(preds)
        predictions.extend(preds)
        torch.cuda.empty_cache()
    return predictions
"""


def load_data(filename="./data/train.source"):
    with open(filename, "r") as f:
        return f.readlines()


def save_predictions(predictions, filename="t5_2401_predictions_to_rank_val"):
    try:
        with open(f"{filename}.pkl", "wb") as f:
            pickle.dump(predictions, f)
    except:
        with open(f"{filename}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(predictions)


if __name__ == "__main__":
    train_texts = load_data("./datos_0802/val.source")
    # model = BartForConditionalGeneration.from_pretrained(f"./{MODELNAME}")
    # tokenizer = BartTokenizer.from_pretrained(f"./{MODELNAME}")
    model = T5ForConditionalGeneration.from_pretrained(MODELNAME)
    tokenizer = T5Tokenizer.from_pretrained(MODELNAME)
    # tokenizer = PegasusTokenizer.from_pretrained(MODELNAME)
    # model = PegasusForConditionalGeneration.from_pretrained(MODELNAME)
    predictions = get_predictions(model, tokenizer, train_texts, split="val")
    save_predictions(predictions)
