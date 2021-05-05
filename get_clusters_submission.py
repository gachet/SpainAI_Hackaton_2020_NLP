import csv
import os
import pickle

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from transformers import BartForConditionalGeneration, BartTokenizer

models_dir = "cluster_models"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def clean_text(text):
    text = text.replace("<br/>", "")
    text = text.replace("<br>", "")
    text = text.replace("\xa0", "")
    text = text.replace("•", "")
    text = text.replace("&#39;", "'")
    text = text.replace(" +", " ")
    text = text.replace("\n", "")
    clean_exprs = [
        "HEIGHT OF MODEL",
        "height of model",
        "model height",
        "MODEL HEIGHT",
        "Contains: ",
    ]
    for expr in clean_exprs:
        if expr in text:
            text = text[: text.find(expr)]
    return text


def get_embeddings(texts, encoder):
    with torch.no_grad():
        return encoder.encode(texts, show_progress_bar=True, device="cuda")


def apply_cluster_model(encoded_texts, cluster_model):
    return cluster_model.predict(encoded_texts)


def get_predictions(model, tokenizer, texts, numseqs=50, device="cuda:0"):
    model.to(device)
    predictions = []
    for text in tqdm(texts):
        preds = []
        text_encoded = tokenizer.encode(text, return_tensors="pt")
        # text_encoded = text_encoded.to(device)
        with torch.no_grad():
            gen = model.generate(
                text_encoded.to(device), num_return_sequences=numseqs, num_beams=numseqs
            )
        decoded = [
            tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for g in gen
        ]
        for d in decoded:
            if d not in preds:
                preds.append(d)
        decoded = preds[:10]
        predictions.append(decoded)
    return predictions


def save_submission_df(predictions, filename="submission_1603_clusters.csv"):
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["name"])
        writer.writerows(predictions)


def set_preds(df, indices, preds):
    for i in range(10):
        df.loc[indices, f"preds_{i}"] = [pred[i] for pred in preds]
    return df


if __name__ == "__main__":
    test = pd.read_csv("test_descriptions.csv")
    test["description"] = test["description"].apply(clean_text)
    encoder = SentenceTransformer("stsb-roberta-large", device="cuda")
    test_embeddings = get_embeddings(test["description"].tolist(), encoder)
    with open("kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)
    test["cluster"] = apply_cluster_model(test_embeddings, kmeans)
    # test["predictions"] = [[] for i in range(test.shape[0])]
    preds_df = pd.DataFrame({"index": range(test.shape[0])})
    for cluster in tqdm(test["cluster"].unique(), desc="iterating over clusters"):
        texts = test.loc[test["cluster"] == cluster, "description"].tolist()
        model = BartForConditionalGeneration.from_pretrained(
            f"{models_dir}/model_{cluster}"
        )
        tokenizer = BartTokenizer.from_pretrained(f"{models_dir}/model_{cluster}")
        predictions = get_predictions(model, tokenizer, texts)
        indices = test.loc[
            test["cluster"] == cluster, "description"
        ].index.values.tolist()
        preds_df = set_preds(preds_df, indices, predictions)
        del model
        del tokenizer
    # assert test["predictions"].isna().sum() == 0
    total_predictions = []
    cols_coger = [col for col in preds_df.columns if "preds_" in col]
    for i in range(test.shape[0]):
        total_predictions.append(preds_df.loc[i, cols_coger].tolist())
    save_submission_df(total_predictions)
