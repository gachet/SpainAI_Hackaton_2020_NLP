import csv
import os
from argparse import ArgumentParser

import pandas as pd
import torch
from tqdm import tqdm

from create_advanced_data import _save_pickle, files_add_train, whole_process
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_types = {
    "t5": {"model_cls": T5ForConditionalGeneration, "tokenizer_cls": T5Tokenizer},
    "pegasus": {
        "model_cls": PegasusForConditionalGeneration,
        "tokenizer_cls": PegasusTokenizer,
    },
    "bart": {"model_cls": BartForConditionalGeneration, "tokenizer_cls": BartTokenizer},
}


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _read_subm_file(file):
    submission = []
    with open(file, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            submission.append(line)
    submission = submission[1:]
    return submission


def _process_file_name(filename):
    return filename.replace("train", "test").replace(".pkl", ".csv")


def _transform_submission_file(filename, new_filename):
    subm_file = _read_subm_file(filename)
    _save_pickle(subm_file, new_filename)


def _files_to_special_files(filenames):
    # filenames = [file for file in os.listdir(files_dir)]
    filenames = list(map(_process_file_name, filenames))
    new_filenames = [file.replace(".csv", ".pkl") for file in filenames]
    for file, new_file in zip(filenames, new_filenames):
        _transform_submission_file(file, new_file)
    return new_filenames


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


def clean_text(text):
    text = text.replace("<br/>", " ")
    text = text.replace("<BR>", " ")
    text = text.replace("<BR/>", " ")
    text = text.replace("<br>", " ")
    text = text.replace("\xa0", " ")
    text = text.replace("•", " ")
    text = text.replace("&#39;", "'")
    text = text.replace("\n", " ")
    text = text.replace("[<>]", " ")
    clean_exprs = [
        "HEIGHT OF MODEL",
        "height of model",
        "model height",
        "MODEL HEIGHT",
        # "Contains: ",
        # "Heel height",
        # "Sole height",
        # "Height of sole",
        # "Height x Length x Width",
        # "Height x Width x Depth",
        "WARNING",
    ]
    for expr in clean_exprs:
        if expr in text:
            text = text[: text.find(expr)]
    text = text.replace(" +", " ")
    if text[-1] == " ":
        text = text[:-1]
    return text


def save_submission_df(predictions, filename="submission.csv"):
    di = {f"pred_{i}": [] for i in range(10)}
    for names in predictions:
        if len(names) != 10:
            print(names)
            print(len(names))
        for i in range(len(names)):
            di[f"pred_{i}"].append(names[i])
    df = pd.DataFrame(di)
    df.to_csv(filename, header=False, index=False)


def _get_model_tokenizer(model_type, model_name):
    model = model_types[model_type]["model_cls"].from_pretrained(model_name)
    tokenizer = model_types[model_type]["tokenizer_cls"].from_pretrained(model_name)
    return model, tokenizer


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Model to use")
    parser.add_argument(
        "--max_len",
        type=int,
        required=False,
        default=128,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--submission_name",
        type=str,
        required=False,
        default="submission.csv",
        help="Name for submission file",
    )
    parser.add_argument(
        "--model_type",
        required=True,
        type=str,
        help="The model architecture you're using: BART, Pegasus or T5.",
    )
    parser.add_argument(
        "--special",
        required=False,
        default=False,
        type=bool,
        help="Whether or not to use special data for this.",
    )
    args = parser.parse_args()
    test = pd.read_csv("test_descriptions.csv")
    test["description"] = test["description"].apply(clean_text)
    test_texts = test["description"].tolist()
    # model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    # tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    if args.special:
        new_filenames = _files_to_special_files(files_add_train)
        test_texts = whole_process(test_texts, new_filenames)
    # model = PegasusForConditionalGeneration.from_pretrained("pegasus_2501")
    # tokenizer = PegasusTokenizer.from_pretrained("pegasus_2501")
    # ~/ray_results/PBT_BART/_inner_0322a_00000_0_num_train_epochs\=4_2021-02-01_03-41-40/run-0322a_00000/checkpoint-2088/
    # model = BartForConditionalGeneration.from_pretrained("bart_0702")
    # tokenizer = BartTokenizer.from_pretrained("bart_0702")
    model, tokenizer = _get_model_tokenizer(args.model_type, args.model_name)
    predictions = get_predictions(
        model, tokenizer, test_texts, numseqs=30, device="cuda", maxlen=args.max_len
    )
    save_submission_df(predictions, args.submission_name)
