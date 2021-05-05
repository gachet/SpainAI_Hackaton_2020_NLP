import os
import pickle
import re

import pandas as pd
from tqdm import tqdm, trange


def _read(file):
    with open(file, "r") as f:
        return f.readlines()


def _load_pickle(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def _save_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def _save(texts, file):
    print(len(texts))
    with open(file, "w") as f:
        for text in tqdm(texts, desc="Saving texts"):
            f.write(f"{text}")


files_add_train = [
    "t5_0802_predictions_to_rank_train.pkl",
    "t5_2401_predictions_to_rank_train.pkl",
    "pegasus_0102_predictions_to_rank_train.pkl",
    "pegasus_0802_predictions_to_rank_train.pkl",
]


files_add_val = [
    "t5_0802_predictions_to_rank_val.pkl",
    "t5_2401_predictions_to_rank_val.pkl",
    "pegasus_0102_predictions_to_rank_val.pkl",
    "pegasus_0802_predictions_to_rank_val.pkl",
]

expr_remove = "_predictions_to_rank_*"

# suggestions must be in the form [n_texts, n_models, 10], that is,


def _create_suggestions(files_add):
    suggestions = []
    preds_dict = {
        file[: file.find("_predictions_to")]: _load_pickle(file) for file in files_add
    }
    for i in trange(len(preds_dict["pegasus_0102"])):
        models_suggestions = []
        for model in preds_dict:
            models_suggestions.extend(preds_dict[model][i])
        suggestions.append(models_suggestions)
    return suggestions


def _fix_suggestions(suggestions):
    return "</s><s>Preds:" + ",".join(suggestions)


def _create_tuples(texts, suggestions):
    return [(text, suggestion) for text, suggestion in zip(texts, suggestions)]


def whole_process(texts, files_add):
    suggestions = _create_suggestions(files_add)
    suggestions = list(map(_fix_suggestions, suggestions))
    tuples = _create_tuples(texts, suggestions)
    final_texts = list(map(lambda x: f"{x[0]}{x[1]}", tuples))
    return final_texts


if __name__ == "__main__":
    train_texts = _read("./datos_0802/train.source")
    train_names = _read("./datos_0802/train.target")
    val_texts = _read("./datos_0802/val.source")
    val_names = _read("./datos_0802/val.target")
    final_texts_tr = whole_process(train_texts, files_add_train)
    assert len(final_texts_tr) == 35032
    final_texts_val = whole_process(val_texts, files_add_val)
    assert len(final_texts_val) == 2651
    os.makedirs("special_data", exist_ok=True)
    _save(final_texts_tr, "special_data/train.source")
    _save(final_texts_val, "special_data/val.source")
