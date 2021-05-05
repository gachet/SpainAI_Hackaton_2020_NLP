import random

import numpy as np
import pandas as pd

from get_submission import clean_text
from utils_perturbation import _read, _to_str, get_weights_and_n_remove


def replace_multiple_masks(lista):
    newlist = []
    for i in range(len(lista)):
        if i == 0:
            newlist.append(lista[0])
        else:
            if lista[i] == "<mask>":
                if lista[i - 1] != "<mask>":
                    newlist.append(lista[i])
            else:
                newlist.append(lista[i])
    return newlist


def _get_minimum_mask(length):
    if length < 10:
        return 1
    else:
        return 2


def perturb_sentence(s):
    """
    Perturb sentence in BART-fashion.
    First, it gets the weights for words sampling for masking, and the
    number of words to mask aswell. Then, it sets those word to <mask>.
    After that, masked tokens are replaced by extra_ids, concatenating
    those that appear together.

    Parameters
    ----------
    s: str
        Sentence to perturb.

    Returns
    -------
    new_sentence: str
        Sentence perturbed.
    opposite_sentence: str
        Labels of perturbed sentence.
    """
    s = s.replace("\n", "")
    new_sentence = []
    words = s.split(" ")
    weights, n_remove = get_weights_and_n_remove(
        words, minimum=_get_minimum_mask(len(words))
    )
    words_mask = random.choices(words, weights=weights, k=n_remove)
    spans_lengths = np.random.poisson(lam=3.0, size=n_remove)
    if len(words) <= 20:
        spans_lengths = [min(4, a) for a in spans_lengths]
    indexes_mask = []
    search = words.copy()
    for i in range(len(words_mask)):
        word_to_mask = words_mask[i]
        idx = search.index(word_to_mask)
        allidx = list(set([idx] + [idx + i_ for i_ in range(spans_lengths[i])]))
        allidx = filter(lambda x: x <= len(words) - 1, allidx)
        indexes_mask.extend(allidx)
    new_sentence = words.copy()
    # print(f"Proportion: {len(set(indexes_mask)) / len(new_sentence)}")
    for idx in set(indexes_mask):
        new_sentence[idx] = "<mask>"
    new_sentence = replace_multiple_masks(new_sentence)
    return _to_str(new_sentence), s


def _filter_func(x):
    if not all([x_ == "<mask>" for x_ in x[0].split(" ")]) and not all(
        [x_ != "<mask>" for x_ in x[0].split(" ")]
    ):
        return x


def get_features_df(file, test=False):
    texts = _read(file)
    if test:
        texts = texts[1:]
        texts = list(map(clean_text, texts))
    text_pairs = list(map(perturb_sentence, texts))
    text_pairs = list(filter(_filter_func, text_pairs))
    df = pd.DataFrame(
        {
            "sentence": [pair[0] for pair in text_pairs],
            "labels": [pair[1] for pair in text_pairs],
        }
    )
    return df
