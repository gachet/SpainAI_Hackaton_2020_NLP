import random
import string

import pandas as pd
from nltk.corpus import stopwords

omitwords = [s for s in string.punctuation] + stopwords.words("english")


def _read(file):
    with open(file, "r") as f:
        return f.readlines()


def get_weights_and_n_remove(words, prop=0.15, minimum=2):
    """
    Gets weights for the sampling of the random words
    that will be masked. For that, it sets the weights
    of words belonging to punctuation to 0.
    Additionally, gets the number of words to be removed.
    """
    words_lower = [w.lower() for w in words]
    pos_omitwords = [(i, w) for i, w in enumerate(words_lower) if w in omitwords]
    pos_omitwords = [elem[0] for elem in pos_omitwords]
    divide = len(words) - len(pos_omitwords)
    weight = 1 / divide
    weights = [weight] * len(words_lower)
    for pos in pos_omitwords:
        weights[pos] = 0.0
    number_words = len(words) - len(pos_omitwords)
    n_remove = int(prop*number_words)
    if minimum is not None:
        n_remove = max(minimum, n_remove)
    return weights, n_remove


def replace_masks_with_extra_ids(sentence_list):
    """Replaces those places marked with mask with the corresponding extra ids token."""
    extra_idx = 0
    if sentence_list[0] == "<mask>":
        sentence_list[0] = f"<extra_id_{extra_idx}>"
        extra_idx += 1
    for i in range(1, len(sentence_list)):
        if sentence_list[i] == "<mask>":
            if "<extra_id_" in sentence_list[i - 1]:
                sentence_list[i] = sentence_list[i - 1]
            else:
                sentence_list[i] = f"<extra_id_{extra_idx}>"
                extra_idx += 1
    return sentence_list


def get_opposite_sentence(sentence_list, original):
    """Once the input sentence has been encoded, we can get the opposite for the decoder."""
    assert len(sentence_list) == len(original)
    opposite = []
    for i in range(len(sentence_list)):
        if "<mask>" in sentence_list[i]:
            opposite.append(original[i])
        else:
            opposite.append("<mask>")
    return opposite


def remove_duplicates(lista):
    """Removes extra ids which are duplicated, substituting them by a single one"""
    newlist = []
    for i in range(len(lista)):
        if "<extra_id_" not in lista[i]:
            newlist.append(lista[i])
        else:
            if lista[i] not in newlist:
                newlist.append(lista[i])
    return newlist


def get_definitive_list(lista):
    """Replaces masks with extra ids and remove duplicates"""
    lista = replace_masks_with_extra_ids(lista)
    lista = remove_duplicates(lista)
    return lista


def _to_str(l):
    """Transforms sentence in list format to string, and replace multiple spaces"""
    return " ".join(l).replace(" +", " ")


def perturb_sentence(s):
    """
    Perturb sentence in T5-fashion.
    First, it gets the weights for words sampling for masking, and the
    number of words to mask aswell. Then, it sets those word to <mask>.
    After that, masked tokens are replaced by extra_ids, concatenating
    those that appear together.
    With the masked sentence in list format (before substituting for
    extra_ids), we get the opposite sentence, which has words where
    masked sentence has <mask> and <mask> where masked sentence has
    words. Then, those masks in the opposite sentence (which will be
    the labels), are also transformed to extra_ids.

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
    weights, n_remove = get_weights_and_n_remove(words)
    words_mask = random.choices(words, weights=weights, k=n_remove)
    indexes_mask = []
    search = words.copy()
    for i in range(len(words_mask)):
        word_to_mask = words_mask[i]
        idx = search.index(word_to_mask)
        # search = search[idx + 1:]
        indexes_mask.append(idx)
    new_sentence = words.copy()
    for idx in indexes_mask:
        new_sentence[idx] = "<mask>"
    opposite_sentence = get_opposite_sentence(new_sentence, words)
    final_sentence = get_definitive_list(new_sentence)
    final_opposite = get_definitive_list(opposite_sentence)
    return _to_str(final_sentence), _to_str(final_opposite)


def get_features_df(file):
    texts = _read(file)
    text_pairs = list(map(perturb_sentence, texts))
    df = pd.DataFrame(
        {
            "sentence": [pair[0] for pair in text_pairs],
            "labels": [pair[1] for pair in text_pairs],
        }
    )
    return df
