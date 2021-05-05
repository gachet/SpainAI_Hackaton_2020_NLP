from argparse import ArgumentParser

import pandas as pd
import torch
from tqdm import tqdm

from romance import romance
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, MarianMTModel,
                          MarianTokenizer, T5ForConditionalGeneration,
                          T5Tokenizer)

translation_dict = {
    "en_de": "Helsinki-NLP/opus-mt-en-de",
    "de_en": "Helsinki-NLP/opus-mt-de-en",
    "es_en": "Helsinki-NLP/opus-mt-es-en",
    "en_es": "Helsinki-NLP/opus-mt-en-es",
    "en_fr": "Helsinki-NLP/opus-mt-en-fr",
    "fr_en": "Helsinki-NLP/opus-mt-fr-en",
    "en_it": "Helsinki-NLP/opus-mt-en-it",
    "it_en": "Helsinki-NLP/opus-mt-it-en",
}


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def to_device(d, device):
    for k in d:
        d[k] = d[k].to(device)
    return d


class Translator:
    def __init__(self, translation_code, device="cuda:0"):
        self.translation_code = translation_code
        self.device = device
        self.model = MarianMTModel.from_pretrained(
            translation_dict[self.translation_code]
        )
        self.tokenizer = MarianTokenizer.from_pretrained(
            translation_dict[self.translation_code]
        )

    def __call__(self, texts, batchsize):
        self.model.to(self.device)
        self.model.eval()
        batches = chunks(texts, batchsize)
        preds = []
        for batch in tqdm(
            batches,
            desc=f"Iterating over batches for translating {self.translation_code}",
        ):
            batch_encoded = self.tokenizer.batch_encode_plus(
                batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                # max_length=maxlen,
            )
            batch_encoded = to_device(batch_encoded, self.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **batch_encoded, num_return_sequences=1, num_beams=1
                )
            for gen_out in gen:
                preds.append(
                    self.tokenizer.decode(
                        gen_out.cpu().detach().numpy(),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                )
            torch.cuda.empty_cache()
        return preds


def whole_process(df, batchsize, different_langs=["de", "es", "fr"]):
    all_df_list = [df.copy()]
    for lang in different_langs:
        df2 = (
            df.copy()
            .sample(frac=1)
            .reset_index(drop=True)
            .drop_duplicates(subset=["name"], keep=False)
            .sample(frac=0.05)
            .reset_index(drop=True)
        )
        texts_en = df2["description"].tolist()
        translator_1 = Translator(translation_code=f"en_{lang}")
        translated_texts = translator_1(texts_en, batchsize)
        del translator_1
        translator_2 = Translator(translation_code=f"{lang}_en")
        new_en = translator_2(translated_texts, batchsize)
        new_df = pd.DataFrame({"description": new_en, "name": df2["name"].tolist()})
        all_df_list.append(new_df)
    return pd.concat(all_df_list)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--save_name", type=str, default="train_1803.csv")
    args = parser.parse_args()
    train_df = pd.read_csv("train_0802.csv")
    print("Augmenting data...")
    new_train_df = whole_process(train_df, args.bs)
    new_train_df.to_csv(args.save_name, header=True, index=False)
