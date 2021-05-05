import csv
import os
import re
from argparse import ArgumentParser

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# TODO: IMPLEMENTAR POSIBLE IDEA: EN LOS TEXTOS DEMASIADO LARGOS COMO ESTE:
# Set of 3 sterling silver rings, 2 rings gold plated in 2 microns and 1 ring with rhodium. Can be worn separately.This item is made with 925 sterling silver and plated in 24kt gold and rhodium. The thickness of the gold layer reaches 2 microns.The item has passed all internal quality control tests and all other mandatory tests in accordance with the applicable regulations during the manufacturing process. This item bears the mandatory hallmark in accordance with the applicable regulations, that is, the hallmark indicating origin and guarantee. Where this has not been possible, due to reduced size or specific design of the item, it has a tag bearing the aforementioned hallmark. The laboratory that has provided this hallmark is Ensayos y Contraste de Metales Preciosos de Andalucía, S.L.U. (ECOMEP), guarantee mark 925 A2.The approximate weight of the item is 7.95 g. Composition of the item: 100% silver. Recommendations for use: For greater durability of the product, we recommend that you do not spray perfume directly onto it. We also recommend that you clean the product with a cloth on a regular basis. You will find more information on the guarantee of products and after sales support in the Conditions of use and purchase and in the Shopping Guide and Help sections of the platform. You will also find extended information on the label of the item, the accompanying leaflet and the ZARA sales channels. For the daily price of the precious metals used in the items, please consult the relevant sources of information. In any case, the rights recognised by the legislation in force are not affected.
# PROBAR A PASARLE PRIMERO UN MODELO DE RESUMEN YA ENTRENADO (XSUM) PARA ELIMINAR INFORMACIÓN INNECESARIA.


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
    return text


def get_splits(data):
    tr, val = train_test_split(train, test_size=0.15)
    tr = pd.concat([tr, val[val["name"].isin(tr["name"])]])
    val = val[~val["name"].isin(tr["name"])]
    return tr, val


def add_zara_to_train(df, zara_file, home=False):
    if not home:
        zara = pd.read_csv(zara_file)
    else:
        zara = pd.read_csv(zara_file, sep=";")
    zara["description"] = zara["description"].apply(clean_text)
    # zara = zara[~zara["name"].isin(df["name"])]
    df_total = pd.concat([df, zara])
    df_total.drop_duplicates(["name", "description"], inplace=True)
    return df_total


def save_one_file(file, df, column):
    with open(file, "w") as f:
        for text in df[column]:
            f.write(f"{text}\n")


def save_data(train, val, datadir):
    os.makedirs(datadir, exist_ok=True)
    save_one_file(f"{datadir}/train.source", train, "description")
    save_one_file(f"{datadir}/train.target", train, "name")
    save_one_file(f"{datadir}/val.source", val, "description")
    save_one_file(f"{datadir}/val.target", val, "name")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--zara_file",
        type=str,
        help="Zara file.",
        required=False,
        default="zara_total10.csv",
    )
    parser.add_argument(
        "--zarahome_file",
        type=str,
        help="Zara Home file.",
        required=False,
        default="datos_zarahome2.csv",
    )
    parser.add_argument("--data_dir", type=str, help="Dir to save data", required=False)
    args = parser.parse_args()
    train = pd.read_csv("train.csv")
    print("Applying cleaning")
    train["description"] = train["description"].apply(clean_text)
    print("Getting splits")
    tr, val = get_splits(train)
    print("Adding zara to train and removing duplicates")
    tr = add_zara_to_train(tr, args.zara_file)
    tr = add_zara_to_train(tr, args.zarahome_file, home=True)
    tr = pd.concat([tr, val[val["name"].isin(tr["name"])]])
    val = val[~val["name"].isin(tr["name"])]
    print("Saving data...")
    # save_data(tr, val, args.data_dir)
    tr.to_csv("train_2903.csv", header=True, index=False)
    val.to_csv("val_2903.csv", header=True, index=False)
