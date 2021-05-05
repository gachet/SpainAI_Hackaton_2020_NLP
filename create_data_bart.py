import os

import pandas as pd

from utils_bart_perturbation import get_features_df

if __name__ == "__main__":
    df_tr = get_features_df("./data_2401/train.source")
    df_val = get_features_df("./data_2401/val.source")
    df_test = get_features_df("test_descriptions.csv", test=True)
    df_tr_total = pd.concat([df_tr, df_test])
    os.makedirs("perturbed_bart_data", exist_ok=True)
    df_tr_total.to_csv("perturbed_bart_data/tr.csv", header=True, index=False)
    df_val.to_csv("perturbed_bart_data/val.csv", header=True, index=False)
