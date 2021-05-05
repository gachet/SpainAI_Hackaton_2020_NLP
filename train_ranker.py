import os
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from process_data_for_ranker_trainer import create_data
from sentence_transformers.cross_encoder import CrossEncoder

print(f"TORCH VERSION:{torch.__version__}")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--epochs", "-e", type=str, help="Number of epochs", required=False, default=5
    )
    parser.add_argument(
        "--eval_steps",
        type=str,
        help="Number of evaluation steps to perform",
        required=False,
        default=1500,
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        help="Name of the experiment",
        required=False,
        default="ce2101",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Model to use for training",
        required=False,
        default="roberta-large",
    )
    parser.add_argument(
        "--without_eval",
        "-w_e",
        required=False,
        default=False,
        type=bool,
    )
    args = parser.parse_args()
    if not args.without_eval:
        samples = create_data()
        evaluator = create_data(split="val")
    else:
        samples, evaluator = create_data()
    # warmup_steps = 5000
    train_dataloader = DataLoader(samples, shuffle=False, batch_size=8)
    model = CrossEncoder(args.model, num_labels=1, max_length=400)
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=args.epochs,
        evaluation_steps=args.eval_steps,
        warmup_steps=1000,
        output_path=args.name,
        use_amp=True,
    )
    model.save(args.name + "-latest")
