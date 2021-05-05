import argparse
import logging
import os

import pandas as pd
from datasets import load_dataset

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    HfArgumentParser,
    MBartTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5Tokenizer,
    Trainer,
    default_data_collator,
    set_seed,
)

# from utils_perturbation import get_features_df

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logger = logging.getLogger(__name__)

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=6, early_stopping_threshold=0.005
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=False,
        default="t5-large",
        type=str,
        help="Model to use for training",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Output directory to save model",
    )
    parser.add_argument(
        "--batch_size", type=int, required=False, default=4, help="Train batch size"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        required=False,
        default=16,
        help="Eval batch size",
    )
    parser.add_argument(
        "--grad_acc_steps",
        type=int,
        required=False,
        default=128,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--num_epochs",
        required=False,
        default=10,
        type=int,
        help="Number of epochs to train.",
    )
    parser.add_argument("--lr", type=float, required=False, help="Learning rate to use")
    parser.add_argument(
        "--save_steps",
        required=True,
        type=int,
        help="Number of steps to make before evaluating, logging and saving",
    )
    args = parser.parse_args()
    # train_dataset = load_dataset("csv", data_files="data_perturbed/features_train.csv")
    # val_dataset = load_dataset("csv", data_files="data_perturbed/features_val.csv")
    dataset = load_dataset(
        "csv",
        data_files={
            "train": "perturbed_bart_data/tr.csv",
            "val": "perturbed_bart_data/val.csv",
        },
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        prediction_loss_only=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        eval_accumulation_steps=10,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        save_steps=args.save_steps,
        logging_steps=args.save_steps,
        eval_steps=args.save_steps,
        fp16=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    set_seed(training_args.seed)
    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, config=config)

    def tokenize(examples):
        inputs = examples["sentence"]
        targets = examples["labels"]
        model_inputs = tokenizer(
            inputs, max_length=346, padding="max_length", truncation=True
        )

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=30, padding="max_length", truncation=True
            )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = dataset["train"]
    train_dataset = train_dataset.map(
        tokenize, batched=True, remove_columns=["sentence"], num_proc=5
    )
    val_dataset = dataset["val"]
    val_dataset = val_dataset.map(
        tokenize, batched=True, remove_columns=["sentence"], num_proc=5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=[early_stopping],
    )
    train_result = trainer.train()
    trainer.save_model()
    output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
    if trainer.is_world_process_zero():
        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        results = trainer.evaluate()

        output_eval_file = os.path.join(
            training_args.output_dir, "eval_results_seq2seq.txt"
        )
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
