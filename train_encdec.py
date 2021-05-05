import os
from argparse import ArgumentParser

import datasets
import pandas as pd
import torch.utils.cpp_extension

import transformers
from transformers import (
    BertTokenizerFast,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    EncoderDecoderModel,
    RobertaTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

torch.utils.cpp_extension.CUDA_HOME = "/usr/local/cuda-10.1"

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=6, early_stopping_threshold=0.001
)

model_classes = {
    "roberta": {"tokenizer": RobertaTokenizerFast},
    "bert": {"tokenizer": BertTokenizerFast},
}


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def filter_predlist(predlist):
    newlist = []
    for pred in predlist:
        if pred not in newlist:
            newlist.append(pred)
    return newlist[:10]


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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-base")
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--encoder_max_len", type=int, default=360)
    parser.add_argument("--decoder_max_len", type=int, default=60)
    parser.add_argument("--save_steps", type=int, default=250)
    parser.add_argument("--max_target_length", type=int, default=60)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument(
        "--submission_name", type=str, default="submission_2803_deberta.csv"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--grad_acc_steps", default=4, type=int)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name)
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token
    dataset = datasets.load_dataset(
        "csv", data_files={"train": "train_2503.csv", "val": "val_2503.csv"}
    )

    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels
        inputs = tokenizer(
            batch["description"],
            padding="max_length",
            truncation=True,
            max_length=args.encoder_max_len,
        )
        outputs = tokenizer(
            batch["name"],
            padding="max_length",
            truncation=True,
            max_length=args.decoder_max_len,
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()

        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
        # We have to make sure that the PAD token is ignored
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in batch["labels"]
        ]

        return batch

    dataset["train"] = dataset["train"].map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=int(args.bs) * 200,
        num_proc=20,
        remove_columns=["description", "name"],
    )
    dataset["val"] = dataset["val"].map(
        process_data_to_model_inputs,
        batched=True,
        num_proc=20,
        batch_size=int(args.bs) * 200,
        remove_columns=["description", "name"],
    )
    train_data = dataset["train"]
    val_data = dataset["val"]
    # train_data.set_format(
    #    type="torch",
    #    columns=[
    #        "input_ids",
    #        "attention_mask",
    #        "decoder_input_ids",
    #        "decoder_attention_mask",
    #        "labels",
    #    ],
    # )
    # val_data.set_format(
    #    type="torch",
    #    columns=[
    #        "input_ids",
    #        "attention_mask",
    #        "decoder_input_ids",
    #        "decoder_attention_mask",
    #        "labels",
    #    ],
    # )
    bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained(
        args.model_name, args.model_name, tie_encoder_decoder=True
    )
    # set special tokens
    bert2bert.config.decoder_start_token_id = tokenizer.bos_token_id
    bert2bert.config.eos_token_id = tokenizer.eos_token_id
    bert2bert.config.pad_token_id = tokenizer.pad_token_id

    # sensible parameters for beam search
    bert2bert.config.vocab_size = bert2bert.config.decoder.vocab_size
    bert2bert.config.max_length = 60
    bert2bert.config.min_length = 0
    bert2bert.config.no_repeat_ngram_size = 3
    bert2bert.config.early_stopping = True
    bert2bert.config.length_penalty = 2.0
    bert2bert.config.num_beams = 4
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"encoder_decoder_{args.model_name}",
        evaluation_strategy="steps",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs * 2,
        predict_with_generate=False,
        logging_steps=args.save_steps,  # set to 1000 for full training
        save_steps=args.save_steps,  # set to 500 for full training
        eval_steps=args.save_steps,  # set to 8000 for full training
        warmup_steps=500,  # set to 2000 for full training
        overwrite_output_dir=True,
        save_total_limit=10,
        fp16=True,
        gradient_accumulation_steps=args.grad_acc_steps,
        load_best_model_at_end=True,
        # deepspeed="./folder_aws/ds_config.json",
        # max_steps=1,  # TODO: TAKE THIS OUT
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=bert2bert,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    trainer = Seq2SeqTrainer(
        model=bert2bert,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        callbacks=[early_stopping],
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train_results = trainer.train()
    trainer.save_model()
    metrics = train_results.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    metrics_val = trainer.evaluate(
        max_length=args.max_target_length,
        num_beams=args.num_beams,
        metric_key_prefix="eval",
    )
    trainer.log_metrics("eval", metrics_val)
    trainer.save_metrics("eval", metrics_val)
    trainer.args.per_device_eval_batch_size = 1
    test_dataset = datasets.load_dataset("csv", data_files={"test": "test.csv"})
    test_dataset = test_dataset["test"]
    test_dataset = test_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        num_proc=20,
        batch_size=int(args.bs) * 200,
        remove_columns=["description", "name"],
    )
    trainer.args.predict_with_generate = True
    trainer.args.prediction_loss_only = False
    trainer._num_beams = 30
    output = trainer.predict(
        test_dataset, num_beams=30, max_length=args.encoder_max_len
    )
    predictions_detokenized = [
        trainer.tokenizer.decode(
            pred, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for predictions in output.predictions
        for pred in predictions
    ]
    predictions = list(chunks(predictions_detokenized, 30))
    predictions = list(map(filter_predlist, predictions))
    save_submission_df(
        predictions,
        f"/home/alejandro.vaca/SpainAI_Hackaton_2020/{args.submission_name}",
    )
