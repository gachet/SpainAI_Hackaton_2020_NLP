import math
import os
from argparse import ArgumentParser

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from get_submission import chunks, clean_text, filter_predlist, save_submission_df
from transformers import (
    Adafactor,
    DataCollatorForSeq2Seq,
    PegasusConfig,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def to_device(d, device):
    for k, v in d.items():
        d[k] = v.to(device)
    return d


# model.zero_grad()                                   # Reset gradients tensors
# for i, (inputs, labels) in enumerate(training_set):
#     predictions = model(inputs)                     # Forward pass
#     loss = loss_function(predictions, labels)       # Compute loss function
#     loss = loss / accumulation_steps                # Normalize our loss (if averaged)
#     loss.backward()                                 # Backward pass
#     if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
#         optimizer.step()                            # Now we can do an optimizer step
#         model.zero_grad()                           # Reset gradients tensors
#         if (i+1) % evaluation_steps == 0:           # Evaluate the model when we...
#             evaluate_model()


def get_predictions(
    model, tokenizer, texts, batchsize=2, numseqs=30, device="cuda", maxlen=364
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
                **batch_encoded,
                num_return_sequences=numseqs,
                num_beams=numseqs,
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


def evaluate(model, val_dataloader, device="cuda"):
    model.eval()
    model.zero_grad()
    val_loss = torch.tensor(0.0).to(device)
    for _, batch in tqdm(enumerate(val_dataloader), desc="Iterating over val data"):
        batch = to_device(batch, device)
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        val_loss += loss.detach()
    val_loss_scalar = val_loss.item()
    val_loss_final = val_loss_scalar / len(val_dataloader)
    return val_loss_final


def save(model, optimizer, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))


def train(
    model,
    train_dataloader,
    val_dataloader,
    savedir,
    device="cuda",
    epochs=10,
    grad_acc_steps=32,
    eval_steps=125,
    checkpoint=None,
):
    if grad_acc_steps is not None:
        eval_steps = int(eval_steps * grad_acc_steps)
    os.makedirs(savedir, exist_ok=True)
    model.to(torch.device(device))
    model.train()
    optimizer = Adafactor(
        model.parameters(),
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=True,
        scale_parameter=True,
        warmup_init=True,
    )
    if checkpoint is not None:
        optimizer.load_state_dict(torch.load(os.path.join(checkpoint, "optimizer.pt")))
    update_steps_per_epoch = len(train_dataloader) // grad_acc_steps
    update_steps_per_epoch = math.ceil(update_steps_per_epoch)
    len_last_batch = len(train_dataloader) - update_steps_per_epoch * grad_acc_steps
    model.zero_grad()
    # optimizer.zero_grad()
    tr_loss = torch.tensor(0.0).to(device)
    ckpt_num = 0
    global_step = 0
    globalstep_last_logged = 0
    for epoch in range(epochs):
        for step, batch in tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch}:"):
            batch = to_device(batch, device)
            outputs = model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            if (step + 1) == len(train_dataloader):
                loss = loss / len_last_batch
            else:
                loss = loss / grad_acc_steps
            loss.backward()
            tr_loss += loss.detach()
            if (step + 1) % grad_acc_steps == 0 or (step + 1) == len(train_dataloader):
                optimizer.step()
                model.zero_grad()

            if (global_step + 1) % eval_steps == 0:
                tr_loss_scalar = tr_loss.item()
                tr_loss -= tr_loss
                loss = round(
                    tr_loss_scalar / (global_step - globalstep_last_logged),
                    4,
                )
                globalstep_last_logged = global_step
                # torch.cuda.empty_cache()
                val_loss = evaluate(model, val_dataloader, device)
                print(f"Train loss is {loss}; Val loss is {val_loss}")
                print(f"Saving model to {savedir}/checkpoint-{ckpt_num}")
                save(model, optimizer, f"{savedir}/checkpoint-{ckpt_num}")
                ckpt_num += 1
                model.train()
            global_step += 1
    return model, optimizer


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--train_file", required=False, default="train_2203.csv", type=str
    )
    parser.add_argument("--val_file", required=False, default="val_2203.csv", type=str)
    parser.add_argument("--prefix", default="", type=str, required=False)
    parser.add_argument("--max_len", type=int, default=364, required=False)
    parser.add_argument("--checkpoint", type=str, default=None, required=False)
    parser.add_argument("--batch_size", type=int, default=6, required=False)
    parser.add_argument("--eval_batch_size", type=int, default=12, required=False)
    parser.add_argument("--model_type", default="t5", type=str, required=False)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--get_submission", type=bool, default=False, required=False)
    parser.add_argument(
        "--submission_name", type=str, default="submission_alternative_t5_2403.csv"
    )
    parser.add_argument("--eval_steps", type=int, default=125)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--grad_acc_steps", type=int, default=32, required=False)
    parser.add_argument("--train", type=bool, default=True)
    # parser.add_argument("--base_model", type=str, required=True)
    args = parser.parse_args()
    if args.model_type == "t5":
        config = T5Config.from_pretrained("t5-large")
        tokenizer = T5Tokenizer.from_pretrained("t5-large")
        if args.checkpoint is None:
            model = T5ForConditionalGeneration.from_pretrained(
                "t5-large", config=config
            )
        else:
            model = T5ForConditionalGeneration.from_pretrained(
                args.checkpoint, config=config
            )
    elif args.model_type == "pegasus":
        config = PegasusConfig.from_pretrained("google/pegasus-large")
        tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")
        if args.checkpoint is None:
            model = PegasusForConditionalGeneration.from_pretrained(
                "google/pegasus-large", config=config
            )
        else:
            model = PegasusForConditionalGeneration.from_pretrained(
                args.checkpoint, config=config
            )

    task = "summarization"
    task_specific_params = model.config.task_specific_params
    task_specific_params.update(
        {
            "summarization": {
                "min_length": 0,
                "max_length": 60,
                "length_penalty": 1.0,
                "num_beams": 1,
            }
        }
    )
    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        model.config.update(pars)

    dataset = load_dataset(
        "csv", data_files={"train": args.train_file, "val": args.val_file}
    )
    tr_dataset = dataset["train"]
    val_dataset = dataset["val"]
    prefix = args.prefix

    def preprocess_function(examples):
        inputs = examples["description"]
        targets = examples["name"]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(
            inputs,
            max_length=args.max_len,
            padding="longest",
            truncation=True,
        )

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=60, padding="longest", truncation=True
            )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tr_dataset = tr_dataset.map(
        preprocess_function,
        batched=True,
        # batch_size=200,
        num_proc=25,
        remove_columns=["description", "name"],
    )
    val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        # batch_size=200,
        num_proc=25,
        remove_columns=["description", "name"],
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        # max_length=364
    )

    tr_dataloader = DataLoader(
        tr_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        drop_last=False,
        num_workers=25,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        drop_last=False,
        num_workers=25,
    )
    if args.train:
        model, optimizer = train(
            model,
            tr_dataloader,
            val_dataloader,
            savedir=args.model_name,
            device="cuda",
            checkpoint=args.checkpoint,
            epochs=args.epochs,
            eval_steps=args.eval_steps,
            grad_acc_steps=args.grad_acc_steps,
        )
    save(model, optimizer, f"{args.model_name}/final")
    del optimizer
    torch.cuda.empty_cache()
    if args.get_submission:
        print("Getting submission!")
        # test = pd.read_csv("test_descriptions.csv")
        # test["description"] = test["description"].apply(clean_text)
        # test_texts = test["description"].tolist()
        dftest = pd.read_csv("test_descriptions.csv")
        dftest["description"] = dftest["description"].apply(clean_text)
        texts = dftest["description"].tolist()
        # dftest["name"] = dftest["description"]
        # dftest.to_csv("test.csv", header=True, index=False)
        # test_dataset = load_dataset("csv", data_files={"test": "test.csv"})
        # test_dataset = test_dataset["test"]
        #
        # def preprocess_function_test(examples):
        #    inputs = examples["description"]
        #    targets = examples["name"]
        #    inputs = [prefix + inp for inp in inputs]
        #    model_inputs = tokenizer(
        #        inputs,
        #        max_length=args.max_len,
        #        padding="longest",
        #        truncation=True,
        #    )
        #
        #    # Setup the tokenizer for targets
        #    with tokenizer.as_target_tokenizer():
        #        labels = tokenizer(
        #            targets, max_length=60, padding="longest", truncation=True
        #        )
        #
        #    ## If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        #    ## padding in the loss.
        #    labels["input_ids"] = [
        #        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        #        for label in labels["input_ids"]
        #    ]
        #
        #    model_inputs["labels"] = labels["input_ids"]
        #    return model_inputs
        #
        # test_dataset = test_dataset.map(
        #    preprocess_function_test,
        #    batched=True,
        #    # batch_size=200,
        #    num_proc=25,
        #    remove_columns=["description", "name"],
        # )
        # test_dataloader = DataLoader(
        #    test_dataset,
        #    batch_size=2,
        #    shuffle=False,
        #    collate_fn=data_collator,
        #    drop_last=False,
        #    num_workers=25,
        # )
        predictions = get_predictions(
            model,
            tokenizer,
            texts,
            numseqs=30,
            device="cuda",
            maxlen=args.max_len,
        )
        save_submission_df(predictions, args.submission_name)
