import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import ray
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining
from torch.utils.data import Dataset
from tqdm import tqdm

import transformers
from transformers import (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer,
                          EarlyStoppingCallback, EvalPrediction,
                          HfArgumentParser, MBartTokenizer, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, TrainerCallback, set_seed)
from transformers.trainer_utils import EvaluationStrategy, is_main_process
from transformers.training_args import ParallelMode
from utils import (Seq2SeqDataCollator, Seq2SeqDataset, assert_all_frozen,
                   build_compute_metrics_fn, check_output_dir, freeze_embeds,
                   freeze_params, lmap, save_json, use_task_specific_params,
                   write_txt_file)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    freeze_encoder: bool = field(
        default=False, metadata={"help": "Whether tp freeze the encoder."}
    )
    freeze_embeds: bool = field(
        default=False, metadata={"help": "Whether  to freeze the embeddings."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={
            "help": "The input data dir. Should contain the .tsv files (or other data files) for the task."
        }
    )
    task: Optional[str] = field(
        default="summarization",
        metadata={
            "help": "Task name, summarization (or summarization_{dataset} for pegasus) or translation"
        },
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=142,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. "
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    test_max_target_length: Optional[int] = field(
        default=142,
        metadata={
            "help": "The maximum total sequence length for test target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    n_train: Optional[int] = field(
        default=-1, metadata={"help": "# training examples. -1 means use all."}
    )
    n_val: Optional[int] = field(
        default=-1, metadata={"help": "# validation examples. -1 means use all."}
    )
    n_test: Optional[int] = field(
        default=-1, metadata={"help": "# test examples. -1 means use all."}
    )
    src_lang: Optional[str] = field(
        default=None, metadata={"help": "Source language id for translation."}
    )
    tgt_lang: Optional[str] = field(
        default=None, metadata={"help": "Target language id for translation."}
    )
    eval_beams: Optional[int] = field(
        default=None, metadata={"help": "# num_beams to use for evaluation."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."
        },
    )


def handle_metrics(split, metrics, output_dir):
    """
    Log and save metrics

    Args:
    - split: one of train, val, test
    - metrics: metrics dict
    - output_dir: where to save the metrics
    """

    logger.info(f"***** {split} metrics *****")
    for key in sorted(metrics.keys()):
        logger.info(f"  {key} = {metrics[key]}")
    save_json(metrics, os.path.join(output_dir, f"{split}_results.json"))


def eval_steps_func(spec):
    return 250


def train_model(args, checkpoint_dir=None):
    model_args = ModelArguments(
        model_name_or_path=args.model, config_name=args.model, tokenizer_name=args.model
    )
    data_args = DataTrainingArguments(
        data_dir=args.data_dir,
        max_source_length=args.max_source_len,
        max_target_length=args.max_target_len,
        val_max_target_length=args.max_target_len,
        eval_beams=args.eval_beams,
    )
    training_args = Seq2SeqTrainingArguments(
        sortish_sampler=args.sortish_sampler,
        predict_with_generate=False,
        output_dir=".",  # args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        do_predict=False,
        evaluation_strategy="epoch",
        prediction_loss_only=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        # eval_accumulation_steps=4,
        gradient_accumulation_steps=32,
        num_train_epochs=2,
        max_steps=-1,
        fp16=True,
        # save_total_limit=10,
        # eval_steps=250, #int(34593 / (args.batch_size*16)),
        # run_name=args.run_name,
        load_best_model_at_end=True,
        greater_is_better=False,
        metric_for_best_model="eval_loss",
        logging_dir="./logs"
        # fp16_backend="apex",
    )
    # training_args._n_gpu = 1
    # check_output_dir(training_args)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.parallel_mode == ParallelMode.DISTRIBUTED),
        training_args.fp16,
    )
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)
    set_seed(training_args.seed)
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    # model = AutoModelForSeq2SeqLM.from_pretrained(
    #    model_args.model_name_or_path,
    #    from_tf=".ckpt" in model_args.model_name_or_path,
    #    config=config,
    #    cache_dir=model_args.cache_dir,
    # )
    # use_task_specific_params(model, data_args.task)
    task = "summarization"
    task_specific_params = config.task_specific_params
    task_specific_params.update(
        {
            "summarization": {
                "min_length": 0,
                "max_length": 60,
                "length_penalty": 1.0,
                "num_beams": 4,
            }
        }
    )

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        logger.info(
            f"setting model.config to task specific params for {task}:\n {pars}"
        )
        logger.info("note: command line args may override some of these")
        config.update(pars)

    if data_args.eval_beams is None:
        data_args.eval_beams = config.num_beams
    dataset_class = Seq2SeqDataset
    train_dataset = (
        dataset_class(
            tokenizer,
            type_path="train",
            data_dir=data_args.data_dir,
            n_obs=data_args.n_train,
            max_target_length=data_args.max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=config.prefix or "",
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        dataset_class(
            tokenizer,
            type_path="val",
            data_dir=data_args.data_dir,
            n_obs=data_args.n_val,
            max_target_length=data_args.val_max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=config.prefix or "",
        )
        if training_args.do_eval
        or training_args.evaluation_strategy != EvaluationStrategy.NO
        else None
    )
    compute_metrics_fn = (
        build_compute_metrics_fn(data_args.task, tokenizer)
        if training_args.predict_with_generate
        else None
    )
    # val_df = get_validation_df(data_args.data_dir)
    # dcg_callback = DCGCallback(tokenizer, val_df)

    def get_model():
        return AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=".ckpt" in model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
        )

    trainer = Seq2SeqTrainer(
        model_init=get_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Seq2SeqDataCollator(
            tokenizer, data_args, decoder_start_token_id=2
        ),
        compute_metrics=compute_metrics_fn,
        tokenizer=tokenizer,
        # callbacks=[dcg_callback],  # my_callback,
    )
    ray.init(args.ray_address)  # , log_to_driver=True
    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="eval_loss",
        mode="min",
        perturbation_interval=1,
        hyperparam_mutations={
            # TODO: AQUÍ PODRÍA BUSCAR TAMBIÉN LOS PARÁMETROS DE ADAMW.
            "weight_decay": tune.uniform(0.0, 0.3),
            "learning_rate": tune.uniform(1e-5, 7e-5),
            "per_device_train_batch_size": [2, 4],
            # "gradient_accumulation_steps": [2, 4, 8, 16, 64],
            # "warmup_steps": [0, 200, 400, 800],
        },
    )

    tune_config = {
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        # "eval_steps": tune.sample_from(eval_steps_func),
        # "save_steps": tune.sample_from(lambda spec: spec.config["eval_steps"]),
        "num_train_epochs": 10,  # tune.choice([2, 4, 6, 8, 10]),
        "max_steps": 1 if args.smoke_test else -1,  # Used for smoke test.
    }

    reporter = CLIReporter(
        parameter_columns={
            "weight_decay": "w_decay",
            "learning_rate": "lr",
            "per_device_train_batch_size": "train_bs",
            # "warmup_steps": "warmup",
            # "gradient_accumulation_steps": "train_bs/gpu",
            "num_epochs": "num_epochs",
        },
        metric_columns=["eval_loss", "epoch", "training_iteration"],
    )
    trainer.hyperparameter_search(
        hp_space=lambda _: tune_config,
        backend="ray",
        n_trials=args.population_number,
        resources_per_trial={"cpu": 5, "gpu": 1},
        scheduler=scheduler,
        checkpoint_freq=1,
        keep_checkpoints_num=1,
        checkpoint_score_attr="training_iteration",
        stop={"training_iteration": 1}
        if args.smoke_test
        else None,  # {"training_iteration": 1} if smoke_test else
        progress_reporter=reporter,
        local_dir="~/ray_results/",
        name="PBT_BART",
        log_to_file=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Address to use for Ray. "
        'Use "auto" for cluster. '
        "Defaults to None for local.",
    )
    parser.add_argument(
        "--model", type=str, default="facebook/bart-large", help="Model to use"
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument(
        "--max_source_len", type=int, default=1024, help="Max source sequence length"
    )
    parser.add_argument(
        "--max_target_len", type=int, default=1024, help="Max target sequence length"
    )
    parser.add_argument(
        "--eval_beams",
        type=int,
        default=None,
        help="Number of evaluation beams to use.",
    )
    parser.add_argument(
        "--sortish_sampler",
        type=bool,
        required=False,
        default=True,
        help="Whether or not to use sortish sampler.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save the data.",
    )
    parser.add_argument(
        "--batch_size", type=int, required=False, default=8, help="Train batch size"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        required=False,
        default=16,
        help="Eval batch size",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=False,
        default="pbt_summarizer_3001",
        help="Name of the run for tensorboard.",
    )
    parser.add_argument(
        "--population_number",
        required=False,
        default=4,
        type=int,
        help="Number of members of the population for PBT.",
    )
    parser.add_argument(
        "--smoke_test", default=False, required=False, type=bool, help="For debugging."
    )
    args = parser.parse_args()
    train_model(args)
