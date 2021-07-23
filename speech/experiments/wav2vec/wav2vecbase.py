"""Train classifier with Base wav2vec features"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, load_metric
from packaging import version
from sklearn.model_selection import train_test_split
from speech.experiments.metrics import compute_metrics
from speech.soxan.src.collator import DataCollatorCTCWithPadding
from speech.soxan.src.modeling_outputs import SpeechClassifierOutput
from speech.soxan.src.models import (
    HubertForSpeechClassification,
    Wav2Vec2ForSpeechClassification,
)
from speech.soxan.src.trainer import CTCTrainer
from speech.utils.common import (
    build_dataframe_for_classification,
    clean_dataframe,
    get_wav_files,
    label_to_id,
    speech_file_to_array,
)
from torch import nn
from transformers import (
    AutoConfig,
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
    is_apex_available,
)
from transformers.file_utils import ModelOutput

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


def train():
    # TODO parameterize these hyperparameters
    model_name_or_path = "facebook/wav2vec2-base-960h"
    pooling_mode = "mean"

    # TODO specify datapaths on input
    save_path = "."
    speech_df = build_dataframe_for_classification(
        "./speech/data/data/td",
        "./speech/data/ssd",  # Test Locally see where filepath is from
    )
    print("Number of Datapoints: ", len(speech_df))
    speech_df = clean_dataframe(speech_df)
    train_df, test_df = train_test_split(
        speech_df, test_size=0.2, random_state=101, stratify=speech_df["labels"]
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
    test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)

    data_files = {
        "train": "./train.csv",
        "validation": "./test.csv",
    }

    dataset = load_dataset(
        "csv",
        data_files=data_files,
        delimiter="\t",
    )
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    input_column = "file_path"
    output_column = "labels"

    label_list = train_dataset.unique(output_column)
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)
    print(f"A classification problem with {num_labels} classes: {label_list}")

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
        finetuning_task="wav2vec2_clf",
    )
    setattr(config, "pooling_mode", pooling_mode)

    processor = Wav2Vec2Processor.from_pretrained(
        model_name_or_path,
    )
    target_sampling_rate = processor.feature_extractor.sampling_rate
    print(f"The target sampling rate: {target_sampling_rate}")

    def preprocess_function(dataset):
        """
        Preprocess Dataset, encode labels and convert speech files to vectors
        """
        speech_list = [speech_file_to_array(path) for path in dataset["file_path"]]
        target_list = [label_to_id(label, label_list) for label in dataset["labels"]]

        result = processor(speech_list, sampling_rate=target_sampling_rate)
        result["labels"] = list(target_list)

        return result

    train_dataset = train_dataset.map(
        preprocess_function, batch_size=50, batched=True, num_proc=4
    )
    eval_dataset = eval_dataset.map(
        preprocess_function, batch_size=50, batched=True, num_proc=4
    )

    data_collator = DataCollatorCTCWithPadding(processor, padding=True)

    model = Wav2Vec2ForSpeechClassification.from_pretrained(
        model_name_or_path,
        config=config,
    )
    model.freeze_feature_extractor()
    # TODO factor out import hyperparameters
    training_args = TrainingArguments(
        # output_dir="/content/wav2vec2-xlsr-greek-speech-emotion-recognition",
        output_dir="/content/drive/MyDrive/data",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=5.0,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=100,
        learning_rate=1e-4,
        save_total_limit=2,
    )

    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        tokenizer=processor.feature_extractor,
        eval_dataset=eval_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    train()
