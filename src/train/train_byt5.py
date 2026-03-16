"""Fine-tune ByT5-base for Akkadian→English translation.

Usage:
    python -m src.train.train_byt5 \
        --data-config configs/data/raw.yaml \
        --view-config configs/data/dual_view.yaml \
        --model-config configs/model/byt5_base.yaml \
        --train-config configs/train/byt5_finetune.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.eval.metrics import compute_generation_metrics
from src.utils.io import ensure_dir, load_yaml
from src.utils.logging import get_logger, write_json
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-config", required=True)
    p.add_argument("--view-config", required=True)
    p.add_argument("--model-config", required=True)
    p.add_argument("--train-config", required=True)
    return p.parse_args()


def load_splits(data_cfg: dict, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = Path(data_cfg["paths"]["processed_dir"]) / "train_prepared.csv"
    if not prepared.exists():
        raise FileNotFoundError(
            f"Expected {prepared}. Run: python -m src.data.prepare "
            f"--data-config configs/data/raw.yaml --view-config configs/data/dual_view.yaml"
        )
    df = pd.read_csv(prepared)
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)
    return train_df, val_df


def build_dataset(df: pd.DataFrame, tokenizer, src_max: int, tgt_max: int) -> Dataset:
    """Tokenize source/target pairs for ByT5."""
    sources = df["packed_source"].tolist()
    targets = df["target_text"].tolist()

    model_inputs = tokenizer(
        sources, max_length=src_max, truncation=True, padding=False,
    )
    labels = tokenizer(
        text_target=targets, max_length=tgt_max, truncation=True, padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return Dataset.from_dict(model_inputs)


def main() -> None:
    args = parse_args()
    logger = get_logger()

    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)

    seed_everything(train_cfg["seed"])

    model_name = model_cfg["model_name_or_path"]
    src_max = model_cfg["source_max_length"]
    tgt_max = model_cfg["target_max_length"]
    gen_cfg = model_cfg.get("generation", {})

    logger.info("Loading tokenizer & model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Log model size
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: %.1fM total, %.1fM trainable", n_params / 1e6, n_trainable / 1e6)

    # Load data
    train_df, val_df = load_splits(data_cfg, train_cfg["fold"])
    logger.info("Train: %d, Val: %d", len(train_df), len(val_df))

    train_ds = build_dataset(train_df, tokenizer, src_max, tgt_max)
    val_ds = build_dataset(val_df, tokenizer, src_max, tgt_max)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # Metric computation
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        # Replace -100 in labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        metrics = compute_generation_metrics(decoded_preds, decoded_labels)
        # Competition score = sqrt(BLEU * chrF++)
        bleu = max(metrics["sacrebleu"], 0.0)
        chrf = max(metrics["chrf"], 0.0)
        metrics["competition_score"] = float(np.sqrt(bleu * chrf))
        return metrics

    output_dir = ensure_dir(train_cfg["output_dir"])

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        run_name=train_cfg["run_name"],
        # Batching
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        # Schedule
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        max_grad_norm=float(train_cfg["max_grad_norm"]),
        # Precision
        bf16=train_cfg.get("bf16", False),
        # Eval & save
        eval_strategy=train_cfg["eval_strategy"],
        eval_steps=train_cfg["eval_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=train_cfg["greater_is_better"],
        # Generation
        predict_with_generate=True,
        generation_max_length=tgt_max,
        generation_num_beams=gen_cfg.get("num_beams", 5),
        # Logging
        logging_steps=train_cfg["logging_steps"],
        report_to=train_cfg.get("report_to", "none"),
        # Seed
        seed=train_cfg["seed"],
        data_seed=train_cfg["seed"],
        # Performance
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save best model + metrics
    trainer.save_model(str(output_dir / "best"))
    tokenizer.save_pretrained(str(output_dir / "best"))

    logger.info("Running final evaluation...")
    metrics = trainer.evaluate()
    write_json(metrics, output_dir / "metrics.json")
    logger.info("Final metrics: %s", metrics)

    # Generate val predictions for error analysis
    logger.info("Generating val predictions...")
    pred_output = trainer.predict(val_ds)
    preds = tokenizer.batch_decode(pred_output.predictions, skip_special_tokens=True)
    labels_clean = np.where(
        pred_output.label_ids != -100, pred_output.label_ids, tokenizer.pad_token_id
    )
    refs = tokenizer.batch_decode(labels_clean, skip_special_tokens=True)

    pred_df = pd.DataFrame({
        "text_id": val_df["text_id"].tolist()[:len(preds)],
        "source": val_df["packed_source"].tolist()[:len(preds)],
        "reference": refs,
        "prediction": preds,
    })
    pred_df.to_csv(output_dir / "val_predictions.csv", index=False)
    logger.info("Saved val predictions to %s", output_dir / "val_predictions.csv")


if __name__ == "__main__":
    main()
