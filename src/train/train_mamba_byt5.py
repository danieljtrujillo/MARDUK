"""Phase 2: Fine-tune Mamba-enhanced ByT5 on expanded Akkadian→English data.

Two-stage training:
  1. Adapter warmup (freeze ByT5, train only Mamba adapter layers)
  2. Full fine-tuning (unfreeze all, lower LR)

Usage:
    python -m src.train.train_mamba_byt5 \
        --data-config configs/data/raw.yaml \
        --view-config configs/data/dual_view.yaml \
        --model-config configs/model/mamba_byt5.yaml \
        --train-config configs/train/mamba_byt5_finetune.yaml
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
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.eval.metrics import compute_generation_metrics
from src.models.mamba_adapter_byt5 import create_mamba_byt5, save_mamba_byt5
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
    sources = df["packed_source"].tolist()
    targets = df["target_text"].tolist()
    model_inputs = tokenizer(sources, max_length=src_max, truncation=True, padding=False)
    labels = tokenizer(text_target=targets, max_length=tgt_max, truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return Dataset.from_dict(model_inputs)


def main() -> None:
    args = parse_args()
    logger = get_logger()

    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)

    seed_everything(train_cfg["seed"])

    src_max = model_cfg["source_max_length"]
    tgt_max = model_cfg["target_max_length"]
    gen_cfg = model_cfg.get("generation", {})

    # Load tokenizer from base model
    base_model = model_cfg["base_model"]
    checkpoint = model_cfg.get("checkpoint", base_model)
    logger.info("Loading tokenizer from: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Create Mamba-enhanced ByT5
    adapter_cfg = model_cfg.get("adapter", {})
    logger.info("Creating Mamba-enhanced ByT5 from checkpoint: %s", checkpoint)
    model, param_info = create_mamba_byt5(
        checkpoint_path=checkpoint,
        n_mamba_layers=adapter_cfg.get("n_layers", 2),
        d_state=adapter_cfg.get("d_state", 16),
        d_conv=adapter_cfg.get("d_conv", 4),
        expand=adapter_cfg.get("expand", 2),
        adapter_dropout=adapter_cfg.get("dropout", 0.1),
        freeze_byt5=train_cfg.get("freeze_byt5_stage1", False),
    )
    logger.info(
        "Parameters: %.1fM total, %.1fM trainable, %.1fM adapter",
        param_info["total"] / 1e6,
        param_info["trainable"] / 1e6,
        param_info["adapter"] / 1e6,
    )

    # Load data
    train_df, val_df = load_splits(data_cfg, train_cfg["fold"])
    logger.info("Train: %d, Val: %d", len(train_df), len(val_df))

    train_ds = build_dataset(train_df, tokenizer, src_max, tgt_max)
    val_ds = build_dataset(val_df, tokenizer, src_max, tgt_max)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        metrics = compute_generation_metrics(decoded_preds, decoded_labels)
        bleu = max(metrics["sacrebleu"], 0.0)
        chrf = max(metrics["chrf"], 0.0)
        metrics["competition_score"] = float(np.sqrt(bleu * chrf))
        return metrics

    output_dir = ensure_dir(train_cfg["output_dir"])

    # ── Stage 1: Adapter warmup (optional, if freeze_byt5_stage1 is set) ──
    if train_cfg.get("freeze_byt5_stage1", False):
        warmup_epochs = train_cfg.get("stage1_epochs", 3)
        logger.info("Stage 1: Adapter warmup (%d epochs, ByT5 frozen)", warmup_epochs)

        stage1_args = Seq2SeqTrainingArguments(
            output_dir=str(output_dir / "stage1"),
            run_name=f"{train_cfg['run_name']}_stage1",
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            num_train_epochs=warmup_epochs,
            learning_rate=float(train_cfg.get("stage1_lr", 1e-3)),
            weight_decay=float(train_cfg["weight_decay"]),
            warmup_ratio=float(train_cfg["warmup_ratio"]),
            lr_scheduler_type="cosine",
            max_grad_norm=float(train_cfg["max_grad_norm"]),
            bf16=train_cfg.get("bf16", False),
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="competition_score",
            greater_is_better=True,
            predict_with_generate=True,
            generation_max_length=tgt_max,
            generation_num_beams=gen_cfg.get("num_beams", 5),
            logging_steps=10,
            report_to="none",
            seed=train_cfg["seed"],
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
        )

        trainer_s1 = Seq2SeqTrainer(
            model=model,
            args=stage1_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )
        trainer_s1.train()

        s1_metrics = trainer_s1.evaluate()
        logger.info("Stage 1 metrics: %s", s1_metrics)

        # Unfreeze ByT5 for stage 2
        for param in model.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Unfroze ByT5 — %d trainable params for stage 2", trainable)

    # ── Stage 2: Full fine-tuning ──
    stage2_epochs = train_cfg.get("stage2_epochs", train_cfg["num_train_epochs"])
    stage2_lr = float(train_cfg.get("stage2_lr", train_cfg["learning_rate"]))
    logger.info("Stage 2: Full fine-tuning (%d epochs, lr=%.2e)", stage2_epochs, stage2_lr)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        run_name=train_cfg["run_name"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=stage2_epochs,
        learning_rate=stage2_lr,
        weight_decay=float(train_cfg["weight_decay"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        max_grad_norm=float(train_cfg["max_grad_norm"]),
        bf16=train_cfg.get("bf16", False),
        eval_strategy=train_cfg["eval_strategy"],
        eval_steps=train_cfg["eval_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=train_cfg["greater_is_better"],
        predict_with_generate=True,
        generation_max_length=tgt_max,
        generation_num_beams=gen_cfg.get("num_beams", 5),
        logging_steps=train_cfg["logging_steps"],
        report_to=train_cfg.get("report_to", "none"),
        seed=train_cfg["seed"],
        data_seed=train_cfg["seed"],
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

    # Save best model (custom save for Mamba adapter)
    save_mamba_byt5(model, tokenizer, str(output_dir / "best"))
    logger.info("Saved Mamba-ByT5 model to %s", output_dir / "best")

    # Final eval
    logger.info("Running final evaluation...")
    metrics = trainer.evaluate()
    write_json(metrics, output_dir / "metrics.json")
    logger.info("Final metrics: %s", metrics)

    # Val predictions for error analysis
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
