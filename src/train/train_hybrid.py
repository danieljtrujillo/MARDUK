from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data.bio_labels import generate_bio_labels, load_lexicon
from src.data.collators import ByteSourceEncoder, HybridCollator
from src.eval.error_buckets import build_error_buckets
from src.eval.metrics import all_metrics
from src.models.hybrid_seq2seq import HybridSeq2Seq
from src.utils.io import ensure_dir, load_yaml
from src.utils.logging import get_logger, write_json
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--view-config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", required=True)
    return parser.parse_args()


class HybridDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, source_max_length: int = 1024) -> None:
        self.frame = frame.reset_index(drop=True)
        self.source_max_length = source_max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict:
        row = self.frame.iloc[idx]
        source = str(row["packed_source"])
        # Byte-level BIO labels using lexicon-aware generator
        n_bytes = len(source.encode("utf-8", errors="replace")) + 2  # +2 for BOS/EOS
        n_bytes = min(n_bytes, self.source_max_length)
        aux = generate_bio_labels(source, n_bytes)
        return {
            "text_id": str(row["text_id"]),
            "source": source,
            "target": str(row["target_text"]),
            **aux,
        }


def load_prepared_frame(data_cfg: dict, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared_path = Path(data_cfg["paths"]["processed_dir"]) / "train_prepared.csv"
    if not prepared_path.exists():
        raise FileNotFoundError(f"Expected prepared dataset at {prepared_path}. Run scripts/prepare_data.sh first.")
    df = pd.read_csv(prepared_path)
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)
    return train_df, val_df


@torch.no_grad()
def evaluate_model(model, loader, tokenizer, device, num_beams: int = 5):
    model.eval()
    preds: list[str] = []
    refs: list[str] = []
    sources: list[str] = []
    ids: list[str] = []

    for batch in loader:
        source_ids = batch.source_ids.to(device)
        source_mask = batch.source_mask.to(device)

        generated = model.generate(
            source_ids=source_ids,
            source_mask=source_mask,
            max_new_tokens=128,
            num_beams=num_beams,
            length_penalty=0.9,
            no_repeat_ngram_size=3,
        )
        pred_text = tokenizer.batch_decode(generated, skip_special_tokens=True)

        ref_text = tokenizer.batch_decode(batch.target_ids, skip_special_tokens=True)
        preds.extend(pred_text)
        refs.extend(ref_text)
        sources.extend([""] * len(pred_text))
        ids.extend([""] * len(pred_text))

    metrics = all_metrics(preds, refs)
    return metrics, preds, refs


def main() -> None:
    args = parse_args()
    logger = get_logger()
    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)
    seed_everything(train_cfg["seed"])

    # Load lexicon for BIO labels
    lexicon_path = data_cfg["paths"].get("lexicon_csv")
    if lexicon_path:
        load_lexicon(lexicon_path)
        logger.info("Loaded OA Lexicon from %s", lexicon_path)

    train_df, val_df = load_prepared_frame(data_cfg, train_cfg["fold"])

    output_dir = ensure_dir(train_cfg["output_dir"])
    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")

    source_encoder = ByteSourceEncoder(max_length=model_cfg["input"]["source_max_length"])
    target_tokenizer = AutoTokenizer.from_pretrained(model_cfg["target_tokenizer_name_or_path"])
    if target_tokenizer.pad_token is None:
        target_tokenizer.pad_token = target_tokenizer.eos_token

    collator = HybridCollator(
        source_encoder=source_encoder,
        target_tokenizer=target_tokenizer,
        target_max_length=model_cfg["input"]["target_max_length"],
    )

    src_max_len = model_cfg["input"]["source_max_length"]
    train_ds = HybridDataset(train_df, source_max_length=src_max_len)
    val_ds = HybridDataset(val_df, source_max_length=src_max_len)
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["eval_batch_size"], shuffle=False, collate_fn=collator)

    model = HybridSeq2Seq(
        source_vocab_size=source_encoder.vocab_size,
        target_tokenizer_name_or_path=model_cfg["target_tokenizer_name_or_path"],
        encoder_cfg=model_cfg["encoder"],
        decoder_cfg=model_cfg["decoder"],
        aux_weights=model_cfg["auxiliary_losses"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    # Cosine annealing with warmup
    total_steps = len(train_loader) * train_cfg["epochs"]
    warmup_steps = train_cfg.get("warmup_steps", 500)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    best_metric = -1.0
    best_path = output_dir / "best.pt"

    for epoch in range(train_cfg["epochs"]):
        model.train()
        running = 0.0
        epoch_steps = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            global_step += 1
            epoch_steps += 1
            optimizer.zero_grad(set_to_none=True)

            source_ids = batch.source_ids.to(device)
            source_mask = batch.source_mask.to(device)
            target_ids = batch.target_ids.to(device)
            labels = batch.labels.to(device)
            aux = {k: v.to(device) for k, v in batch.aux.items()}

            out = model(
                source_ids=source_ids,
                source_mask=source_mask,
                target_ids=target_ids,
                labels=labels,
                aux=aux,
            )
            out.loss.backward()
            clip_grad_norm_(model.parameters(), train_cfg["grad_clip_norm"])
            optimizer.step()
            scheduler.step()

            running += float(out.loss.item())
            pbar.set_postfix(
                loss=f"{running / epoch_steps:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

            if global_step % train_cfg["eval_every"] == 0:
                metrics, preds, refs = evaluate_model(model, val_loader, target_tokenizer, device)
                logger.info("step=%s metrics=%s", global_step, metrics)
                score = metrics.get("competition_score", metrics["sacrebleu"])
                if score > best_metric:
                    best_metric = score
                    torch.save(model.state_dict(), best_path)
                    write_json(metrics, output_dir / "metrics.json")
                    logger.info("New best competition_score=%.2f at step %d", score, global_step)

        # end-of-epoch checkpoint
        torch.save(model.state_dict(), output_dir / f"epoch_{epoch}.pt")

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    metrics, preds, refs = evaluate_model(model, val_loader, target_tokenizer, device)
    write_json(metrics, output_dir / "metrics.json")

    pred_df = pd.DataFrame(
        {
            "text_id": val_df["text_id"].tolist()[: len(preds)],
            "source": val_df["packed_source"].tolist()[: len(preds)],
            "reference": refs,
            "prediction": preds,
        }
    )
    pred_df.to_csv(output_dir / "val_predictions.csv", index=False)
    errors = build_error_buckets(pred_df)
    errors.to_csv(output_dir / "error_buckets.csv", index=False)

    logger.info("Final metrics: %s", metrics)


if __name__ == "__main__":
    main()
