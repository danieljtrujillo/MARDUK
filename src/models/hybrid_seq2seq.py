from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoTokenizer

from src.models.heads import TokenClassificationHead
from src.models.mamba_encoder import MambaEncoderWrapper


@dataclass
class HybridOutput:
    loss: torch.Tensor
    lm_loss: torch.Tensor
    aux_loss: torch.Tensor
    logits: torch.Tensor
    aux_logits: dict[str, torch.Tensor]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class HybridSeq2Seq(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        target_tokenizer_name_or_path: str,
        encoder_cfg: dict,
        decoder_cfg: dict,
        aux_weights: dict | None = None,
    ) -> None:
        super().__init__()
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer_name_or_path)
        if self.target_tokenizer.pad_token is None:
            self.target_tokenizer.pad_token = self.target_tokenizer.eos_token

        self.encoder = MambaEncoderWrapper(
            vocab_size=source_vocab_size,
            d_model=encoder_cfg["d_model"],
            n_layers=encoder_cfg["n_layers"],
            dropout=encoder_cfg["dropout"],
            use_mamba_if_available=encoder_cfg.get("use_mamba_if_available", True),
            bidirectional=encoder_cfg.get("bidirectional", True),
        )

        d_model = decoder_cfg["d_model"]
        if d_model != encoder_cfg["d_model"]:
            self.enc_proj = nn.Linear(encoder_cfg["d_model"], d_model)
        else:
            self.enc_proj = nn.Identity()

        self.tgt_embed = nn.Embedding(self.target_tokenizer.vocab_size, d_model)
        self.tgt_pos = PositionalEncoding(d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=decoder_cfg["n_heads"],
            dim_feedforward=d_model * decoder_cfg.get("ff_mult", 4),
            dropout=decoder_cfg["dropout"],
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=decoder_cfg["n_layers"])
        self.lm_head = nn.Linear(d_model, self.target_tokenizer.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.name_head = TokenClassificationHead(d_model, 3)
        self.number_head = TokenClassificationHead(d_model, 3)
        self.damage_head = TokenClassificationHead(d_model, 3)
        self.aux_ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.aux_weights = aux_weights or {"name_weight": 0.2, "number_weight": 0.1, "damage_weight": 0.1}

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((size, size), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        target_ids: torch.Tensor,
        labels: torch.Tensor,
        aux: dict[str, torch.Tensor],
    ) -> HybridOutput:
        enc = self.encoder(source_ids)
        memory = self.enc_proj(enc)

        decoder_in = target_ids[:, :-1]
        decoder_labels = labels[:, 1:].contiguous()
        tgt_emb = self.tgt_pos(self.tgt_embed(decoder_in))
        tgt_mask = self._causal_mask(tgt_emb.size(1), tgt_emb.device)

        decoded = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=(source_mask == 0),
        )
        logits = self.lm_head(decoded)

        lm_loss = self.loss_fn(logits.view(-1, logits.size(-1)), decoder_labels.view(-1))

        aux_logits = {
            "name": self.name_head(memory),
            "number": self.number_head(memory),
            "damage": self.damage_head(memory),
        }
        aux_loss = torch.tensor(0.0, device=lm_loss.device)

        if aux:
            if "name_labels" in aux:
                aux_loss = aux_loss + self.aux_weights["name_weight"] * self.aux_ce(
                    aux_logits["name"].view(-1, 3), aux["name_labels"].view(-1)
                )
            if "number_labels" in aux:
                aux_loss = aux_loss + self.aux_weights["number_weight"] * self.aux_ce(
                    aux_logits["number"].view(-1, 3), aux["number_labels"].view(-1)
                )
            if "damage_labels" in aux:
                aux_loss = aux_loss + self.aux_weights["damage_weight"] * self.aux_ce(
                    aux_logits["damage"].view(-1, 3), aux["damage_labels"].view(-1)
                )

        return HybridOutput(
            loss=lm_loss + aux_loss,
            lm_loss=lm_loss,
            aux_loss=aux_loss,
            logits=logits,
            aux_logits=aux_logits,
        )

    @torch.no_grad()
    def generate(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        max_new_tokens: int = 128,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        num_beams: int = 1,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> torch.Tensor:
        bos = bos_token_id if bos_token_id is not None else self.target_tokenizer.pad_token_id
        eos = eos_token_id if eos_token_id is not None else self.target_tokenizer.eos_token_id

        enc = self.encoder(source_ids)
        memory = self.enc_proj(enc)

        if num_beams > 1:
            return self._beam_search(
                memory=memory,
                source_mask=source_mask,
                bos=bos,
                eos=eos,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

        # Greedy decoding
        generated = torch.full((source_ids.size(0), 1), bos, dtype=torch.long, device=source_ids.device)
        finished = torch.zeros(source_ids.size(0), dtype=torch.bool, device=source_ids.device)

        for _ in range(max_new_tokens):
            tgt_emb = self.tgt_pos(self.tgt_embed(generated))
            tgt_mask = self._causal_mask(tgt_emb.size(1), tgt_emb.device)
            decoded = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=(source_mask == 0),
            )
            logits = self.lm_head(decoded[:, -1])
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            finished |= next_token.squeeze(1).eq(eos)
            if finished.all():
                break
        return generated

    def _beam_search(
        self,
        memory: torch.Tensor,
        source_mask: torch.Tensor,
        bos: int,
        eos: int,
        max_new_tokens: int,
        num_beams: int,
        length_penalty: float,
        no_repeat_ngram_size: int,
    ) -> torch.Tensor:
        """Batch beam search over encoder memory."""
        batch_size = memory.size(0)
        device = memory.device

        # Expand for beams: (batch * beams, seq, d)
        memory = memory.repeat_interleave(num_beams, dim=0)
        source_mask = source_mask.repeat_interleave(num_beams, dim=0)

        # (batch * beams, 1)
        generated = torch.full((batch_size * num_beams, 1), bos, dtype=torch.long, device=device)
        # Log probs per beam: (batch * beams,)
        beam_scores = torch.zeros(batch_size * num_beams, device=device)
        # Initialize non-first beams to -inf so only first beam is active initially
        beam_scores[1::num_beams] = -1e9

        finished = torch.zeros(batch_size * num_beams, dtype=torch.bool, device=device)

        for step in range(max_new_tokens):
            tgt_emb = self.tgt_pos(self.tgt_embed(generated))
            tgt_mask = self._causal_mask(tgt_emb.size(1), tgt_emb.device)
            decoded = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=(source_mask == 0),
            )
            logits = self.lm_head(decoded[:, -1])  # (batch*beams, vocab)
            log_probs = torch.log_softmax(logits, dim=-1)

            # Block repeated n-grams
            if no_repeat_ngram_size > 0 and generated.size(1) >= no_repeat_ngram_size:
                for idx in range(generated.size(0)):
                    gen = generated[idx].tolist()
                    ngrams: set = set()
                    for i in range(len(gen) - no_repeat_ngram_size + 1):
                        ngram = tuple(gen[i : i + no_repeat_ngram_size])
                        ngrams.add(ngram)
                    # Block tokens that would create a repeated n-gram
                    prefix = tuple(gen[-(no_repeat_ngram_size - 1) :])
                    for ngram in ngrams:
                        if ngram[:-1] == prefix:
                            log_probs[idx, ngram[-1]] = -1e9

            vocab_size = log_probs.size(-1)
            # Calculate scores: current beam score + new token log prob
            # (batch*beams, vocab)
            next_scores = beam_scores.unsqueeze(1) + log_probs

            # Reshape to (batch, beams * vocab) for top-k selection
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)
            # Select top-k = num_beams per batch
            top_scores, top_indices = torch.topk(next_scores, num_beams, dim=1)

            # Map back to beam and token indices
            beam_indices = top_indices // vocab_size  # which beam
            token_indices = top_indices % vocab_size  # which token

            # Reindex: gather the right beams
            new_generated = []
            for b in range(batch_size):
                for k in range(num_beams):
                    src_beam = b * num_beams + beam_indices[b, k].item()
                    new_tok = token_indices[b, k].item()
                    new_generated.append(
                        torch.cat([generated[src_beam], torch.tensor([new_tok], device=device)])
                    )

            generated = torch.stack(new_generated, dim=0)
            beam_scores = top_scores.view(-1)

            # Check for EOS
            finished = generated[:, -1].eq(eos)
            if finished.all():
                break

        # Apply length penalty and select best beam per batch
        lengths = (generated != eos).sum(dim=1).float()
        final_scores = beam_scores / (lengths ** length_penalty)
        final_scores = final_scores.view(batch_size, num_beams)
        best_beams = final_scores.argmax(dim=1)

        results = []
        for b in range(batch_size):
            idx = b * num_beams + best_beams[b].item()
            results.append(generated[idx])

        # Pad to same length
        max_len = max(r.size(0) for r in results)
        padded = torch.full((batch_size, max_len), eos, dtype=torch.long, device=device)
        for i, r in enumerate(results):
            padded[i, : r.size(0)] = r
        return padded
