"""
LLM summary for file 
-----------------------------------------------------
Noise-Robust Fine-Tuning via Latent Space Alignment
=====================================================

Architecture:
  - Frozen model   : clean Q  → anchor hidden states at layers {8, 18, 19}
  - Trainable model: noisy Q' → live hidden states at layers {8, 18, 19} + logits
  - Projection heads W_8, W_18, W_19 : hidden_dim → proj_dim (trained jointly)

Loss:
  L_total = L_SFT + L_align + L_VICReg

  L_SFT    = CrossEntropy( trainable_model(Q') logits, original_reasoning_trace )

  L_align  = λ1*(1 - cos(W_8(h_8(Q)),   W_8(h_8(Q'))))
           + λ2*(1 - cos(W_18(h_18(Q)),  W_18(h_18(Q'))))
           + λ3*(1 - cos(W_19(h_19(Q)),  W_19(h_19(Q'))))

  L_VICReg = Σ_L [ μ * variance_loss(z_L)    # prevent collapse
                 + ν * covariance_loss(z_L) ] # prevent redundancy

  Hidden states taken from last token position at each forward pass.
  Length mismatch (Q shorter than Q') handled by last-token extraction.

Dataset (JSONL):
  Each line: {
    "original_question"  : str,
    "original_raw"       : str,   SFT target (full reasoning + #### answer)
    "modified_questions" : {
        "adverserials": [Q'_1, Q'_2, Q'_3],
        ...
    }
  }
  Each original question yields up to 3 (Q, Q') training pairs.

Usage:
    python train_noise_robust.py \
        --input-jsonl "shard_outputs_v1/*.jsonl" \
        --output-dir checkpoints_noise_robust \
        --model /Qwen2.5-1.5B-Instruct_gsm8k \
        --hook-layers 8 18 19 \
        --proj-dim 256 \
        --lambda1 0.5 \
        --lambda2 1.0 \
        --lambda3 1.0 \
        --vicreg-mu 1.0 \
        --vicreg-nu 0.1 \
        --batch-size 4 \
        --lr 2e-5 \
        --epochs 3
"""

import argparse
import json
import os
import random
import math
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm



DEFAULT_MODEL       = ""
DEFAULT_HOOK_LAYERS = [8, 18, 19]
DEFAULT_PROJ_DIM    = 256
IGNORE_INDEX        = -100   # standard HuggingFace ignore index for CE loss



def build_prompt(question: str) -> str:
    return (
        f"Solve this math problem step by step:\n\n"
        f"{question}\n\n"
        f"Provide your final answer in the format:\n"
        f"[reasoning steps]\n####\n[final answer (just the number)]"
    )



class NoisyPairDataset(Dataset):
    """
    Yields (clean_question, noisy_question, reasoning_trace) triplets.
    Each JSONL record contributes up to `max_variants` pairs.
    """

    def __init__(self, jsonl_paths: List[str], max_variants: int = 3):
        self.triplets = []
        for path in jsonl_paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    clean_q   = record.get("original_question")
                    raw_trace = record.get("original_raw")          # SFT target
                    adverserials = record.get("modified_questions", {}).get("adverserials", [])

                    if not clean_q or not raw_trace or not adverserials:
                        continue

                    for noisy_q in adverserials[:max_variants]:
                        if noisy_q:
                            self.triplets.append((clean_q, noisy_q, raw_trace))

        print(f"Dataset: {len(self.triplets)} (clean, noisy, trace) triplets "
              f"from {len(jsonl_paths)} file(s)")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]


def collate_fn(batch, tokenizer, max_length: int = 512):
    """
    Tokenizes a batch of (clean_q, noisy_q, trace) triplets.

    Returns dict with:
      clean_input_ids    : (B, T_clean)
      clean_attention_mask
      noisy_input_ids    : (B, T_noisy)   — T_noisy >= T_clean always
      noisy_attention_mask
      sft_input_ids      : (B, T_sft)     — noisy prompt + trace (for CE loss)
      sft_labels         : (B, T_sft)     — IGNORE_INDEX on prompt, trace tokens as targets
      sft_attention_mask
    """
    clean_qs, noisy_qs, traces = zip(*batch)

    # Clean prompts (anchor — no generation target needed)
    clean_prompts = [build_prompt(q) for q in clean_qs]
    clean_enc = tokenizer(
        clean_prompts, return_tensors="pt",
        padding=True, truncation=True, max_length=max_length
    )

    # Noisy prompts (used for hidden state extraction only)
    noisy_prompts = [build_prompt(q) for q in noisy_qs]
    noisy_enc = tokenizer(
        noisy_prompts, return_tensors="pt",
        padding=True, truncation=True, max_length=max_length
    )

    # SFT: noisy prompt + trace concatenated — labels mask the prompt part
    sft_input_ids_list = []
    sft_labels_list    = []

    for noisy_prompt, trace in zip(noisy_prompts, traces):
        MIN_TRACE_TOKENS = 32  # guarantee at least this many trace tokens
        prompt_ids = tokenizer(
            noisy_prompt, add_special_tokens=True,
            truncation=True, max_length=max_length - MIN_TRACE_TOKENS
        )["input_ids"]
        trace_ids = tokenizer(
            trace, add_special_tokens=False,
            truncation=True, max_length=max_length - len(prompt_ids)
        )["input_ids"]

        input_ids = prompt_ids + trace_ids
        # Labels: IGNORE on prompt tokens, actual ids on trace tokens
        labels    = [IGNORE_INDEX] * len(prompt_ids) + trace_ids

        sft_input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
        sft_labels_list.append(torch.tensor(labels,    dtype=torch.long))

    # Pad SFT sequences
    sft_input_ids = torch.nn.utils.rnn.pad_sequence(
        sft_input_ids_list, batch_first=True,
        padding_value=tokenizer.pad_token_id or 0
    )
    sft_labels = torch.nn.utils.rnn.pad_sequence(
        sft_labels_list, batch_first=True,
        padding_value=IGNORE_INDEX
    )
    sft_attention_mask = (sft_input_ids != (tokenizer.pad_token_id or 0)).long()

    return {
        "clean_input_ids":      clean_enc["input_ids"],
        "clean_attention_mask": clean_enc["attention_mask"],
        "noisy_input_ids":      noisy_enc["input_ids"],
        "noisy_attention_mask": noisy_enc["attention_mask"],
        "sft_input_ids":        sft_input_ids,
        "sft_labels":           sft_labels,
        "sft_attention_mask":   sft_attention_mask,
    }



class ProjectionHead(nn.Module):
    """
    Small 2-layer MLP projection: hidden_dim → hidden_dim//2 → proj_dim
    Applied to last-token hidden states before cosine alignment loss.
    """
    def __init__(self, hidden_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjectionHeads(nn.Module):
    """Container for per-layer projection heads."""
    def __init__(self, hook_layers: List[int], hidden_dim: int, proj_dim: int):
        super().__init__()
        self.hook_layers = hook_layers
        self.heads = nn.ModuleDict({
            str(L): ProjectionHead(hidden_dim, proj_dim)
            for L in hook_layers
        })

    def forward(self, hidden_states_dict: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        hidden_states_dict: {layer_idx: tensor (B, hidden_dim)}  — last token already extracted
        Returns: {layer_idx: tensor (B, proj_dim)}
        """
        return {
            L: self.heads[str(L)](h.float())
            for L, h in hidden_states_dict.items()
        }



def cosine_alignment_loss(
    proj_clean: Dict[int, torch.Tensor],
    proj_noisy: Dict[int, torch.Tensor],
    lambdas:    Dict[int, float],
) -> torch.Tensor:
    """
    L_align = Σ_L λ_L * mean(1 - cosine_sim(z_L(Q), z_L(Q')))
    """
    loss = torch.tensor(0.0, device=next(iter(proj_clean.values())).device)
    for L, z_clean in proj_clean.items():
        z_noisy = proj_noisy[L]
        cos_sim = F.cosine_similarity(z_clean, z_noisy, dim=-1)   # (B,)
        loss = loss + lambdas[L] * (1.0 - cos_sim).mean()
    return loss


def vicreg_loss(
    proj_noisy: Dict[int, torch.Tensor],
    mu:    float = 0.1,
    nu:    float = 0.01,
    gamma: float = 1.0,
    accum_buffer: Optional[Dict[int, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    VICReg variance + covariance terms on projected noisy representations.

    accum_buffer: if provided, representations are accumulated across microbatches
                  before computing statistics — gives VICReg a larger effective
                  batch size without increasing GPU memory per step.
                  Pass the same dict across microbatch steps, reset after optimizer.step().

    variance  : std of each proj dimension across batch should be > gamma
    covariance: off-diagonal entries of cov matrix should be near 0
    """
    device = next(iter(proj_noisy.values())).device
    loss   = torch.tensor(0.0, device=device)

    for L, z in proj_noisy.items():
        # Accumulate across microbatches if buffer provided
        if accum_buffer is not None:
            if L not in accum_buffer:
                accum_buffer[L] = z.detach()
            else:
                accum_buffer[L] = torch.cat([accum_buffer[L], z.detach()], dim=0)
            z_for_stats = accum_buffer[L]
        else:
            z_for_stats = z

        B, D = z_for_stats.shape
        if B < 2:
            continue

        z_c = z_for_stats - z_for_stats.mean(dim=0)   # center

        # Variance loss — std per dimension should exceed gamma
        std      = z_c.std(dim=0)                              # (D,)
        var_loss = F.relu(gamma - std).pow(2).mean()

        # Covariance loss — off-diagonal should be near zero
        cov      = (z_c.T @ z_c) / (B - 1)                    # (D, D)
        diag_mask = torch.eye(D, device=device, dtype=torch.bool)
        cov_loss = cov[~diag_mask].pow(2).sum() / D

        loss = loss + mu * var_loss + nu * cov_loss

    return loss



def extract_last_token_hidden_states(
    outputs,
    attention_mask: torch.Tensor,
    hook_layers: List[int],
) -> Dict[int, torch.Tensor]:
    """
    Extract last non-padding token hidden state at each hook layer.
    outputs.hidden_states: tuple of (n_layers+1) tensors, each (B, T, hidden)
    Returns: {layer_idx: (B, hidden_dim)}
    """
    # Last real token index per sample
    last_token_idx = attention_mask.sum(dim=1) - 1   # (B,)

    result = {}
    for L in hook_layers:
        hs = outputs.hidden_states[L]                 # (B, T, hidden)
        # Gather last token for each sample in batch
        idx = last_token_idx.view(-1, 1, 1).expand(-1, 1, hs.size(-1))
        result[L] = hs.gather(1, idx).squeeze(1)      # (B, hidden_dim)

    return result



def training_step(
    batch:            dict,
    frozen_model:     nn.Module,
    trainable_model:  nn.Module,
    proj_heads:       ProjectionHeads,
    hook_layers:      List[int],
    lambdas:          Dict[int, float],
    vicreg_mu:        float,
    vicreg_nu:        float,
    device:           str,
    accum_buffer:     Optional[Dict[int, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, dict]:

    # Move batch to device
    clean_ids   = batch["clean_input_ids"].to(device)
    clean_mask  = batch["clean_attention_mask"].to(device)
    noisy_ids   = batch["noisy_input_ids"].to(device)
    noisy_mask  = batch["noisy_attention_mask"].to(device)
    sft_ids     = batch["sft_input_ids"].to(device)
    sft_labels  = batch["sft_labels"].to(device)
    sft_mask    = batch["sft_attention_mask"].to(device)

    #  1 Frozen model on clean Q -> anchor hidden states
    with torch.no_grad():
        clean_outputs = frozen_model(
            input_ids=clean_ids,
            attention_mask=clean_mask,
            output_hidden_states=True,
        )
    hs_clean = extract_last_token_hidden_states(
        clean_outputs, clean_mask, hook_layers)

    #  2 Trainable model on noisy Q' -> live hidden states 
    noisy_outputs = trainable_model(
        input_ids=noisy_ids,
        attention_mask=noisy_mask,
        output_hidden_states=True,
    )
    hs_noisy = extract_last_token_hidden_states(
        noisy_outputs, noisy_mask, hook_layers)

    #  3 Project both to alignment space 
    proj_clean = proj_heads(hs_clean)
    proj_noisy = proj_heads(hs_noisy)

    #  4 SFT loss on noisy input -> original trace 
    sft_outputs = trainable_model(
        input_ids=sft_ids,
        attention_mask=sft_mask,
        output_hidden_states=False,
    )
    logits = sft_outputs.logits                                # (B, T, vocab)
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = sft_labels[:, 1:].contiguous()
    # Guard against all-IGNORE batches (happens when noisy prompt fills max_length)
    valid_tokens = (shift_labels != IGNORE_INDEX).sum()
    if valid_tokens == 0:
        l_sft = torch.tensor(0.0, device=shift_logits.device, requires_grad=True)
    else:
        l_sft = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )

    #  5 Alignment loss 
    l_align = cosine_alignment_loss(proj_clean, proj_noisy, lambdas)

    #  6. VICReg loss 
    l_vicreg = vicreg_loss(proj_noisy, mu=vicreg_mu, nu=vicreg_nu, accum_buffer=accum_buffer)

    #7 Total 
    l_total = l_sft + l_align + l_vicreg

    metrics = {
        "loss_total":  l_total.item(),
        "loss_sft":    l_sft.item(),
        "loss_align":  l_align.item(),
        "loss_vicreg": l_vicreg.item(),
    }

    return l_total, metrics


# Checkpointing 

def save_checkpoint(trainable_model, proj_heads, optimizer, epoch, step,
                    metrics, output_dir):
    ckpt_dir = Path(output_dir) / f"checkpoint_epoch{epoch}_step{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainable_model.save_pretrained(str(ckpt_dir))
    torch.save({
        "proj_heads":  proj_heads.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "epoch":       epoch,
        "step":        step,
        "metrics":     metrics,
    }, ckpt_dir / "training_state.pt")
    print(f"Checkpoint saved: {ckpt_dir}")


#  Main 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl",   type=str,   required=True,
                        help="Glob pattern for adversarial JSONL files")
    parser.add_argument("--output-dir",    type=str,   default="checkpoints_noise_robust")
    parser.add_argument("--model",         type=str,   default=DEFAULT_MODEL)
    parser.add_argument("--hook-layers",   type=int,   nargs="+", default=DEFAULT_HOOK_LAYERS)
    parser.add_argument("--proj-dim",      type=int,   default=DEFAULT_PROJ_DIM)
    parser.add_argument("--lambda1",       type=float, default=0.5,
                        help="Lambda for layer 8  alignment (default: 0.5)")
    parser.add_argument("--lambda2",       type=float, default=1.0,
                        help="Lambda for layer 18 alignment (default: 1.0)")
    parser.add_argument("--lambda3",       type=float, default=1.0,
                        help="Lambda for layer 19 alignment (default: 1.0)")
    parser.add_argument("--vicreg-mu",     type=float, default=0.1,
                        help="VICReg variance weight (default: 1.0)")
    parser.add_argument("--vicreg-nu",     type=float, default=0.01,
                        help="VICReg covariance weight (default: 0.1)")
    parser.add_argument("--batch-size",    type=int,   default=4)
    parser.add_argument("--lr",            type=float, default=2e-5)
    parser.add_argument("--epochs",        type=int,   default=3)
    parser.add_argument("--max-length",    type=int,   default=512)
    parser.add_argument("--save-every",    type=int,   default=200,
                        help="Save checkpoint every N steps (default: 200)")
    parser.add_argument("--grad-accum-steps", type=int, default=8,
                        help="Gradient accumulation steps — effective_batch = batch_size * accum (default: 8)")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--max-variants",  type=int,   default=3,
                        help="Max noisy variants per question (default: 3)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")


    hook_layers = args.hook_layers
    lambda_values = [args.lambda1, args.lambda2, args.lambda3]
    lambdas = {L: lam for L, lam in zip(hook_layers, lambda_values)}
    print(f"Hook layers: {hook_layers}  Lambdas: {lambdas}")

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading frozen anchor model...")
    frozen_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    frozen_model.eval()
    for p in frozen_model.parameters():
        p.requires_grad = False
    print("Frozen model loaded and gradients disabled.")


    print("Loading trainable model...")
    trainable_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    trainable_model.train()


    hidden_dim = trainable_model.config.hidden_size
    print(f"Hidden dim: {hidden_dim}  Proj dim: {args.proj_dim}")


    proj_heads = ProjectionHeads(hook_layers, hidden_dim, args.proj_dim).to(device)
    print(f"Projection heads: {sum(p.numel() for p in proj_heads.parameters()):,} params")


    optimizer = torch.optim.AdamW(
        list(trainable_model.parameters()) + list(proj_heads.parameters()),
        lr=args.lr, weight_decay=0.01
    )


    jsonl_files = glob(args.input_jsonl)
    if not jsonl_files:
        raise FileNotFoundError(f"No files found: {args.input_jsonl}")
    print(f"Found {len(jsonl_files)} JSONL file(s)")

    dataset = NoisyPairDataset(jsonl_files, max_variants=args.max_variants)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length),
    )


    global_step   = 0
    accum_steps   = args.grad_accum_steps
    eff_batch     = args.batch_size * accum_steps
    log_path      = Path(args.output_dir) / "training_log.jsonl"

    print(f"Effective batch size: {eff_batch} "
          f"(micro={args.batch_size} x accum={accum_steps})")

    for epoch in range(1, args.epochs + 1):
        epoch_metrics  = {k: [] for k in ["loss_total", "loss_sft", "loss_align", "loss_vicreg"]}
        accum_metrics  = {k: 0.0 for k in ["loss_total", "loss_sft", "loss_align", "loss_vicreg"]}
        vicreg_buffer  = {}   # accumulates proj_noisy across microbatches for VICReg stats
        micro_count    = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):

            loss, metrics = training_step(
                batch=batch,
                frozen_model=frozen_model,
                trainable_model=trainable_model,
                proj_heads=proj_heads,
                hook_layers=hook_layers,
                lambdas=lambdas,
                vicreg_mu=args.vicreg_mu,
                vicreg_nu=args.vicreg_nu,
                device=device,
                accum_buffer=vicreg_buffer,
            )

            if not torch.isfinite(loss):
                print(f"[Epoch {epoch} batch {batch_idx}] NaN/Inf loss, skipping microbatch.")
                continue

            # Scale loss by accum steps so gradients average correctly
            (loss / accum_steps).backward()

            micro_count += 1
            for k, v in metrics.items():
                if math.isfinite(v):
                    accum_metrics[k] += v / accum_steps


            is_last_batch  = (batch_idx == len(dataloader) - 1)
            do_update      = (micro_count % accum_steps == 0) or is_last_batch

            if do_update and micro_count > 0:
                torch.nn.utils.clip_grad_norm_(trainable_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                vicreg_buffer = {}   # reset VICReg buffer after each optimizer step

                for k, v in accum_metrics.items():
                    if math.isfinite(v):
                        epoch_metrics[k].append(v)

                pbar.set_postfix({
                    "total": f"{accum_metrics['loss_total']:.3f}",
                    "sft":   f"{accum_metrics['loss_sft']:.3f}",
                    "align": f"{accum_metrics['loss_align']:.3f}",
                    "vic":   f"{accum_metrics['loss_vicreg']:.3f}",
                    "eff_b": str(eff_batch),
                })

                with open(log_path, "a") as lf:
                    lf.write(json.dumps({
                        "epoch": epoch, "step": global_step, **accum_metrics
                    }) + "\n")

                if global_step % args.save_every == 0:
                    save_checkpoint(
                        trainable_model, proj_heads, optimizer,
                        epoch, global_step, accum_metrics, args.output_dir
                    )

                # Reset accum metrics for next accumulation window
                accum_metrics = {k: 0.0 for k in accum_metrics}
                micro_count   = 0

        # Epoch summary
        print(f"\nEpoch {epoch} summary:")
        for k, vals in epoch_metrics.items():
            print(f"  {k:15s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        save_checkpoint(
            trainable_model, proj_heads, optimizer,
            epoch, global_step, accum_metrics, args.output_dir
        )

    print(f"\nTraining complete. Checkpoints in: {args.output_dir}")
    print(f"Training log: {log_path}")


if __name__ == "__main__":
    main()