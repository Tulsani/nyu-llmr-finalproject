"""
Noise-Robust Fine-Tuning via Latent Space Alignment

Usage:
    python train_noise_robust_attn_clean.py \
        --input-jsonl "pit-train.jsonl" \
        --output-dir checkpoints_v3 \
        --model jahyungu/Qwen2.5-1.5B-Instruct_gsm8k \
        --hook-layers 8 18 19 \
        --proj-dim 256 \
        --lambda1 0.5 --lambda2 1.0 --lambda3 1.0 \
        --vicreg-mu 0.1 --vicreg-nu 0.01 \
        --attn-lambda 0.3 --attn-mode match \
        --clean-sft-weight 0.5 \
        --max-variants 2 \
        --batch-size 4 --grad-accum-steps 8 \
        --lr 5e-6 --warmup-steps 50 --epochs 3
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


DEFAULT_MODEL       = "jahyungu/Qwen2.5-1.5B-Instruct_gsm8k"
DEFAULT_HOOK_LAYERS = [8, 18, 19]
DEFAULT_PROJ_DIM    = 256
IGNORE_INDEX        = -100



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

    Default max_variants=2 (down from 3) to reduce trace repetition which
    causes the model to memorise traces rather than learn noise invariance.
    """

    def __init__(self, jsonl_paths: List[str], max_variants: int = 2):
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

                    clean_q      = record.get("original_question")
                    raw_trace    = record.get("original_raw")
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
      clean_input_ids       : (B, T_clean)   — clean prompt only (for alignment)
      clean_attention_mask
      noisy_input_ids       : (B, T_noisy)   — noisy prompt only (for alignment)
      noisy_attention_mask
      sft_noisy_input_ids   : (B, T_sft)     — noisy prompt + trace (noisy SFT loss)
      sft_noisy_labels      : (B, T_sft)
      sft_noisy_mask
      sft_clean_input_ids   : (B, T_sft)     — clean prompt + trace (clean SFT anchor)
      sft_clean_labels      : (B, T_sft)
      sft_clean_mask
    """
    clean_qs, noisy_qs, traces = zip(*batch)

    clean_prompts = [build_prompt(q) for q in clean_qs]
    clean_enc = tokenizer(
        clean_prompts, return_tensors="pt",
        padding=True, truncation=True, max_length=max_length
    )

    noisy_prompts = [build_prompt(q) for q in noisy_qs]
    noisy_enc = tokenizer(
        noisy_prompts, return_tensors="pt",
        padding=True, truncation=True, max_length=max_length
    )

    def build_sft_sequences(prompts):
        input_ids_list, labels_list = [], []
        for prompt, trace in zip(prompts, traces):
            MIN_TRACE_TOKENS = 32
            prompt_ids = tokenizer(
                prompt, add_special_tokens=True,
                truncation=True, max_length=max_length - MIN_TRACE_TOKENS
            )["input_ids"]
            trace_ids = tokenizer(
                trace, add_special_tokens=False,
                truncation=True, max_length=max_length - len(prompt_ids)
            )["input_ids"]
            ids    = prompt_ids + trace_ids
            labels = [IGNORE_INDEX] * len(prompt_ids) + trace_ids
            input_ids_list.append(torch.tensor(ids,    dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        pad_id = tokenizer.pad_token_id or 0
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=pad_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=IGNORE_INDEX)
        mask = (input_ids != pad_id).long()
        return input_ids, labels, mask

    sft_noisy_ids, sft_noisy_labels, sft_noisy_mask = build_sft_sequences(noisy_prompts)
    sft_clean_ids, sft_clean_labels, sft_clean_mask = build_sft_sequences(clean_prompts)

    return {
        "clean_input_ids":       clean_enc["input_ids"],
        "clean_attention_mask":  clean_enc["attention_mask"],
        "noisy_input_ids":       noisy_enc["input_ids"],
        "noisy_attention_mask":  noisy_enc["attention_mask"],
        # noisy SFT
        "sft_noisy_input_ids":   sft_noisy_ids,
        "sft_noisy_labels":      sft_noisy_labels,
        "sft_noisy_mask":        sft_noisy_mask,
        # clean SFT anchor  ← new
        "sft_clean_input_ids":   sft_clean_ids,
        "sft_clean_labels":      sft_clean_labels,
        "sft_clean_mask":        sft_clean_mask,
    }




class ProjectionHead(nn.Module):
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
    def __init__(self, hook_layers: List[int], hidden_dim: int, proj_dim: int):
        super().__init__()
        self.hook_layers = hook_layers
        self.heads = nn.ModuleDict({
            str(L): ProjectionHead(hidden_dim, proj_dim)
            for L in hook_layers
        })

    def forward(self, hidden_states_dict: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        return {
            L: self.heads[str(L)](h.float())
            for L, h in hidden_states_dict.items()
        }




def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on next-token prediction, ignoring IGNORE_INDEX positions."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid_tokens = (shift_labels != IGNORE_INDEX).sum()
    if valid_tokens == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


def cosine_alignment_loss(
    proj_clean: Dict[int, torch.Tensor],
    proj_noisy: Dict[int, torch.Tensor],
    lambdas:    Dict[int, float],
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=next(iter(proj_clean.values())).device)
    for L, z_clean in proj_clean.items():
        z_noisy = proj_noisy[L]
        cos_sim = F.cosine_similarity(z_clean, z_noisy, dim=-1)
        loss = loss + lambdas[L] * (1.0 - cos_sim).mean()
    return loss


def vicreg_loss(
    proj_noisy:   Dict[int, torch.Tensor],
    mu:           float = 0.1,
    nu:           float = 0.01,
    gamma:        float = 1.0,
    accum_buffer: Optional[Dict[int, torch.Tensor]] = None,
) -> torch.Tensor:
    device = next(iter(proj_noisy.values())).device
    loss   = torch.tensor(0.0, device=device)

    for L, z in proj_noisy.items():
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

        z_c      = z_for_stats - z_for_stats.mean(dim=0)
        std      = z_c.std(dim=0)
        var_loss = F.relu(gamma - std).pow(2).mean()
        cov      = (z_c.T @ z_c) / (B - 1)
        diag_mask = torch.eye(D, device=device, dtype=torch.bool)
        cov_loss = cov[~diag_mask].pow(2).sum() / D
        loss     = loss + mu * var_loss + nu * cov_loss

    return loss


def attention_entropy(attn_weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Per-head Shannon entropy at the last query token. Shape: (B, n_heads).
    Length-agnostic — works regardless of clean/noisy sequence length difference.
    """
    last_attn = attn_weights[:, :, -1, :]   # (B, n_heads, T_k)
    p = last_attn.clamp(min=eps)
    return -(p * p.log()).sum(dim=-1)        # (B, n_heads)


def attention_entropy_alignment_loss(
    clean_attns: Dict[int, torch.Tensor],
    noisy_attns: Dict[int, torch.Tensor],
    lambdas:     Dict[int, float],
    mode:        str = "match",
) -> torch.Tensor:
    device = next(iter(clean_attns.values())).device
    loss   = torch.tensor(0.0, device=device)

    for L in clean_attns:
        if L not in noisy_attns:
            continue
        H_clean = attention_entropy(clean_attns[L])
        H_noisy = attention_entropy(noisy_attns[L])

        if mode == "match":
            layer_loss = F.mse_loss(H_noisy, H_clean.detach())
        elif mode == "suppress_noisy":
            layer_loss = H_noisy.mean()
        else:
            raise ValueError(f"Unknown attn_mode: {mode!r}")

        loss = loss + lambdas.get(L, 1.0) * layer_loss

    return loss



def extract_last_token_hidden_states(
    outputs, attention_mask: torch.Tensor, hook_layers: List[int],
) -> Dict[int, torch.Tensor]:
    last_token_idx = attention_mask.sum(dim=1) - 1
    result = {}
    for L in hook_layers:
        hs  = outputs.hidden_states[L]
        idx = last_token_idx.view(-1, 1, 1).expand(-1, 1, hs.size(-1))
        result[L] = hs.gather(1, idx).squeeze(1)
    return result


def extract_hook_layer_attentions(
    outputs, hook_layers: List[int], detach: bool = True,
) -> Dict[int, torch.Tensor]:
    result = {}
    for L in hook_layers:
        attn = outputs.attentions[L]
        result[L] = attn.detach() if detach else attn
    return result



def get_lr_scale(step: int, warmup_steps: int) -> float:
    """Linear warmup from 0 → 1 over warmup_steps, then constant 1."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, step / warmup_steps)



def training_step(
    batch:            dict,
    frozen_model:     nn.Module,
    trainable_model:  nn.Module,
    proj_heads:       ProjectionHeads,
    hook_layers:      List[int],
    lambdas:          Dict[int, float],
    vicreg_mu:        float,
    vicreg_nu:        float,
    attn_lambda:      float,
    attn_mode:        str,
    clean_sft_weight: float,
    device:           str,
    accum_buffer:     Optional[Dict[int, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    L_total = L_SFT + L_align + L_VICReg + attn_lambda * L_attn_entropy

    L_SFT = (1 - clean_sft_weight) * CE(noisy → trace)
           +      clean_sft_weight  * CE(clean → trace)

    The clean SFT term anchors the model to the clean distribution every step,
    directly countering the clean accuracy regression seen in earlier runs where
    SFT was 100% noisy-input.
    """

    clean_ids   = batch["clean_input_ids"].to(device)
    clean_mask  = batch["clean_attention_mask"].to(device)
    noisy_ids   = batch["noisy_input_ids"].to(device)
    noisy_mask  = batch["noisy_attention_mask"].to(device)

    sft_noisy_ids    = batch["sft_noisy_input_ids"].to(device)
    sft_noisy_labels = batch["sft_noisy_labels"].to(device)
    sft_noisy_mask   = batch["sft_noisy_mask"].to(device)

    sft_clean_ids    = batch["sft_clean_input_ids"].to(device)
    sft_clean_labels = batch["sft_clean_labels"].to(device)
    sft_clean_mask   = batch["sft_clean_mask"].to(device)


    with torch.no_grad():
        clean_outputs = frozen_model(
            input_ids=clean_ids,
            attention_mask=clean_mask,
            output_hidden_states=True,
            output_attentions=True,
        )
    hs_clean    = extract_last_token_hidden_states(clean_outputs, clean_mask, hook_layers)
    clean_attns = extract_hook_layer_attentions(clean_outputs, hook_layers, detach=True)
    del clean_outputs


    noisy_outputs = trainable_model(
        input_ids=noisy_ids,
        attention_mask=noisy_mask,
        output_hidden_states=True,
        output_attentions=True,
    )
    hs_noisy    = extract_last_token_hidden_states(noisy_outputs, noisy_mask, hook_layers)
    noisy_attns = extract_hook_layer_attentions(noisy_outputs, hook_layers, detach=False)
    del noisy_outputs

    
    proj_clean = proj_heads(hs_clean)
    proj_noisy = proj_heads(hs_noisy)

    
    noisy_sft_out = trainable_model(
        input_ids=sft_noisy_ids,
        attention_mask=sft_noisy_mask,
        output_hidden_states=False,
        output_attentions=False,
    )
    l_sft_noisy = sft_loss(noisy_sft_out.logits, sft_noisy_labels)

   
    # Trains the model on clean Q -> trace alongside noisy Q -> trace.
    # This directly prevents the clean accuracy regression observed in v1/v2
    # where 100% noisy SFT gradually drifted the model off the clean distribution.
    clean_sft_out = trainable_model(
        input_ids=sft_clean_ids,
        attention_mask=sft_clean_mask,
        output_hidden_states=False,
        output_attentions=False,
    )
    l_sft_clean = sft_loss(clean_sft_out.logits, sft_clean_labels)

    # Weighted combination — clean_sft_weight=0.5 means equal contribution
    noisy_weight = 1.0 - clean_sft_weight
    l_sft = noisy_weight * l_sft_noisy + clean_sft_weight * l_sft_clean

   
    l_align = cosine_alignment_loss(proj_clean, proj_noisy, lambdas)

    
    l_vicreg = vicreg_loss(proj_noisy, mu=vicreg_mu, nu=vicreg_nu,
                           accum_buffer=accum_buffer)

   
    l_attn = attention_entropy_alignment_loss(
        clean_attns=clean_attns,
        noisy_attns=noisy_attns,
        lambdas=lambdas,
        mode=attn_mode,
    )

   
    l_total = l_sft + l_align + l_vicreg + attn_lambda * l_attn

    metrics = {
        "loss_total":     l_total.item(),
        "loss_sft":       l_sft.item(),
        "loss_sft_noisy": l_sft_noisy.item(),
        "loss_sft_clean": l_sft_clean.item(),
        "loss_align":     l_align.item(),
        "loss_vicreg":    l_vicreg.item(),
        "loss_attn":      l_attn.item(),
    }

    return l_total, metrics




def save_checkpoint(trainable_model, proj_heads, optimizer, epoch, step,
                    metrics, output_dir):
    ckpt_dir = Path(output_dir) / f"checkpoint_epoch{epoch}_step{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainable_model.save_pretrained(str(ckpt_dir))
    torch.save({
        "proj_heads": proj_heads.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "epoch":      epoch,
        "step":       step,
        "metrics":    metrics,
    }, ckpt_dir / "training_state.pt")
    print(f"Checkpoint saved: {ckpt_dir}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl",      type=str,   required=True)
    parser.add_argument("--output-dir",       type=str,   default="checkpoints_noise_robust")
    parser.add_argument("--model",            type=str,   default=DEFAULT_MODEL)
    parser.add_argument("--hook-layers",      type=int,   nargs="+", default=DEFAULT_HOOK_LAYERS)
    parser.add_argument("--proj-dim",         type=int,   default=DEFAULT_PROJ_DIM)
    parser.add_argument("--lambda1",          type=float, default=0.5)
    parser.add_argument("--lambda2",          type=float, default=1.0)
    parser.add_argument("--lambda3",          type=float, default=1.0)
    parser.add_argument("--vicreg-mu",        type=float, default=0.1)
    parser.add_argument("--vicreg-nu",        type=float, default=0.01)
    parser.add_argument("--attn-lambda",      type=float, default=0.3)
    parser.add_argument("--attn-mode",        type=str,   default="match",
                        choices=["match", "suppress_noisy"])
    parser.add_argument("--clean-sft-weight", type=float, default=0.5,
                        help="Weight for clean SFT anchor loss. 0.5 = equal clean/noisy. "
                             "Increase toward 1.0 if clean accuracy regression is severe. "
                             "Set to 0.0 to disable (reverts to v2 behaviour).")
    parser.add_argument("--batch-size",       type=int,   default=4)
    parser.add_argument("--lr",               type=float, default=5e-6,
                        help="Peak learning rate (default: 5e-6, down from 2e-5 in v2)")
    parser.add_argument("--warmup-steps",     type=int,   default=50,
                        help="Linear LR warmup steps (default: 50)")
    parser.add_argument("--epochs",           type=int,   default=3)
    parser.add_argument("--max-length",       type=int,   default=512)
    parser.add_argument("--save-every",       type=int,   default=200)
    parser.add_argument("--grad-accum-steps", type=int,   default=8)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--max-variants",     type=int,   default=2,
                        help="Max noisy variants per question (default: 2, down from 3). "
                             "Reducing from 3 cuts trace repetition which caused overfitting.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    hook_layers   = args.hook_layers
    lambda_values = [args.lambda1, args.lambda2, args.lambda3]
    lambdas       = {L: lam for L, lam in zip(hook_layers, lambda_values)}
    print(f"Hook layers: {hook_layers}  Lambdas: {lambdas}")
    print(f"Attn entropy: λ={args.attn_lambda}  mode={args.attn_mode}")
    print(f"Clean SFT weight: {args.clean_sft_weight}  "
          f"(noisy weight: {1.0 - args.clean_sft_weight})")
    print(f"LR: {args.lr}  Warmup steps: {args.warmup_steps}")
    print(f"Max variants per question: {args.max_variants}")

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading frozen anchor model...")
    frozen_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    frozen_model.eval()
    for p in frozen_model.parameters():
        p.requires_grad = False
    print("Frozen model loaded and gradients disabled.")

    print("Loading trainable model...")
    trainable_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager",
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

    global_step  = 0
    accum_steps  = args.grad_accum_steps
    eff_batch    = args.batch_size * accum_steps
    log_path     = Path(args.output_dir) / "training_log.jsonl"
    metric_keys  = ["loss_total", "loss_sft", "loss_sft_noisy", "loss_sft_clean",
                    "loss_align", "loss_vicreg", "loss_attn"]

    print(f"Effective batch size: {eff_batch} "
          f"(micro={args.batch_size} x accum={accum_steps})")

    for epoch in range(1, args.epochs + 1):
        epoch_metrics = {k: [] for k in metric_keys}
        accum_metrics = {k: 0.0  for k in metric_keys}
        vicreg_buffer = {}
        micro_count   = 0
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
                attn_lambda=args.attn_lambda,
                attn_mode=args.attn_mode,
                clean_sft_weight=args.clean_sft_weight,
                device=device,
                accum_buffer=vicreg_buffer,
            )

            if not torch.isfinite(loss):
                print(f"[Epoch {epoch} batch {batch_idx}] NaN/Inf — skipping.")
                continue

            (loss / accum_steps).backward()

            micro_count += 1
            for k, v in metrics.items():
                if math.isfinite(v):
                    accum_metrics[k] += v / accum_steps

            is_last_batch = (batch_idx == len(dataloader) - 1)
            do_update     = (micro_count % accum_steps == 0) or is_last_batch

            if do_update and micro_count > 0:
                torch.nn.utils.clip_grad_norm_(trainable_model.parameters(), 1.0)

                
                lr_scale = get_lr_scale(global_step + 1, args.warmup_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr * lr_scale

                optimizer.step()
                optimizer.zero_grad()

                global_step  += 1
                vicreg_buffer = {}

                for k, v in accum_metrics.items():
                    if math.isfinite(v):
                        epoch_metrics[k].append(v)

                pbar.set_postfix({
                    "total":  f"{accum_metrics['loss_total']:.3f}",
                    "sft_n":  f"{accum_metrics['loss_sft_noisy']:.3f}",
                    "sft_c":  f"{accum_metrics['loss_sft_clean']:.3f}",
                    "align":  f"{accum_metrics['loss_align']:.3f}",
                    "attn":   f"{accum_metrics['loss_attn']:.3f}",
                    "lr":     f"{args.lr * lr_scale:.2e}",
                })

                with open(log_path, "a") as lf:
                    lf.write(json.dumps({
                        "epoch": epoch, "step": global_step,
                        "lr": args.lr * lr_scale,
                        **accum_metrics,
                    }) + "\n")

                if global_step % args.save_every == 0:
                    save_checkpoint(
                        trainable_model, proj_heads, optimizer,
                        epoch, global_step, accum_metrics, args.output_dir
                    )

                accum_metrics = {k: 0.0 for k in metric_keys}
                micro_count   = 0

        print(f"\nEpoch {epoch} summary:")
        for k, vals in epoch_metrics.items():
            print(f"  {k:18s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        save_checkpoint(
            trainable_model, proj_heads, optimizer,
            epoch, global_step, accum_metrics, args.output_dir
        )

    print(f"\nTraining complete. Checkpoints in: {args.output_dir}")
    print(f"Training log: {log_path}")


if __name__ == "__main__":
    main()