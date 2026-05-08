"""
LLM summary for file 
-----------------------------------------------------
Layer Noise Divergence Analysis
=================================
For each (Q, Q') pair in the adversarial JSONL dataset, extract hidden states
at candidate layers and compute:

  - KL divergence   : KL( h_L(Q) || h_L(Q') )  — distribution shift
  - Cosine distance : 1 - cosine( h_L(Q), h_L(Q') )  — geometric displacement

This validates whether candidate hook layers are actually sensitive to noise
before committing them to the training regularization loss.

Candidate layer groups (from prior analysis):
  Group A — early stability   (adjacent KL analysis) : [6, 7, 8, 9]
  Group B — late stability    (adjacent KL analysis) : [23, 24, 25]
  Group C — last-layer stable (M1 analysis)          : [18, 19, 20]

Usage:
    python layer_noise_divergence.py \
        --input shard_outputs_v1/gsm8k_adv_4982_5482.jsonl \
        --output-dir noise_divergence_results \
        --candidate-layers 6 7 8 9 18 19 20 23 24 25 \
        --n-pairs 200
"""

import argparse
import json
import os
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL   = ""
# Layer groups from prior analysis
GROUP_A = [6, 7, 8, 9]        # early stability  (adjacent KL)
GROUP_B = [23, 24, 25]        # late stability   (adjacent KL)
GROUP_C = [18, 19, 20]        # M1 stable        (KL vs last)
DEFAULT_LAYERS  = sorted(set(GROUP_A + GROUP_B + GROUP_C))



def build_prompt(question: str) -> str:
    return (
        f"Solve this math problem step by step:\n\n"
        f"{question}\n\n"
        f"Provide your final answer in the format:\n"
        f"[reasoning steps]\n####\n[final answer (just the number)]"
    )


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    """Symmetric KL divergence between two hidden state vectors."""
    p = F.softmax(p.float(), dim=-1) + eps
    q = F.softmax(q.float(), dim=-1) + eps
    p = p / p.sum()
    q = q / q.sum()
    kl_pq = (p * (p / q).log()).sum().item()
    kl_qp = (q * (q / p).log()).sum().item()
    return 0.5 * (kl_pq + kl_qp)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine distance (1 - cosine similarity) between two vectors."""
    return 1.0 - F.cosine_similarity(
        a.float().unsqueeze(0), b.float().unsqueeze(0)
    ).item()



@torch.no_grad()
def extract_hidden_states_at_layers(model, tokenizer, prompt: str,
                                    device: str, layers: list):
    """
    Forward pass extracting mean-pooled hidden states only at requested layers.
    Returns dict {layer_idx: tensor of shape (hidden_dim,)}
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=512).to(device)
    outputs = model(**inputs, output_hidden_states=True)
    # outputs.hidden_states: tuple of (n_layers+1) tensors (1, seq_len, hidden)
    result = {}
    for L in layers:
        result[L] = outputs.hidden_states[L][0].mean(dim=0).cpu()
    return result


def compute_pair_divergence(hs_clean: dict, hs_noisy: dict, layers: list):
    """
    Given hidden states for a (Q, Q') pair, compute KL and cosine distance
    at each candidate layer.
    Returns dict {layer: {"kl": float, "cosine_dist": float}}
    """
    result = {}
    for L in layers:
        result[L] = {
            "kl":          kl_divergence(hs_clean[L], hs_noisy[L]),
            "cosine_dist": cosine_distance(hs_clean[L], hs_noisy[L]),
        }
    return result


def load_pairs_from_jsonl(jsonl_path: str, n_pairs: int, seed: int = 42):
    """
    Load (original_question, adversarial_question) pairs from adversarial JSONL.
    Each JSONL record has one clean Q and up to 3 noisy Q' variants.
    Returns list of (Q, Q') tuples.
    """
    random.seed(seed)
    pairs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            q_clean = record.get("original_question")
            adverserials = record.get("modified_questions", {}).get("adverserials", [])

            if not q_clean or not adverserials:
                continue

            for q_noisy in adverserials:
                if q_noisy:
                    pairs.append((q_clean, q_noisy))

    if len(pairs) > n_pairs:
        pairs = random.sample(pairs, n_pairs)

    print(f"Loaded {len(pairs)} (Q, Q') pairs from {jsonl_path}")
    return pairs



def aggregate_divergences(all_pair_divergences: list, layers: list):
    """
    Aggregate per-pair divergences into per-layer statistics.
    Returns dict {layer: {"kl_mean", "kl_std", "cosine_mean", "cosine_std"}}
    """
    layer_kl   = defaultdict(list)
    layer_cos  = defaultdict(list)

    for pair_div in all_pair_divergences:
        for L in layers:
            layer_kl[L].append(pair_div[L]["kl"])
            layer_cos[L].append(pair_div[L]["cosine_dist"])

    stats = {}
    for L in layers:
        stats[L] = {
            "kl_mean":      float(np.mean(layer_kl[L])),
            "kl_std":       float(np.std(layer_kl[L])),
            "cosine_mean":  float(np.mean(layer_cos[L])),
            "cosine_std":   float(np.std(layer_cos[L])),
        }
    return stats


def layer_group_label(L, group_a, group_b, group_c):
    groups = []
    if L in group_a: groups.append("A(early-adj)")
    if L in group_b: groups.append("B(late-adj)")
    if L in group_c: groups.append("C(M1-stable)")
    return "/".join(groups) if groups else "?"



def plot_divergences(stats, layers, group_a, group_b, group_c, output_dir):
    kl_means   = [stats[L]["kl_mean"]     for L in layers]
    kl_stds    = [stats[L]["kl_std"]      for L in layers]
    cos_means  = [stats[L]["cosine_mean"] for L in layers]
    cos_stds   = [stats[L]["cosine_std"]  for L in layers]

    group_a_s = set(group_a)
    group_b_s = set(group_b)
    group_c_s = set(group_c)

    # Assign bar colours by group
    colors = []
    for L in layers:
        if L in group_a_s:
            colors.append("steelblue")
        elif L in group_b_s:
            colors.append("darkorange")
        elif L in group_c_s:
            colors.append("seagreen")
        else:
            colors.append("grey")

    x = np.arange(len(layers))
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    bars = ax.bar(x, kl_means, color=colors, alpha=0.8, yerr=kl_stds,
                  capsize=4, error_kw={"elinewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in layers])
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Symmetric KL divergence")
    ax.set_title("KL Divergence: Clean Q vs Noisy Q'  per Candidate Layer\n"
                 "Higher = layer is more sensitive to noise (better hook target)")
    ax.grid(True, alpha=0.3, axis="y")

    patch_a = mpatches.Patch(color="steelblue",   alpha=0.8, label="Group A — early adjacent-stable [6-9]")
    patch_b = mpatches.Patch(color="darkorange",  alpha=0.8, label="Group B — late adjacent-stable [23-25]")
    patch_c = mpatches.Patch(color="seagreen",    alpha=0.8, label="Group C — M1-stable [18-20]")
    ax.legend(handles=[patch_a, patch_b, patch_c])

    ax = axes[1]
    ax.bar(x, cos_means, color=colors, alpha=0.8, yerr=cos_stds,
           capsize=4, error_kw={"elinewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in layers])
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Cosine distance (1 - similarity)")
    ax.set_title("Cosine Distance: Clean Q vs Noisy Q'  per Candidate Layer\n"
                 "Higher = hidden states displaced further by noise")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(handles=[patch_a, patch_b, patch_c])

    plt.tight_layout()
    out_path = Path(output_dir) / "layer_noise_divergence.png"
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved: {out_path}")
    plt.close()


def save_results(stats, layers, group_a, group_b, group_c, n_pairs, output_dir):
    group_a_s = set(group_a)
    group_b_s = set(group_b)
    group_c_s = set(group_c)

    # Rank layers by KL divergence (descending — higher = more noise-sensitive)
    ranked_by_kl = sorted(layers, key=lambda L: stats[L]["kl_mean"], reverse=True)

    out = {
        "n_pairs_analysed": n_pairs,
        "candidate_layer_groups": {
            "group_A_early_adjacent_stable": group_a,
            "group_B_late_adjacent_stable":  group_b,
            "group_C_m1_stable":             group_c,
        },
        "per_layer_stats": {
            str(L): {
                "group": layer_group_label(L, group_a_s, group_b_s, group_c_s),
                **stats[L]
            }
            for L in layers
        },
        "ranked_by_kl_divergence": ranked_by_kl,
        "top_hook_candidates": ranked_by_kl[:5],
    }

    # Console summary
    print(f"\n{'='*60}")
    print(f"Pairs analysed: {n_pairs}")
    print(f"\nPer-layer divergence (Q vs Q'):")
    print(f"{'Layer':>6}  {'Group':<20}  {'KL mean':>10}  {'KL std':>8}  {'Cos mean':>10}  {'Cos std':>8}")
    print("-" * 72)
    for L in layers:
        s = stats[L]
        grp = layer_group_label(L, group_a_s, group_b_s, group_c_s)
        print(f"{L:>6}  {grp:<20}  {s['kl_mean']:>10.4f}  {s['kl_std']:>8.4f}  "
              f"{s['cosine_mean']:>10.6f}  {s['cosine_std']:>8.6f}")
    print(f"\nTop 5 hook candidates (by KL): {ranked_by_kl[:5]}")
    print(f"{'='*60}\n")

    out_path = Path(output_dir) / "noise_divergence_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved: {out_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to adversarial JSONL file (or glob pattern)")
    parser.add_argument("--output-dir", type=str, default="noise_divergence_results")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--candidate-layers", type=int, nargs="+",
                        default=DEFAULT_LAYERS,
                        help="Layer indices to probe (default: 6 7 8 9 18 19 20 23 24 25)")
    parser.add_argument("--group-a", type=int, nargs="+", default=GROUP_A)
    parser.add_argument("--group-b", type=int, nargs="+", default=GROUP_B)
    parser.add_argument("--group-c", type=int, nargs="+", default=GROUP_C)
    parser.add_argument("--n-pairs", type=int, default=200,
                        help="Number of (Q, Q') pairs to analyse (default: 200)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    layers = sorted(set(args.candidate_layers))

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # Load pairs — support multiple JSONL files via glob
    from glob import glob
    input_files = glob(args.input) if "*" in args.input else [args.input]
    all_pairs = []
    for f in input_files:
        all_pairs.extend(load_pairs_from_jsonl(f, n_pairs=args.n_pairs * 2,
                                               seed=args.seed))
    random.seed(args.seed)
    if len(all_pairs) > args.n_pairs:
        all_pairs = random.sample(all_pairs, args.n_pairs)
    print(f"Total pairs selected for analysis: {len(all_pairs)}")

    # Run inference and collect divergences
    all_pair_divergences = []
    failed = 0

    for q_clean, q_noisy in tqdm(all_pairs, desc="Computing divergences"):
        try:
            hs_clean = extract_hidden_states_at_layers(
                model, tokenizer, build_prompt(q_clean), device, layers)
            hs_noisy = extract_hidden_states_at_layers(
                model, tokenizer, build_prompt(q_noisy), device, layers)
            pair_div = compute_pair_divergence(hs_clean, hs_noisy, layers)
            all_pair_divergences.append(pair_div)
        except Exception as e:
            print(f"Failed on pair: {e}")
            failed += 1

    print(f"Processed {len(all_pair_divergences)} pairs ({failed} failed)")

    # Aggregate and report
    stats = aggregate_divergences(all_pair_divergences, layers)
    save_results(stats, layers, args.group_a, args.group_b, args.group_c,
                 len(all_pair_divergences), args.output_dir)
    plot_divergences(stats, layers, args.group_a, args.group_b, args.group_c,
                     args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()