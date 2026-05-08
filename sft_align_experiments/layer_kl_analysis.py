"""
LLM summary for file 
-----------------------------------------------------
Layer KL Divergence Analysis
=============================
Two independent stability metrics to identify concept maturity layers:

  Metric 1 — KL vs last layer:
      KL( h_L || h_last ) — how close is layer L to the final representation
      Low = converged TO the final answer representation

  Metric 2 — Adjacent layer KL (delta KL):
      KL( h_L || h_{L+1} ) — how much does the representation change between layers
      Low = stopped CHANGING, representation has stabilized locally

  Hook layer candidates = stable by BOTH metrics
      These are the "concept maturity" layers where the model has settled
      into its final representation AND stopped refining it.

Usage:
    python layer_kl_analysis.py \
        --input dataset/gsm8k_processed_train.json \
        --n-samples 500 \
        --output-dir layer_analysis_results
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = ""



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



@torch.no_grad()
def extract_hidden_states(model, tokenizer, prompt: str, device: str):
    """
    Forward pass returning mean-pooled hidden state per layer.
    Returns list of tensors (one per layer incl. embedding), each shape (hidden_dim,)
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=512).to(device)
    outputs = model(**inputs, output_hidden_states=True)
    # hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, hidden_dim)
    return [h[0].mean(dim=0).cpu() for h in outputs.hidden_states]



def compute_kl_vs_last(hidden_states_list):
    """
    For each sample and each layer L, compute KL( h_L || h_last ).
    Returns: np array (n_samples, n_layers)
    """
    n_samples = len(hidden_states_list)
    n_layers = len(hidden_states_list[0])
    matrix = np.zeros((n_samples, n_layers))
    for i, hs in enumerate(hidden_states_list):
        last = hs[-1]
        for L in range(n_layers):
            matrix[i, L] = kl_divergence(hs[L], last)
    return matrix



def compute_adjacent_kl(hidden_states_list):
    """
    For each sample and each layer L, compute KL( h_L || h_{L+1} ).
    Returns: np array (n_samples, n_layers-1)
             index L corresponds to the transition L -> L+1
    """
    n_samples = len(hidden_states_list)
    n_layers = len(hidden_states_list[0])
    matrix = np.zeros((n_samples, n_layers - 1))
    for i, hs in enumerate(hidden_states_list):
        for L in range(n_layers - 1):
            matrix[i, L] = kl_divergence(hs[L], hs[L + 1])
    return matrix



def find_stable_layers(kl_vs_last_matrix, adjacent_kl_matrix, percentile=25):
    """
    Identify layers stable by each metric independently, then find the intersection.

    M1 stable: mean KL vs last < percentile threshold of M1 values
    M2 stable: mean adjacent KL < percentile threshold of M2 values
               (layer L is M2-stable if the L->L+1 transition is small)

    Returns dict with all metrics and stable layer sets.
    """
    mean_kl_vs_last = kl_vs_last_matrix.mean(axis=0)    # (n_layers,)
    mean_adjacent_kl = adjacent_kl_matrix.mean(axis=0)  # (n_layers-1,)

    n_layers = len(mean_kl_vs_last)

    # M1: exclude embedding (0) and last layer (n_layers-1)
    m1_vals = mean_kl_vs_last[1:-1]
    m1_threshold = np.percentile(m1_vals, percentile)
    stable_m1 = [L for L in range(1, n_layers - 1) if mean_kl_vs_last[L] <= m1_threshold]

    # M2: layer L is stable if transition L->L+1 is small
    # skip embedding->first layer transition (index 0)
    m2_vals = mean_adjacent_kl[1:]
    m2_threshold = np.percentile(m2_vals, percentile)
    stable_m2 = [L for L in range(1, n_layers - 1) if mean_adjacent_kl[L] <= m2_threshold]

    stable_both = sorted(set(stable_m1) & set(stable_m2))

    return {
        "mean_kl_vs_last": mean_kl_vs_last,
        "mean_adjacent_kl": mean_adjacent_kl,
        "std_kl_vs_last": kl_vs_last_matrix.std(axis=0),
        "std_adjacent_kl": adjacent_kl_matrix.std(axis=0),
        "stable_m1": stable_m1,
        "stable_m2": stable_m2,
        "stable_both": stable_both,
        "m1_threshold": float(m1_threshold),
        "m2_threshold": float(m2_threshold),
        "n_layers": n_layers,
    }


def plot_results(results, output_dir):
    mean_kl_vs_last = results["mean_kl_vs_last"]
    mean_adj_kl = results["mean_adjacent_kl"]
    std_kl_vs_last = results["std_kl_vs_last"]
    std_adj_kl = results["std_adjacent_kl"]

    n_layers = results["n_layers"]
    layers = list(range(n_layers))
    adj_layers = list(range(n_layers - 1))

    stable_m1 = set(results["stable_m1"])
    stable_m2 = set(results["stable_m2"])
    stable_both = set(results["stable_both"])

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))


    ax = axes[0]
    ax.plot(layers, mean_kl_vs_last, color="steelblue", linewidth=2,
            label="Mean KL vs last layer (M1)")
    ax.fill_between(layers,
                    mean_kl_vs_last - std_kl_vs_last,
                    mean_kl_vs_last + std_kl_vs_last,
                    alpha=0.2, color="steelblue")
    ax.axhline(y=results["m1_threshold"], color="steelblue", linestyle="--",
               alpha=0.6, label=f"M1 threshold ({results['m1_threshold']:.4f})")
    for L in stable_m1:
        ax.axvspan(L - 0.5, L + 0.5, alpha=0.15, color="steelblue")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Symmetric KL divergence")
    ax.set_title("Metric 1: KL vs Last Layer  —  "
                 "Low = layer has converged TO the final answer representation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(adj_layers, mean_adj_kl, color="darkorange", linewidth=2,
            label="Mean adjacent KL (M2)  —  transition L→L+1")
    ax.fill_between(adj_layers,
                    mean_adj_kl - std_adj_kl,
                    mean_adj_kl + std_adj_kl,
                    alpha=0.2, color="darkorange")
    ax.axhline(y=results["m2_threshold"], color="darkorange", linestyle="--",
               alpha=0.6, label=f"M2 threshold ({results['m2_threshold']:.4f})")
    for L in stable_m2:
        ax.axvspan(L - 0.5, L + 0.5, alpha=0.15, color="darkorange")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Symmetric KL divergence")
    ax.set_title("Metric 2: Adjacent Layer KL  —  "
                 "Low = representation stopped CHANGING between consecutive layers")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]

    # Normalize both to [0,1] for overlay comparison
    m1_norm = (mean_kl_vs_last - mean_kl_vs_last.min()) / (mean_kl_vs_last.ptp() + 1e-8)
    m2_norm = (mean_adj_kl - mean_adj_kl.min()) / (mean_adj_kl.ptp() + 1e-8)

    ax.plot(layers, m1_norm, color="steelblue", linewidth=1.5,
            label="M1 normalized (KL vs last)", alpha=0.8)
    ax.plot(adj_layers, m2_norm, color="darkorange", linewidth=1.5,
            label="M2 normalized (adjacent KL)", alpha=0.8)

    # Shade stability regions
    for L in range(1, n_layers - 1):
        in_both = L in stable_both
        in_m1_only = (L in stable_m1) and (L not in stable_both)
        in_m2_only = (L in stable_m2) and (L not in stable_both)
        if in_both:
            ax.axvspan(L - 0.5, L + 0.5, alpha=0.4, color="green")
        elif in_m1_only:
            ax.axvspan(L - 0.5, L + 0.5, alpha=0.15, color="steelblue")
        elif in_m2_only:
            ax.axvspan(L - 0.5, L + 0.5, alpha=0.15, color="darkorange")

    patch_both = mpatches.Patch(color="green", alpha=0.5,
                                label=f"Stable by BOTH — hook candidates ({len(stable_both)} layers)")
    patch_m1 = mpatches.Patch(color="steelblue", alpha=0.3,
                               label=f"M1 only ({len(stable_m1 - stable_both)} layers)")
    patch_m2 = mpatches.Patch(color="darkorange", alpha=0.3,
                               label=f"M2 only ({len(stable_m2 - stable_both)} layers)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [patch_both, patch_m1, patch_m2])
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Normalized KL (0 = most stable)")
    ax.set_title("Overlap: Layers Stable by Both Metrics  —  "
                 "Green = concept maturity zone (recommended hook layers)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(output_dir) / "layer_kl_analysis.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved: {out_path}")
    plt.close()



def save_results(results, output_dir):
    stable_both = set(results["stable_both"])
    stable_m1 = set(results["stable_m1"])
    stable_m2 = set(results["stable_m2"])

    out = {
        "n_layers_total": results["n_layers"],
        "transformer_blocks": results["n_layers"] - 1,
        "thresholds": {
            "m1_kl_vs_last": results["m1_threshold"],
            "m2_adjacent_kl": results["m2_threshold"],
        },
        "stable_layers": {
            "metric1_kl_vs_last": results["stable_m1"],
            "metric2_adjacent_kl": results["stable_m2"],
            "both_metrics": results["stable_both"],
            "m1_only": sorted(stable_m1 - stable_both),
            "m2_only": sorted(stable_m2 - stable_both),
        },
        "recommended_hook_layers": results["stable_both"],
        "mean_kl_vs_last_per_layer": results["mean_kl_vs_last"].tolist(),
        "mean_adjacent_kl_per_layer": results["mean_adjacent_kl"].tolist(),
    }

    print(f"\n{'='*60}")
    print(f"Total layers (incl. embedding) : {results['n_layers']}")
    print(f"Transformer blocks             : {results['n_layers'] - 1}")
    print(f"\nM1 stable ({len(stable_m1)} layers) : {results['stable_m1']}")
    print(f"M2 stable ({len(stable_m2)} layers) : {results['stable_m2']}")
    print(f"\nStable by BOTH ({len(stable_both)} layers) — recommended hook layers:")
    print(f"  {results['stable_both']}")
    print(f"{'='*60}\n")

    out_path = Path(output_dir) / "layer_analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved: {out_path}")


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="dataset/gsm8k_processed_train.json")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="layer_analysis_results")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--percentile", type=float, default=25,
                        help="Bottom percentile threshold for stability (default: 25)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)
    if len(records) > args.n_samples:
        records = random.sample(records, args.n_samples)
    print(f"Loaded {len(records)} samples")

    hidden_states_list = []
    failed = 0
    for record in tqdm(records, desc="Extracting hidden states"):
        question = record.get("question")
        if not question:
            failed += 1
            continue
        try:
            hs = extract_hidden_states(model, tokenizer, build_prompt(question), device)
            hidden_states_list.append(hs)
        except Exception as e:
            print(f"Failed: {e}")
            failed += 1

    print(f"Processed {len(hidden_states_list)} samples ({failed} failed)")
    if len(hidden_states_list) < 10:
        raise RuntimeError("Too few samples processed — check model and dataset paths.")

    print("Computing Metric 1: KL vs last layer...")
    kl_vs_last_matrix = compute_kl_vs_last(hidden_states_list)

    print("Computing Metric 2: Adjacent layer KL...")
    adjacent_kl_matrix = compute_adjacent_kl(hidden_states_list)

    results = find_stable_layers(kl_vs_last_matrix, adjacent_kl_matrix,
                                 percentile=args.percentile)

    save_results(results, args.output_dir)
    plot_results(results, args.output_dir)

    print("Done. Use 'recommended_hook_layers' from layer_analysis_results.json for training.")


if __name__ == "__main__":
    main()