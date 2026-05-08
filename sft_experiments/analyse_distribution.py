"""
Distribution Analysis: Clean vs Noisy Questions
=================================================
Proves that noisy question variants represent a genuinely different distribution
from the original clean questions, providing empirical backing for the gain+loss
tradeoff observed in noise-robust training results.

Analyses performed:
  1. Lexical distance  — token-level edit distance and Jaccard similarity
  2. Length statistics — token count distribution shift
  3. Vocabulary shift  — new tokens introduced by noise injection
  4. TF-IDF cosine similarity — semantic surface distance
  5. Perplexity shift  — how much harder the base LM finds the noisy questions
                         (requires --model flag; skip with --no-perplexity)
  6. Embedding distance — mean cosine distance in frozen model hidden space
                          (requires --model flag; skip with --no-embeddings)

Output:
  - distribution_analysis.json  — all numeric results
  - distribution_plots.png      — 6-panel summary figure
  - distribution_report.txt     — human-readable summary

Usage:
    # Fast (lexical only, no model needed):
    python analyze_distribution.py \
        --jsonl pit-train.jsonl \
        --output-dir dist_analysis \
        --no-perplexity --no-embeddings

    # Full (includes model-based analyses):
    python analyze_distribution.py \
        --jsonl pit-train.jsonl \
        --base-model jahyungu/Qwen2.5-1.5B-Instruct_gsm8k \
        --output-dir dist_analysis \
        --n-samples 500
"""

import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm


try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not found. TF-IDF analysis will be skipped.")

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: torch/transformers not found. Model-based analyses will be skipped.")


# Data loading

def load_pairs(jsonl_path: str, n_samples: Optional[int], seed: int = 42,
               max_variants: int = 1) -> Tuple[List[str], List[str]]:
    """
    Load (clean_q, noisy_q) pairs from JSONL.
    Returns two parallel lists: clean questions and noisy questions.
    """
    random.seed(seed)
    clean_qs, noisy_qs = [], []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            clean_q      = record.get("original_question", "")
            adverserials = record.get("modified_questions", {}).get("adverserials", [])
            adverserials = [v for v in adverserials if v]

            if not clean_q or not adverserials:
                continue

            for noisy_q in adverserials[:max_variants]:
                clean_qs.append(clean_q)
                noisy_qs.append(noisy_q)

    if n_samples and len(clean_qs) > n_samples:
        indices  = random.sample(range(len(clean_qs)), n_samples)
        clean_qs = [clean_qs[i] for i in indices]
        noisy_qs = [noisy_qs[i] for i in indices]

    print(f"Loaded {len(clean_qs)} (clean, noisy) pairs from {jsonl_path}")
    return clean_qs, noisy_qs


# Lexical utilities

def simple_tokenize(text: str) -> List[str]:
    """Whitespace + punctuation split — no external dep needed."""
    return re.findall(r"\w+", text.lower())


def edit_distance_ratio(a: str, b: str) -> float:
    """
    Normalised Levenshtein distance at the word level (0=identical, 1=fully different).
    Uses DP — O(|a|*|b|) but fast enough for sentence-length inputs.
    """
    wa, wb = simple_tokenize(a), simple_tokenize(b)
    if not wa and not wb:
        return 0.0
    la, lb = len(wa), len(wb)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, lb + 1):
            temp = dp[j]
            if wa[i-1] == wb[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[lb] / max(la, lb)


def jaccard_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity (0=no overlap, 1=identical sets)."""
    sa, sb = set(simple_tokenize(a)), set(simple_tokenize(b))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def new_token_rate(clean: str, noisy: str) -> float:
    """Fraction of noisy tokens not present in clean question."""
    tc = set(simple_tokenize(clean))
    tn = simple_tokenize(noisy)
    if not tn:
        return 0.0
    return sum(1 for t in tn if t not in tc) / len(tn)


# Analysis functions 

def lexical_analysis(clean_qs: List[str], noisy_qs: List[str]) -> dict:
    """Compute edit distance, Jaccard similarity, length stats, vocab shift."""
    print("Running lexical analysis...")

    edit_dists   = []
    jaccards     = []
    new_tok_rates = []
    clean_lens   = []
    noisy_lens   = []

    for cq, nq in tqdm(zip(clean_qs, noisy_qs), total=len(clean_qs), desc="  Lexical"):
        edit_dists.append(edit_distance_ratio(cq, nq))
        jaccards.append(jaccard_similarity(cq, nq))
        new_tok_rates.append(new_token_rate(cq, nq))
        clean_lens.append(len(simple_tokenize(cq)))
        noisy_lens.append(len(simple_tokenize(nq)))

    results = {
        "edit_distance": {
            "mean":   float(np.mean(edit_dists)),
            "median": float(np.median(edit_dists)),
            "std":    float(np.std(edit_dists)),
            "pct_above_0.1": float(np.mean(np.array(edit_dists) > 0.1)),
            "pct_above_0.3": float(np.mean(np.array(edit_dists) > 0.3)),
            "values": edit_dists,
        },
        "jaccard_similarity": {
            "mean":   float(np.mean(jaccards)),
            "median": float(np.median(jaccards)),
            "std":    float(np.std(jaccards)),
            "pct_below_0.7": float(np.mean(np.array(jaccards) < 0.7)),
            "values": jaccards,
        },
        "new_token_rate": {
            "mean":   float(np.mean(new_tok_rates)),
            "median": float(np.median(new_tok_rates)),
            "std":    float(np.std(new_tok_rates)),
            "values": new_tok_rates,
        },
        "length": {
            "clean_mean":  float(np.mean(clean_lens)),
            "noisy_mean":  float(np.mean(noisy_lens)),
            "clean_std":   float(np.std(clean_lens)),
            "noisy_std":   float(np.std(noisy_lens)),
            "mean_increase": float(np.mean(noisy_lens) - np.mean(clean_lens)),
            "clean_values": clean_lens,
            "noisy_values": noisy_lens,
        },
    }

    print(f"  Edit distance  : {results['edit_distance']['mean']:.3f} ± "
          f"{results['edit_distance']['std']:.3f}  "
          f"({results['edit_distance']['pct_above_0.1']:.1%} > 0.1)")
    print(f"  Jaccard sim    : {results['jaccard_similarity']['mean']:.3f} ± "
          f"{results['jaccard_similarity']['std']:.3f}")
    print(f"  New token rate : {results['new_token_rate']['mean']:.3f} ± "
          f"{results['new_token_rate']['std']:.3f}")
    print(f"  Length clean   : {results['length']['clean_mean']:.1f} tokens")
    print(f"  Length noisy   : {results['length']['noisy_mean']:.1f} tokens "
          f"(+{results['length']['mean_increase']:.1f})")

    return results


def tfidf_analysis(clean_qs: List[str], noisy_qs: List[str]) -> dict:
    """TF-IDF cosine similarity between paired clean/noisy questions."""
    if not HAS_SKLEARN:
        return {"skipped": True, "reason": "scikit-learn not installed"}

    print("Running TF-IDF analysis...")
    all_texts = clean_qs + noisy_qs
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    n = len(clean_qs)
    clean_vecs = tfidf_matrix[:n]
    noisy_vecs = tfidf_matrix[n:]

    # Per-pair cosine similarity
    sims = []
    batch = 64
    for i in range(0, n, batch):
        c_batch = clean_vecs[i:i+batch]
        n_batch = noisy_vecs[i:i+batch]
        sim_batch = sklearn_cosine(c_batch, n_batch).diagonal()
        sims.extend(sim_batch.tolist())

    # Also compare clean-clean (same-distribution baseline) using random pairs
    rng = np.random.default_rng(42)
    idx_a = rng.integers(0, n, size=min(n, 500))
    idx_b = rng.integers(0, n, size=min(n, 500))
    baseline_sims = sklearn_cosine(
        clean_vecs[idx_a], clean_vecs[idx_b]
    ).diagonal().tolist()

    results = {
        "clean_noisy_cosine": {
            "mean":   float(np.mean(sims)),
            "median": float(np.median(sims)),
            "std":    float(np.std(sims)),
            "values": sims,
        },
        "clean_clean_baseline_cosine": {
            "mean":   float(np.mean(baseline_sims)),
            "median": float(np.median(baseline_sims)),
            "std":    float(np.std(baseline_sims)),
            "values": baseline_sims,
        },
        "distribution_gap": float(np.mean(baseline_sims) - np.mean(sims)),
    }

    print(f"  Clean↔Noisy cosine : {results['clean_noisy_cosine']['mean']:.3f} ± "
          f"{results['clean_noisy_cosine']['std']:.3f}")
    print(f"  Clean↔Clean baseline: {results['clean_clean_baseline_cosine']['mean']:.3f} ± "
          f"{results['clean_clean_baseline_cosine']['std']:.3f}")
    print(f"  Distribution gap   : {results['distribution_gap']:.3f} "
          f"({'significant' if results['distribution_gap'] > 0.05 else 'small'})")

    return results


def perplexity_analysis(clean_qs: List[str], noisy_qs: List[str],
                        model_name: str, device: str) -> dict:
    """
    Compute per-token perplexity of clean vs noisy questions under the base LM.
    Higher perplexity on noisy = model finds them OOD.
    """
    if not HAS_TORCH:
        return {"skipped": True, "reason": "torch not installed"}

    print("Running perplexity analysis (loading model)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()

    def compute_perplexity(texts: List[str], label: str) -> List[float]:
        ppls = []
        with torch.no_grad():
            for text in tqdm(texts, desc=f"  PPL ({label})"):
                enc = tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=256).to(device)
                if enc["input_ids"].shape[1] < 2:
                    continue
                outputs = model(**enc, labels=enc["input_ids"])
                ppl = torch.exp(outputs.loss).item()
                if np.isfinite(ppl) and ppl < 1e6:
                    ppls.append(ppl)
        return ppls

    clean_ppls = compute_perplexity(clean_qs, "clean")
    noisy_ppls = compute_perplexity(noisy_qs, "noisy")

    del model
    if HAS_TORCH:
        torch.cuda.empty_cache()

    results = {
        "clean_perplexity": {
            "mean":   float(np.mean(clean_ppls)),
            "median": float(np.median(clean_ppls)),
            "std":    float(np.std(clean_ppls)),
            "values": clean_ppls,
        },
        "noisy_perplexity": {
            "mean":   float(np.mean(noisy_ppls)),
            "median": float(np.median(noisy_ppls)),
            "std":    float(np.std(noisy_ppls)),
            "values": noisy_ppls,
        },
        "ppl_ratio": float(np.mean(noisy_ppls) / np.mean(clean_ppls))
                     if np.mean(clean_ppls) > 0 else None,
    }

    print(f"  Clean PPL  : {results['clean_perplexity']['mean']:.2f} ± "
          f"{results['clean_perplexity']['std']:.2f}")
    print(f"  Noisy PPL  : {results['noisy_perplexity']['mean']:.2f} ± "
          f"{results['noisy_perplexity']['std']:.2f}")
    if results["ppl_ratio"]:
        print(f"  PPL ratio  : {results['ppl_ratio']:.2f}x "
              f"({'noisy is OOD' if results['ppl_ratio'] > 1.2 else 'similar distribution'})")

    return results


def embedding_analysis(clean_qs: List[str], noisy_qs: List[str],
                       model_name: str, device: str) -> dict:
    """
    Extract last-token hidden states from the frozen base model for clean and
    noisy questions, then measure cosine distance between paired representations.
    Also computes clean-clean baseline distance for comparison.
    """
    if not HAS_TORCH:
        return {"skipped": True, "reason": "torch not installed"}

    print("Running embedding analysis (loading model)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()

    def get_embeddings(texts: List[str], label: str) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for text in tqdm(texts, desc=f"  Embeddings ({label})"):
                enc = tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=256).to(device)
                outputs = model(**enc, output_hidden_states=True)
                # Last token of layer 18 (mid-network, most semantic)
                last_idx = enc["attention_mask"].sum() - 1
                h = outputs.hidden_states[18][0, last_idx, :].float().cpu().numpy()
                embeddings.append(h)
        return np.array(embeddings)

    clean_embs = get_embeddings(clean_qs, "clean")
    noisy_embs = get_embeddings(noisy_qs, "noisy")

    del model
    if HAS_TORCH:
        torch.cuda.empty_cache()

    # Paired clean↔noisy cosine similarity
    def batch_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return (a_norm * b_norm).sum(axis=1)

    pair_sims = batch_cosine(clean_embs, noisy_embs).tolist()

    # Clean-clean baseline: random pairs from clean set
    rng = np.random.default_rng(42)
    n   = len(clean_embs)
    idx_a = rng.integers(0, n, size=min(n, 500))
    idx_b = rng.integers(0, n, size=min(n, 500))
    baseline_sims = batch_cosine(clean_embs[idx_a], clean_embs[idx_b]).tolist()

    results = {
        "clean_noisy_cosine": {
            "mean":   float(np.mean(pair_sims)),
            "median": float(np.median(pair_sims)),
            "std":    float(np.std(pair_sims)),
            "values": pair_sims,
        },
        "clean_clean_baseline": {
            "mean":   float(np.mean(baseline_sims)),
            "median": float(np.median(baseline_sims)),
            "std":    float(np.std(baseline_sims)),
            "values": baseline_sims,
        },
        "embedding_gap": float(np.mean(baseline_sims) - np.mean(pair_sims)),
        "layer": 18,
    }

    print(f"  Clean↔Noisy cosine  : {results['clean_noisy_cosine']['mean']:.4f} ± "
          f"{results['clean_noisy_cosine']['std']:.4f}")
    print(f"  Clean↔Clean baseline: {results['clean_clean_baseline']['mean']:.4f} ± "
          f"{results['clean_clean_baseline']['std']:.4f}")
    print(f"  Embedding gap       : {results['embedding_gap']:.4f} "
          f"({'significant' if results['embedding_gap'] > 0.02 else 'small'})")

    return results



def plot_results(lex: dict, tfidf: dict, ppl: dict, emb: dict, output_dir: str):
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(lex["edit_distance"]["values"], bins=40, color="#e05c5c", alpha=0.8,
             edgecolor="white", linewidth=0.5)
    ax1.axvline(lex["edit_distance"]["mean"], color="#333", linestyle="--",
                linewidth=1.5, label=f"Mean = {lex['edit_distance']['mean']:.3f}")
    ax1.set_title("Word-Level Edit Distance\n(Clean vs Noisy)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Normalised edit distance (0=identical, 1=fully different)")
    ax1.set_ylabel("Count")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(lex["jaccard_similarity"]["values"], bins=40, color="#5c8ee0", alpha=0.8,
             edgecolor="white", linewidth=0.5)
    ax2.axvline(lex["jaccard_similarity"]["mean"], color="#333", linestyle="--",
                linewidth=1.5, label=f"Mean = {lex['jaccard_similarity']['mean']:.3f}")
    ax2.set_title("Jaccard Similarity\n(Word Set Overlap)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Jaccard similarity (1=identical sets)")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(lex["length"]["clean_values"], bins=30, color="#5cb85c", alpha=0.6,
             label=f"Clean (μ={lex['length']['clean_mean']:.1f})",
             edgecolor="white", linewidth=0.5)
    ax3.hist(lex["length"]["noisy_values"], bins=30, color="#e05c5c", alpha=0.6,
             label=f"Noisy (μ={lex['length']['noisy_mean']:.1f})",
             edgecolor="white", linewidth=0.5)
    ax3.set_title("Question Length Distribution\n(Word Tokens)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Word count")
    ax3.set_ylabel("Count")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)


    ax4 = fig.add_subplot(gs[1, 0])
    if not tfidf.get("skipped"):
        ax4.hist(tfidf["clean_noisy_cosine"]["values"], bins=40,
                 color="#e05c5c", alpha=0.7,
                 label=f"Clean↔Noisy (μ={tfidf['clean_noisy_cosine']['mean']:.3f})",
                 edgecolor="white", linewidth=0.5)
        ax4.hist(tfidf["clean_clean_baseline_cosine"]["values"], bins=40,
                 color="#5c8ee0", alpha=0.7,
                 label=f"Clean↔Clean (μ={tfidf['clean_clean_baseline_cosine']['mean']:.3f})",
                 edgecolor="white", linewidth=0.5)
        ax4.set_title("TF-IDF Cosine Similarity\nPaired vs Same-Distribution Baseline",
                      fontsize=11, fontweight="bold")
        ax4.set_xlabel("Cosine similarity")
        ax4.set_ylabel("Count")
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "TF-IDF skipped\n(scikit-learn not installed)",
                 ha="center", va="center", transform=ax4.transAxes, fontsize=10)
        ax4.set_title("TF-IDF Cosine Similarity", fontsize=11, fontweight="bold")

    ax5 = fig.add_subplot(gs[1, 1])
    if not ppl.get("skipped"):
        data   = [ppl["clean_perplexity"]["values"], ppl["noisy_perplexity"]["values"]]
        bp = ax5.boxplot(data, labels=["Clean Q", "Noisy Q'"],
                         patch_artist=True, notch=False,
                         medianprops=dict(color="white", linewidth=2))
        bp["boxes"][0].set_facecolor("#5cb85c")
        bp["boxes"][1].set_facecolor("#e05c5c")
        ax5.set_title(f"Base LM Perplexity\n(ratio={ppl['ppl_ratio']:.2f}x)",
                      fontsize=11, fontweight="bold")
        ax5.set_ylabel("Perplexity (lower = more in-distribution)")
        ax5.grid(True, alpha=0.3, axis="y")
    else:
        ax5.text(0.5, 0.5, "Perplexity skipped\n(use --model to enable)",
                 ha="center", va="center", transform=ax5.transAxes, fontsize=10)
        ax5.set_title("Base LM Perplexity", fontsize=11, fontweight="bold")

    ax6 = fig.add_subplot(gs[1, 2])
    if not emb.get("skipped"):
        ax6.hist(emb["clean_noisy_cosine"]["values"], bins=40,
                 color="#e05c5c", alpha=0.7,
                 label=f"Clean↔Noisy (μ={emb['clean_noisy_cosine']['mean']:.4f})",
                 edgecolor="white", linewidth=0.5)
        ax6.hist(emb["clean_clean_baseline"]["values"], bins=40,
                 color="#5c8ee0", alpha=0.7,
                 label=f"Clean↔Clean (μ={emb['clean_clean_baseline']['mean']:.4f})",
                 edgecolor="white", linewidth=0.5)
        ax6.set_title(f"Frozen Model Embedding Cosine\n(layer 18, last token)",
                      fontsize=11, fontweight="bold")
        ax6.set_xlabel("Cosine similarity")
        ax6.set_ylabel("Count")
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, "Embeddings skipped\n(use --model to enable)",
                 ha="center", va="center", transform=ax6.transAxes, fontsize=10)
        ax6.set_title("Frozen Model Embeddings", fontsize=11, fontweight="bold")

    fig.suptitle("Clean vs Noisy Question Distribution Analysis",
                 fontsize=14, fontweight="bold", y=1.01)

    out_path = Path(output_dir) / "distribution_plots.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {out_path}")
    plt.close()


#

def write_report(lex: dict, tfidf: dict, ppl: dict, emb: dict,
                 n_pairs: int, output_dir: str):
    lines = [
        "=" * 60,
        "  Distribution Analysis: Clean vs Noisy Questions",
        "=" * 60,
        f"  Pairs analysed: {n_pairs}",
        "",
        "── 1. Lexical Distance ──────────────────────────────────────",
        f"  Word-level edit distance",
        f"    Mean   : {lex['edit_distance']['mean']:.3f}  (0=identical, 1=fully different)",
        f"    Median : {lex['edit_distance']['median']:.3f}",
        f"    >0.1   : {lex['edit_distance']['pct_above_0.1']:.1%} of pairs",
        f"    >0.3   : {lex['edit_distance']['pct_above_0.3']:.1%} of pairs",
        f"  Jaccard word-set similarity",
        f"    Mean   : {lex['jaccard_similarity']['mean']:.3f}  (1=identical sets)",
        f"    <0.7   : {lex['jaccard_similarity']['pct_below_0.7']:.1%} of pairs",
        f"  New tokens introduced by noise",
        f"    Mean fraction of noisy tokens not in clean: {lex['new_token_rate']['mean']:.3f}",
        f"  Question length",
        f"    Clean  : {lex['length']['clean_mean']:.1f} ± {lex['length']['clean_std']:.1f} words",
        f"    Noisy  : {lex['length']['noisy_mean']:.1f} ± {lex['length']['noisy_std']:.1f} words",
        f"    Delta  : +{lex['length']['mean_increase']:.1f} words on average",
        "",
    ]

    if not tfidf.get("skipped"):
        lines += [
            "── 2. TF-IDF Surface Similarity ────────────────────────────",
            f"  Clean↔Noisy cosine  : {tfidf['clean_noisy_cosine']['mean']:.3f} "
            f"± {tfidf['clean_noisy_cosine']['std']:.3f}",
            f"  Clean↔Clean baseline: {tfidf['clean_clean_baseline_cosine']['mean']:.3f} "
            f"± {tfidf['clean_clean_baseline_cosine']['std']:.3f}",
            f"  Distribution gap    : {tfidf['distribution_gap']:.3f}",
            f"  Interpretation      : {'Noise introduces meaningful surface-level shift' if tfidf['distribution_gap'] > 0.05 else 'Surface-level shift is minor'}",
            "",
        ]

    if not ppl.get("skipped"):
        lines += [
            "── 3. Base LM Perplexity ────────────────────────────────────",
            f"  Clean PPL  : {ppl['clean_perplexity']['mean']:.2f} ± {ppl['clean_perplexity']['std']:.2f}",
            f"  Noisy PPL  : {ppl['noisy_perplexity']['mean']:.2f} ± {ppl['noisy_perplexity']['std']:.2f}",
            f"  PPL ratio  : {ppl['ppl_ratio']:.2f}x",
            f"  Interpretation: {'Noisy questions are OOD for the base LM (ratio > 1.2)' if ppl['ppl_ratio'] > 1.2 else 'Similar perplexity — noise is subtle'}",
            "",
        ]

    if not emb.get("skipped"):
        lines += [
            "── 4. Frozen Model Embedding Distance (layer 18) ────────────",
            f"  Clean↔Noisy cosine  : {emb['clean_noisy_cosine']['mean']:.4f} "
            f"± {emb['clean_noisy_cosine']['std']:.4f}",
            f"  Clean↔Clean baseline: {emb['clean_clean_baseline']['mean']:.4f} "
            f"± {emb['clean_clean_baseline']['std']:.4f}",
            f"  Embedding gap       : {emb['embedding_gap']:.4f}",
            f"  Interpretation      : {'Noise shifts hidden representations significantly' if emb['embedding_gap'] > 0.02 else 'Hidden representations are largely similar'}",
            "",
        ]

    lines += [
        "── Conclusion ───────────────────────────────────────────────",
        "  The above metrics jointly show whether the noisy variants",
        "  represent a genuinely different input distribution.",
        "  A high edit distance + low Jaccard + high PPL ratio +",
        "  large embedding gap together explain why the trained model",
        "  gains on noisy accuracy but may regress on clean accuracy:",
        "  the noise perturbations push inputs far enough off the",
        "  clean distribution that learning to handle them requires",
        "  the model to change in ways that affect clean performance.",
        "=" * 60,
    ]

    report = "\n".join(lines)
    print("\n" + report)

    out_path = Path(output_dir) / "distribution_report.txt"
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nReport saved: {out_path}")


# 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl",          type=str, required=True,
                        help="Path to JSONL file (train or test)")
    parser.add_argument("--output-dir",     type=str, default="dist_analysis")
    parser.add_argument("--n-samples",      type=int, default=500)
    parser.add_argument("--max-variants",   type=int, default=1)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--base-model",     type=str,
                        default="jahyungu/Qwen2.5-1.5B-Instruct_gsm8k",
                        help="Model for perplexity + embedding analyses")
    parser.add_argument("--no-perplexity",  action="store_true",
                        help="Skip perplexity analysis (no model needed)")
    parser.add_argument("--no-embeddings",  action="store_true",
                        help="Skip embedding analysis (no model needed)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    clean_qs, noisy_qs = load_pairs(
        args.jsonl, n_samples=args.n_samples,
        seed=args.seed, max_variants=args.max_variants,
    )

    # ── Run analyses ──────────────────────────────────────────────────────────
    lex   = lexical_analysis(clean_qs, noisy_qs)
    tfidf = tfidf_analysis(clean_qs, noisy_qs)

    ppl = {"skipped": True, "reason": "--no-perplexity flag set"}
    if not args.no_perplexity and HAS_TORCH:
        ppl = perplexity_analysis(clean_qs, noisy_qs, args.base_model, device)

    emb = {"skipped": True, "reason": "--no-embeddings flag set"}
    if not args.no_embeddings and HAS_TORCH:
        emb = embedding_analysis(clean_qs, noisy_qs, args.base_model, device)

    # ── Save raw numbers ──────────────────────────────────────────────────────
    def strip_values(d):
        """Remove raw value lists before JSON serialisation (keep summary stats)."""
        if not isinstance(d, dict):
            return d
        return {k: strip_values(v) if k != "values" else f"[{len(v)} values]"
                for k, v in d.items()}

    results = {
        "n_pairs":  len(clean_qs),
        "lexical":  strip_values(lex),
        "tfidf":    strip_values(tfidf),
        "perplexity": strip_values(ppl),
        "embeddings": strip_values(emb),
    }
    json_path = Path(args.output_dir) / "distribution_analysis.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAnalysis JSON saved: {json_path}")

    # ── Plot & report ─────────────────────────────────────────────────────────
    plot_results(lex, tfidf, ppl, emb, args.output_dir)
    write_report(lex, tfidf, ppl, emb, len(clean_qs), args.output_dir)

    print(f"\nDone. All outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()