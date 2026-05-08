"""Convert noise_cot_{train,test}.jsonl into a structured-target format.

Each row's `raw` field is rewritten to:

    <relevance>
    - "<sentence>": RELEVANT
    - "<distractor>": DISTRACTOR
    ...
    </relevance>
    <reasoning>
    <math reasoning>
    </reasoning>
    ####
    <answer>

The training pipeline uses `raw` as the completion target and splits on `####`
to extract the final answer, so the suffix is preserved.

Distractors come from the existing `&&&& ... &&&&` block inside `reasoning`.
Relevant items are sentences of `original_question`.
"""
import argparse
import json
import re
from pathlib import Path

NOISE_BLOCK_RE = re.compile(r"@%\^\^\s*NOISE\s*@%\^\^\s*&&&&(.*?)&&&&\s*", re.DOTALL)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def sentence_split(text: str) -> list[str]:
    text = text.strip().strip('"').strip()
    if text.startswith("QUESTION:"):
        text = text[len("QUESTION:"):].strip().strip('"').strip()
    parts = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]
    return parts


def parse_distractors(reasoning: str) -> tuple[list[str], str]:
    """Return (distractor_list, cleaned_reasoning)."""
    m = NOISE_BLOCK_RE.search(reasoning)
    if not m:
        return [], reasoning.strip()
    distractor_block = m.group(1).strip()
    cleaned = reasoning[m.end():].strip()
    cleaned = re.sub(r"^<reasoning steps>\s*", "", cleaned)
    items = [d.strip(" ;,.\n") for d in re.split(r";|\n", distractor_block) if d.strip(" ;,.\n")]
    return items, cleaned


def build_target(sample: dict) -> str:
    typ = sample.get("type", "clean")
    answer = str(sample["answer"]).strip()
    original_q = sample.get("original_question") or sample["question"]
    relevant = sentence_split(original_q)

    if typ == "adversarial":
        distractors, math_reasoning = parse_distractors(sample["reasoning"])
    else:
        distractors = []
        math_reasoning = re.sub(r"^No noise identified in the question\.\s*", "", sample["reasoning"]).strip()

    lines = ["<relevance>"]
    for s in relevant:
        lines.append(f'- "{s}": RELEVANT')
    for d in distractors:
        lines.append(f'- "{d}": DISTRACTOR')
    lines.append("</relevance>")
    lines.append("<reasoning>")
    lines.append(math_reasoning)
    lines.append("</reasoning>")
    lines.append("####")
    lines.append(answer)
    return "\n".join(lines)


def convert_file(in_path: Path, out_path: Path) -> dict:
    n_total = n_adv = n_clean = n_no_marker = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            d = json.loads(line)
            n_total += 1
            if d.get("type") == "adversarial":
                n_adv += 1
                if not NOISE_BLOCK_RE.search(d.get("reasoning", "")):
                    n_no_marker += 1
            else:
                n_clean += 1
            d["raw"] = build_target(d)
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
    return {"total": n_total, "adv": n_adv, "clean": n_clean, "adv_no_marker": n_no_marker}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=str(Path(__file__).parent))
    args = p.parse_args()
    data_dir = Path(args.data_dir)

    for name in ["noise_cot_train.jsonl", "noise_cot_test.jsonl"]:
        in_path = data_dir / name
        if not in_path.exists():
            print(f"skip: {in_path} not found")
            continue
        out_path = data_dir / name.replace(".jsonl", "_structured.jsonl")
        stats = convert_file(in_path, out_path)
        print(f"{name} -> {out_path.name}: {stats}")


if __name__ == "__main__":
    main()
