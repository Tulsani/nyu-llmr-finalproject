"""
Append clean anchor samples from a *_data.jsonl into a noise_cot_*.jsonl file
for any question that isn't already represented in the target.

A question is "represented" if it appears as either a clean `question` or as
the `original_question` of an adversarial row in the target.

When extending the test split, pass --leak_check_against to refuse appending
any question that already appears in the train split (prevents eval leakage).

Appended rows match the clean-row schema in noise_cot files:
    {question, original_question, answer, raw, reasoning, type, answer_match}

Run:
    # train (default)
    python dataset/append_clean_anchors.py
    python dataset/append_clean_anchors.py --dry_run

    # test (with leakage guard against train)
    python dataset/append_clean_anchors.py \\
        --source dataset/test_data.jsonl \\
        --noise_cot dataset/noise_cot_test.jsonl \\
        --leak_check_against dataset/noise_cot_train.jsonl --dry_run
"""

import argparse
import json
import os


CLEAN_PREFIX = "No noise identified in the question. "


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",     default="dataset/train_data.jsonl",
                        help="Source jsonl with clean ('source'=='original') rows to draw anchors from.")
    parser.add_argument("--noise_cot",  default="dataset/noise_cot_train.jsonl",
                        help="Target noise-cot file to append into.")
    parser.add_argument("--leak_check_against", default=None,
                        help="Optional: path to another noise-cot file (e.g. the train split). "
                             "Any source question represented there is dropped to prevent leakage.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Report what would be appended without writing.")
    args = parser.parse_args()

    train_rows = load_jsonl(args.source)
    noise_rows = load_jsonl(args.noise_cot)
    leak_rows  = load_jsonl(args.leak_check_against) if args.leak_check_against else []

    def represented_set(rows):
        s = set()
        for r in rows:
            if r.get("type") == "clean":
                s.add(r["question"])
            else:
                key = r.get("original_question") or r.get("question")
                if key:
                    s.add(key)
        return s

    represented = represented_set(noise_rows)
    leak_set    = represented_set(leak_rows)

    to_append = []
    n_leak_skipped = 0
    for r in train_rows:
        # *_data.jsonl mixes clean ("original") and adversarial rows.
        # Only clean rows are valid anchors.
        if r.get("source") != "original":
            continue
        q = r["question"]
        if q in represented:
            continue
        if q in leak_set:
            n_leak_skipped += 1
            continue
        reasoning = CLEAN_PREFIX + r["reasoning"]
        raw = f"{reasoning}\n#### {r['answer']}"
        to_append.append({
            "question": q,
            "original_question": q,
            "answer": r["answer"],
            "raw": raw,
            "reasoning": reasoning,
            "type": "clean",
            "answer_match": True,
        })

    print(f"source rows:          {len(train_rows)}")
    print(f"noise_cot rows:       {len(noise_rows)}")
    print(f"already represented:  {len(represented)} unique questions")
    if args.leak_check_against:
        print(f"leak-check skipped:   {n_leak_skipped} (present in {args.leak_check_against})")
    print(f"to append:            {len(to_append)} clean anchors")

    if args.dry_run or not to_append:
        if to_append:
            print("\nFirst 2 to-append previews:")
            for r in to_append[:2]:
                print(json.dumps(r, indent=2)[:400])
        return

    with open(args.noise_cot, "a", encoding="utf-8") as f:
        for r in to_append:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nAppended {len(to_append)} rows to {args.noise_cot}")


if __name__ == "__main__":
    main()
