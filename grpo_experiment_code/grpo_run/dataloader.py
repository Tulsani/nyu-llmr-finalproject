import json
import torch
from torch.utils.data import DataLoader, Dataset

class GSMAdversarialDataset(Dataset):
    def __init__(self, records):
        self.items = []
        for record in records:
            # Original question
            self.items.append({
                "question": record["original_question"],
                "answer": record["original_answer"],
                "is_adversarial": False,
            })
            # 3 adversarial questions
            adversarials = record["modified_questions"]["adverserials"]
            answers = record["modified_questions"]["answers"]
            for q, a in zip(adversarials, answers):
                self.items.append({
                    "question": q,
                    "answer": record["original_answer"],
                    "is_adversarial": True,
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def format_prompt(question: str) -> str:
    return f"""Solve this math problem step by step:

{question}

Provide your final answer in the format:
[reasoning steps]
####
[final answer (just the number)]"""


def get_gsm_adversarial_dataloaders(
    dataset_path: str,
    n_prompts_per_rollout_batch: int,
    reduce: float = None,
):
    # Load JSONL
    records = []
    with open(dataset_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if reduce is not None:
        n_keep = max(1, int(len(records) * reduce))
        records = records[:n_keep]

    dataset = GSMAdversarialDataset(records)

    def collate_fn(batch):
        return {
            "prompts":        [format_prompt(item["question"]) for item in batch],
            "ground_truths":  [item["answer"] for item in batch],
            "is_adversarial": [item["is_adversarial"] for item in batch],
        }

    loader = DataLoader(
        dataset,
        batch_size=n_prompts_per_rollout_batch,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
    )

    return loader