import json
import torch
from torch.utils.data import DataLoader, Dataset

class GSMAdversarialDataset(Dataset):
    def __init__(self, records):
        self.items = []
        for record in records:
            questions = []
            answers = []

            # original
            questions.append(record["original_question"])
            answers.append(record["original_answer"])

            # adversarials
            adversarials = record["modified_questions"]["adverserials"]
            adv_answers = [record["original_answer"] for _ in range(len(adversarials))]

            for q, a in zip(adversarials, adv_answers):
                questions.append(q)
                answers.append(a)

            self.items.append({
                "questions": questions,  # list of 4
                "answers": answers  # list of 4
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
        question_groups = []

        for item in batch:
            prompts = [format_prompt(q) for q in item["questions"]]

            question_groups.append({
                "prompts": prompts,  # list of 4
                "ground_truths": item["answers"],  # list of 4
            })

        return {
            "question_groups": question_groups
        }

    loader = DataLoader(
        dataset,
        batch_size=n_prompts_per_rollout_batch,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
    )

    return loader

if __name__ == '__main__':
    loader = get_gsm_adversarial_dataloaders(
        dataset_path="../data/pit/pit-train.jsonl",
        n_prompts_per_rollout_batch=1,  # 🔥 important for now
    )

    batch = next(iter(loader))

    print("BATCH KEYS:", batch.keys())
    print("Num question groups:", len(batch["question_groups"]))

    group = batch["question_groups"][0]

    print("\n--- GROUP ---")
    print("Num prompts:", len(group["prompts"]))
    print("Num GTs:", len(group["ground_truths"]))

    for i, (p, gt) in enumerate(zip(group["prompts"], group["ground_truths"])):
        print(f"\nPrompt {i}:")
        print(p[:200])  # print first 200 chars
        print("GT:", gt)