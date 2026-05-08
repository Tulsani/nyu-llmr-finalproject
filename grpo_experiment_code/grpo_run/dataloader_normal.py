import json
from torch.utils.data import DataLoader


def format_prompt(question: str) -> str:
    return f"""Solve this math problem step by step:

{question}

Provide your final answer in the format:
[reasoning steps]
####
[final answer (just the number)]"""


def get_gsm_normal_dataloaders(
    dataset_path: str,
    n_prompts_per_rollout_batch: int,
    reduce: float = None,
):
    with open(dataset_path, "r") as f:
        records = json.load(f)

    if reduce is not None:
        records = records[:max(1, int(len(records) * reduce))]

    def collate_fn(batch):
        return {
            "prompts":       [format_prompt(item["question"]) for item in batch],
            "ground_truths": [item["answer"] for item in batch],
        }

    return DataLoader(
        records,
        batch_size=n_prompts_per_rollout_batch,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
    )


"""
uv run python student/sec_7/test.py --model-name <> --dataset_path student/data/pit/gsm8k_processed_test.json --use_normal_dataloader TRUE
"""