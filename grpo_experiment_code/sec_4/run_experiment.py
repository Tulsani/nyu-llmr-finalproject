import os

from datasets import load_from_disk, Dataset, load_dataset
from torch import device
from torch.utils.data import DataLoader
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import torch
from unittest.mock import patch

from vllm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer

from student.evaluate import setup_logger, evaluate, load_prompt
from student.sec_4.sec4 import run_tokenize_prompt_and_output_util, run_get_response_log_probs_util, \
    run_sft_microbatch_train_step_util
from student.utils import DEVICE
import wandb

TRAIN_DEVICE = "cuda:0"
VLLM_DEVICE = "cuda:1"

os.environ["WANDB_API_KEY"] = "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"
wandb.login(key=os.environ["WANDB_API_KEY"])


def get_eval_intellect_dataloader(dataset_path, example_count, batch_size):
    print('Intellect dataloader called')
    dataset = load_from_disk(dataset_path)
    if example_count:
        dataset = dataset.select(range(min(example_count, len(dataset))))

    prompts = []
    responses = []
    for ex in dataset:
        msgs = ex.get("messages", [])
        sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")

        prompt = sys_msg + "\n\n" + user_msg if sys_msg else user_msg
        prompts.append(prompt)
        responses.append(assistant_msg)  # full reasoning chain, not ground_truth

    dataset = Dataset.from_dict({"prompt": prompts, "response": responses})
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_eval_math_dataloader(example_count, batch_size, dataset_path="hiyouga/math12k"):
    prompt_template = load_prompt("intellect")

    math_ds = load_dataset(dataset_path, split="train")
    if example_count:
        math_ds = math_ds.select(range(min(example_count, len(math_ds))))

    prompts = [prompt_template + "\n\n" + ex["problem"] for ex in math_ds]
    gts = [ex["answer"] for ex in math_ds]
    pass


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def compute_eval_loss(model, eval_prompts, eval_resps, tokenizer, device, max_batches=20):
    model.eval()

    total_logprob = 0.0
    total_tokens = 0.0

    with torch.no_grad():
        for i in range(min(max_batches, len(eval_prompts))):
            try:
                res = run_tokenize_prompt_and_output_util(
                    [eval_prompts[i]], [eval_resps[i]], tokenizer
                )
                input_ids = res['input_ids'].to(device)
                labels = res['labels'].to(device)
                response_mask = res['response_mask'].to(device)

                res_logprobs = run_get_response_log_probs_util(
                    model, input_ids, labels, return_token_entropy=False
                )
                log_probs = res_logprobs['log_probs']

                total_logprob += (log_probs * response_mask).sum().item()
                total_tokens += response_mask.sum().item()

            except Exception as e:
                print("Eval loss error", e)

    model.train()

    if total_tokens == 0:
        return 0

    return - total_logprob / total_tokens


def run_sft_loop(
        model_train,
        dataloader,
        tokenizer,
        optimizer,
        eval_vllm_model,
        eval_prompts,
        eval_gts,
        eval_resps,
        device=DEVICE,
        epoch=3,
        grad_accum_steps=32,
        eval_after=20,
        run_name=None,

):
    model_train.train()
    step_count = 0
    optimizer.zero_grad()

    best_acc = -1

    print("Running eval once first")
    load_policy_into_vllm_instance(model_train, eval_vllm_model)
    acc, _ = evaluate(eval_vllm_model, eval_prompts, eval_gts)
    wandb.log({"eval/accuracy": acc}, step=step_count)
    print('EVAL', acc)

    for epoch_id in range(epoch):
        for batch in dataloader:
            step_count += 1
            try:
                print("STEP COUNT", step_count, "grad accumulation", grad_accum_steps)

                prompts = batch["prompt"]
                resps = batch["response"]

                print("Prompt Sizes", [len(el) for el in prompts])
                print("Response sizes", [len(el) for el in resps])

                # print("PROMPTS", prompts)
                # print("RESPONSE", resps)
                # print("")

                res = run_tokenize_prompt_and_output_util(
                    prompts, resps, tokenizer
                )
                print("tokenisze rpompt and output util done")

                input_ids = res['input_ids'].to(device)
                labels = res['labels'].to(device)
                response_mask = res['response_mask'].to(device)

                res_logprobs = run_get_response_log_probs_util(model_train, input_ids, labels,
                                                               return_token_entropy=True)
                log_probs = res_logprobs['log_probs']
                entropy = res_logprobs['token_entropy']

                print("get response log probs done")

                # TODO: Understand normalize constant thing
                loss, metadata = run_sft_microbatch_train_step_util(log_probs, response_mask, grad_accum_steps,
                                                                    response_mask.sum())
                true_loss = loss.item() * grad_accum_steps
                wandb.log({
                    "train/loss": true_loss,
                    "train/entropy": entropy.mean().item()
                }, step=step_count)
                print("loss", loss)

                print("run sft microbatch done")

                if step_count % grad_accum_steps == 0:
                    print("Taking grad accc step")
                    optimizer.step()
                    optimizer.zero_grad()

                if step_count % eval_after == 0:
                    print("Running Eval")
                    print("EVAL PROMPTS SAMPLE", eval_prompts[:2])
                    print("EVAL GROUND TRUTHS", eval_gts[:2])
                    load_policy_into_vllm_instance(model_train, eval_vllm_model)
                    acc, _ = evaluate(eval_vllm_model, eval_prompts, eval_gts)
                    eval_loss = compute_eval_loss(model_train, eval_prompts, eval_resps, tokenizer,
                                                  device) if eval_resps else None
                    print("EVAL LOSS", eval_loss)
                    wandb.log({"eval/accuracy": acc}, step=step_count)
                    if eval_loss is not None:
                        wandb.log({"eval/loss": eval_loss}, step=step_count)
                    else:
                        print("evla lOSS is found none")
                    print('eval', acc)
                    print("EVAL LOSS", eval_loss)

                    # SAVING THE MODEL
                    if acc > best_acc and run_name is not None:
                        best_acc = acc
                        os.makedirs("models", exist_ok=True)

                        # Delete an existing one so that we can have only one for instance
                        for existing in os.listdir("models"):
                            if existing.endswith(f"_{run_name}"):
                                existing_path = os.path.join("models", existing)
                                import shutil
                                shutil.rmtree(existing_path, ignore_errors=True)
                                print(f"deleted old checkpnt: {existing_path}")

                        # Saving the new best
                        save_name = f"{acc:.4f}_{run_name}"
                        save_path = os.path.join("models", save_name)
                        model_train.save_pretrained(save_path)
                        tokenizer.save_pretrained(save_path)
                        print(f"saved new best model to {save_path} (acc={acc:.4f})")

            except Exception as error:
                print("Error occurred", error)

        if step_count % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--dataset-type", type=str, default="INTELLECT")
    parser.add_argument("--intellect-train-path", default="student/data/intellect_math/train")
    parser.add_argument("--intellect-test-path", default="student/data/intellect_math/test")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--log-file", default="eval.log")

    args = parser.parse_args()

    logger = setup_logger(args.log_file)

    print("Loading training model")
    model_train = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16
    ).to(TRAIN_DEVICE)

    print("Loading tokenize")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("===loaded===")

    batch_size = 1
    example_count = 512
    learning_rate = 1e-4
    grad_accum_steps = 16

    run_name = f"SFT-dataset{args.dataset_type}-ec{str(example_count)}-ga{str(grad_accum_steps)}-batch_size{str(batch_size)}_lr{str(learning_rate)}"
    wandb.init(
        project=f"assignment-3-test",
        name=run_name,
        config={
            "model": "transformer",
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "example_count": example_count,
            "dataset": args.dataset_type,
            "grad_accum_steps": grad_accum_steps,
        }
    )

    dataloader = get_eval_math_dataloader(example_count, batch_size) \
        if args.dataset_type == 'MATH' \
        else get_eval_intellect_dataloader(args.intellect_train_path, example_count, batch_size)

    optimizer = torch.optim.AdamW(model_train.parameters(), lr=learning_rate)

    print("Loading initvllm")
    eval_vllm_model = init_vllm(
        model_id=args.model,
        device=VLLM_DEVICE,
        seed=2,
    )
    print("==loaded==")

    prompt_template = load_prompt("intellect")

    eval_prompts = []
    eval_gts = []
    eval_resps = []

    if args.dataset_type == 'MATH':
        math_ds = load_dataset("hiyouga/math12k", split="test")
        if args.max_examples:
            math_ds = math_ds.select(range(min(args.max_examples, len(math_ds))))

        eval_prompts = [prompt_template + "\n\n" + ex["problem"] for ex in math_ds]
        eval_gts = [ex["answer"] for ex in math_ds]
        eval_resps = [ex["solution"] for ex in math_ds]
    else:
        dataset = load_from_disk(args.intellect_test_path)
        if args.max_examples:
            dataset = dataset.select(range(min(args.max_examples, len(dataset))))

        for ex in dataset:
            msgs = ex.get("messages", [])
            sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
            assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")  # add this
            eval_prompts.append(sys_msg + "\n\n" + user_msg if sys_msg else user_msg)
            eval_gts.append(ex.get("ground_truth", ""))
            eval_resps.append(assistant_msg)

    run_sft_loop(
        model_train,
        dataloader,
        tokenizer,
        optimizer,
        eval_vllm_model,
        eval_prompts,
        eval_gts,
        device=TRAIN_DEVICE,
        epoch=3,
        grad_accum_steps=grad_accum_steps,
        eval_resps=eval_resps,
        eval_after=20,
        run_name=run_name
    )

    wandb.finish()


if __name__ == '__main__':
    main()

