import os
from typing import Callable, Literal

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

import utils as utils
from evaluate import evaluate
from sec_4.run_experiment import load_policy_into_vllm_instance, init_vllm, run_get_response_log_probs_util
from grpo_run.sec4 import run_tokenize_prompt_and_output_util, run_masked_normalize_util
from grpo_run.dataloader import get_gsm_adversarial_dataloaders
from grpo_run.defaults import MODEL_NAME
from grpo_run.sec7 import run_compute_policy_gradient_loss_util, run_masked_mean_util, \
    run_compute_group_normalized_rewards_util
from drgrpo_grader import pit_reward_fn
from vllm import SamplingParams
from tqdm import tqdm
import argparse
from itertools import cycle, islice


import copy

"""
CORRECTNESS-ONLY REWARD
This GRPO run is for reward based on correctness only
"""


USE_VLLM = True

TRAIN_DEVICE = "cuda:0"
VLLM_DEVICE = "cuda:0"

if USE_VLLM:
    VLLM_DEVICE = "cuda:1"


def run_grpo_microbatch_train_step_util(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
        wandb=None,
        step_count=None,
        normalize_type = "masked_mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length):
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio.
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            the policy gradient loss and its metadata.
    """

    # print(">>>> GRADIENT CLIP STEP", gradient_accumulation_steps, "loss type", loss_type)

    loss_per_token, metadata = run_compute_policy_gradient_loss_util(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange
    )

    if wandb is not None and step_count is not None:
        if "clip_fraction" in metadata:
            wandb.log({
                "train/clip_fraction": metadata['clip_fraction'],
            }, step=step_count)

    # loss = loss_per_token.mean()

    # print("loss initial 1 shape", loss_per_token.shape, loss_per_token)
    if normalize_type == "masked_mean":
        print("using masked mean")
        masked_loss = run_masked_mean_util(loss_per_token, response_mask, dim=1)
    elif normalize_type == "masked_normalize":
        print("using masked normalize")
        masked_loss = run_masked_normalize_util(loss_per_token, response_mask, dim=-1, normalize_constant=1024)
    else:
        raise Exception("Got normalize type that was not valid")

    # print("masked loss 1", masked_loss)
    masked_loss = masked_loss.mean()
    # print("masked loss 2", masked_loss)
    masked_loss = masked_loss / gradient_accumulation_steps
    # print("masked loss 3", masked_loss)

    masked_loss.backward()

    return masked_loss, metadata
    pass


def run_grpo_training(
        model_train,
        dataloader,
        tokenizer,
        optimizer,
        eval_vllm_model,
        eval_prompts,
        eval_gts,
        device=utils.DEVICE,
        eval_after=5,
        run_name=None,

        # GRPO PARAMETERS
        n_grpo_steps: int = 200,
        advantage_eps: float = 1e-6,
        rollout_batch_size: int = 16,
        group_size: int = 8,
        sampling_temperature: float = 0.7,
        sampling_min_tokens: int = 4,
        sampling_max_tokens: int = 1024,
        epochs_per_rollout_batch: int = 1,  # On-policy
        train_batch_size: int = 64,  # On-policy
        gradient_accumulation_steps: int = 128,
        loss_type: Literal[
            "no_baseline",
            "reinforce_with_baseline",
            "grpo_clip",
        ] = "reinforce_with_baseline",
        use_std_normalization: bool = True,
        grpo_clip=1.0,
        normalize_type = "masked_mean",
):
    model_train.train()
    step_count = 0
    optimizer.zero_grad()

    best_acc = -1
    print("Running eval once first")
    load_policy_into_vllm_instance(model_train, eval_vllm_model)
    acc, reward = evaluate(eval_vllm_model, eval_prompts, eval_gts,
                                           sampling_temperature=sampling_temperature,
                                           sampling_max_tokens=sampling_max_tokens, sampling_min_tokens=sampling_min_tokens,
                                           stop_tokens=['</answer>'],
                                           reward_fn=pit_reward_fn,
                                           )
    wandb.log({"eval/accuracy": acc}, step=step_count)
    print('EVAL', acc)

    # train_iter = iter(dataloader)
    train_iter = cycle(dataloader)

    for step in range(n_grpo_steps):
        print("starting grpo step", step)
        questions_batch = next(train_iter)

        f = model_train.eval()

        try:
            old_policy_model = copy.deepcopy(model_train).eval().to(device)
            for p in old_policy_model.parameters():
                p.requires_grad = False

            if USE_VLLM:
                load_policy_into_vllm_instance(old_policy_model, eval_vllm_model)

            rollout_responses = []
            repeated_ground_truths = []

            sampling_params = SamplingParams(
                temperature=sampling_temperature,
                min_tokens=sampling_min_tokens,
                max_tokens=sampling_max_tokens,
                stop=["</answer>"],  # TODO: might need change
            )

            if USE_VLLM:
                load_policy_into_vllm_instance(model_train, eval_vllm_model)
                for question, gt in zip(questions_batch["prompts"], questions_batch["ground_truths"]):
                    for _ in range(group_size):
                        outputs = eval_vllm_model.generate(question, sampling_params=sampling_params)
                        response = outputs[0].outputs[0].text
                        print("this prompt:", question)
                        print("this response:", response)
                        rollout_responses.append(response)
                        repeated_ground_truths.append(gt)
            else:
                with torch.no_grad():
                    for question, gt in zip(questions_batch["prompts"], questions_batch["ground_truths"]):
                        inputs = tokenizer(question, return_tensors="pt").to(device)
                        for _ in range(group_size):
                            output = model_train.generate(
                                **inputs,
                                max_new_tokens=sampling_max_tokens,
                                min_new_tokens=sampling_min_tokens,
                                temperature=sampling_temperature,
                                do_sample=True,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                            new_tokens = output[0][inputs["input_ids"].shape[1]:]
                            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                            print("response:", response)
                            rollout_responses.append(response)
                            repeated_ground_truths.append(gt)


            print("got rollouts", len(rollout_responses), len(repeated_ground_truths))
            # print("response:", response)
            print("repeated gts:", repeated_ground_truths)
            repeated_prompts = []
            for question in questions_batch["prompts"]:
                for _ in range(group_size):
                    repeated_prompts.append(question)

            advantages, raw_rewards, metadata_rewards = run_compute_group_normalized_rewards_util(
                reward_fn=pit_reward_fn,
                rollout_responses=rollout_responses,
                repeated_ground_truths=repeated_ground_truths,
                group_size=group_size,
                advantage_eps=advantage_eps,
                normalize_by_std=use_std_normalization,
                aggregate_type="CONSISTENCY"
            )

            print("got advantages, raw rewards and metadata rewards", advantages, raw_rewards, metadata_rewards)

            train_reward_reportable = {f"train/reward/{key}": metadata_rewards[key] for key in metadata_rewards}
            wandb.log(train_reward_reportable, step=step_count)

            print("got advantages, raw rewards and metadata rewards", advantages, raw_rewards, metadata_rewards)

            micro_train_batch_size = train_batch_size // gradient_accumulation_steps
            n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

            full_advantages = advantages
            full_raw_rewards = raw_rewards

            model_train.train()
            for epoch_id in range(epochs_per_rollout_batch):
                print("EPOCH", epoch_id)
                for i in range(n_microbatches_per_rollout_batch):
                    start_index = i * micro_train_batch_size
                    end_index = (i + 1) * micro_train_batch_size

                    responses = rollout_responses[start_index:end_index]

                    advantages = full_advantages[start_index:end_index].to(device).unsqueeze(1)
                    raw_rewards = full_raw_rewards[start_index:end_index].to(device).unsqueeze(1)

                    questions = repeated_prompts[start_index:end_index]

                    tokenized = run_tokenize_prompt_and_output_util(
                        prompt_strs=questions,
                        output_strs=responses,
                        tokenizer=tokenizer,
                    )

                    input_ids_input = tokenized["input_ids"].to(device)
                    labels_shifted = tokenized["labels"].to(device)
                    resp_mask = tokenized["response_mask"].to(device).float()

                    log_probs_dict = run_get_response_log_probs_util(
                        model=model_train,
                        input_ids=input_ids_input,
                        labels=labels_shifted,
                        return_token_entropy=True
                    )
                    log_prob = log_probs_dict['log_probs']
                    entropy = log_probs_dict['token_entropy']

                    if USE_VLLM:
                        old_log_probs_dict = run_get_response_log_probs_util(
                            model=old_policy_model,
                            input_ids=input_ids_input,
                            labels=labels_shifted,
                            return_token_entropy=False,
                        )
                        old_log_probs = old_log_probs_dict['log_probs']
                    else:
                        raise Exception("We have to use VLLM")

                    loss, metadata_loss = run_grpo_microbatch_train_step_util(
                        policy_log_probs=log_prob,
                        response_mask=resp_mask,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        loss_type=loss_type,
                        raw_rewards=raw_rewards,
                        advantages=advantages,
                        old_log_probs=old_log_probs,
                        cliprange=grpo_clip,
                        wandb=wandb,  # pass wandb
                        step_count=step_count,
                        normalize_type=normalize_type,
                    )
                    true_loss = loss.item() * gradient_accumulation_steps

                    print("microbatch ran LOSS VALUE:", true_loss)
                    wandb.log({
                        "train/loss": true_loss,
                    }, step=step_count)

                    if entropy is not None:
                        print("reporting entropy", entropy.mean().item())
                        wandb.log({
                            "train/entropy": entropy.mean().item()
                        }, step=step_count)

                    if (i + 1) % gradient_accumulation_steps == 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=1.0)
                        wandb.log({
                            "train/grad_norm": grad_norm.item()
                        }, step=step_count)
                        torch.nn.utils.clip_grad_norm_(model_train.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()

                    step_count += 1

                print("TAKING GRAD STEP after epoch", epoch_id)
                grad_norm = torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=1.0)
                wandb.log({"train/grad_norm": grad_norm.item()}, step=step_count)
                optimizer.step()
                optimizer.zero_grad()


            if step % eval_after == 0:
                #
                # load_policy_into_vllm_instance(model_train, eval_vllm_model)
                # acc, reward = evaluate(eval_vllm_model, eval_prompts, eval_gts, sampling_temperature=sampling_temperature,sampling_max_tokens=sampling_max_tokens,sampling_min_tokens=sampling_min_tokens,stop_tokens=['</answer>'])
                if USE_VLLM:
                    load_policy_into_vllm_instance(model_train, eval_vllm_model)
                    acc, reward = evaluate(eval_vllm_model, eval_prompts, eval_gts,
                                           sampling_temperature=sampling_temperature,
                                           sampling_max_tokens=sampling_max_tokens, sampling_min_tokens=sampling_min_tokens,
                                           stop_tokens=['</answer>'],
                                           reward_fn=pit_reward_fn,
                                           )
                else:
                    model_train.eval()
                    correct = 0
                    total = 0
                    print("Running eval")
                    with torch.no_grad():
                        for prompt, gt in tqdm(zip(eval_prompts, eval_gts)):
                            inputs = tokenizer(prompt, return_tensors="pt").to(device)
                            output = model_train.generate(
                                **inputs,
                                max_new_tokens=sampling_max_tokens,
                                min_new_tokens=sampling_min_tokens,
                                temperature=sampling_temperature,
                                do_sample=True,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                            new_tokens = output[0][inputs["input_ids"].shape[1]:]
                            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                            reward_score = pit_reward_fn([response], [gt])
                            correct += int(reward_score['answer_reward'] > 0)
                            total += 1
                    acc = correct / total if total > 0 else 0

                wandb.log({"eval/accuracy": acc}, step=step_count)
                reward_reportable = {f"eval/reward/{key}": reward[key] for key in reward}
                wandb.log(reward_reportable, step=step_count)
                print(f"Step {step_count} eval:", acc)

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
            print("GRPO STEP", step, "SKIPPED", str(error))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--loss_type", type=str, default="grpo_clip")
    parser.add_argument("--train_dataset_path", type=str, default="student/data/pit/pit-train.jsonl")
    parser.add_argument("--test_dataset_path", type=str, default="student/data/pit/pit-test.jsonl")
    parser.add_argument("--use_std", type=str, default="FALSE")
    parser.add_argument("--reduce", type=float, default=1)
    parser.add_argument("--normalize_type", type=str, default="masked_mean")
    parser.add_argument("--eval_after", type=int, default=5)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)

    args = parser.parse_args()

    print("loading policy model")
    policy_model_name = args.model_name
    policy = AutoModelForCausalLM.from_pretrained(
        policy_model_name,
        torch_dtype=torch.bfloat16,
        device_map=TRAIN_DEVICE,
    )
    tokenizer = AutoTokenizer.from_pretrained(policy_model_name)
    print("==loaded==")

    # Hyper parameters
    # =======
    # n_grpo_steps: int = 200
    n_grpo_steps = 1000000000 # a very large value
    learning_rate: float = args.learning_rate
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 16
    group_size: int = 8

    sampling_temperature: float = 0.7
    sampling_min_tokens: int = 4
    sampling_max_tokens: int = 1024

    epochs_per_rollout_batch: int = 1  # On-policy
    # train_batch_size: int = 64  # On-policy
    train_batch_size = 128
    gradient_accumulation_steps: int = 128
    gpu_memory_utilization: float = 0.8
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
    ] = args.loss_type
    use_std_normalization: bool = args.use_std == "TRUE"
    reduce = args.reduce
    normalize_type = args.normalize_type
    eval_after = int(args.eval_after)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    # =====

    # wandb things
    run_name = f"PIT-GRPO-lt{loss_type}-ga{str(gradient_accumulation_steps)}-ngs{str(n_grpo_steps)}_lr{str(learning_rate)}_usestd{str(use_std_normalization)}_nt{normalize_type}"
    wandb.init(
        project=f"PIT-GRPO",
        name=run_name,
        config={
            "model": "transformer",
            "n_grpo_steps": n_grpo_steps,
            "learning_rate": learning_rate,
        }
    )
    # ===

    n_prompts_per_rollout_batch = rollout_batch_size // group_size

    print("n_prompts_per_rollout_batch", n_prompts_per_rollout_batch)

    print("n_prompts_per_rollout_batch", n_prompts_per_rollout_batch)

    # get the dataloaders
    # train_dataloader, test_dataloader = get_countdown_dataloaders(
    #     "student/data/countdown/dataset", n_prompts_per_rollout_batch, reduce_test=False
    # )

    train_dataloader = get_gsm_adversarial_dataloaders(
        dataset_path=args.train_dataset_path,
        n_prompts_per_rollout_batch=n_prompts_per_rollout_batch,
    )

    test_dataloader = get_gsm_adversarial_dataloaders(
        dataset_path=args.test_dataset_path,
        n_prompts_per_rollout_batch=n_prompts_per_rollout_batch,
        reduce=reduce
    )

    #
    eval_vllm_model = None
    if USE_VLLM:
        print("Loading initvllm in grpo")
        eval_vllm_model = init_vllm(
            model_id=policy_model_name,
            device=VLLM_DEVICE,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=2,
        )
        print("==loaded==")

    eval_prompts = []
    eval_gts = []
    for batch in test_dataloader:
        eval_prompts.extend(batch["prompts"])
        eval_gts.extend(batch["ground_truths"])

    # call run grpo loop here
    run_grpo_training(
        policy,
        train_dataloader,
        tokenizer,
        optimizer,
        eval_vllm_model,
        eval_prompts,
        eval_gts,
        device=TRAIN_DEVICE,
        eval_after=eval_after,
        run_name=run_name,

        # GRPO PARAMETERS
        n_grpo_steps=n_grpo_steps,
        advantage_eps=advantage_eps,
        rollout_batch_size=rollout_batch_size,
        group_size=group_size,
        sampling_temperature=sampling_temperature,
        sampling_min_tokens=sampling_min_tokens,
        sampling_max_tokens=sampling_max_tokens,
        epochs_per_rollout_batch=epochs_per_rollout_batch,  # On-policy
        train_batch_size=train_batch_size,  # On-policy
        gradient_accumulation_steps=gradient_accumulation_steps,
        loss_type=loss_type,
        use_std_normalization=use_std_normalization,
        grpo_clip=0.1,
        normalize_type=normalize_type,
    )
