#!/usr/bin/env python3
"""
Evaluate a model on the transition MCQ dataset by comparing log probabilities
of each multiple-choice option using the same chat prompt format as
training/eval_transition_mcq.py.

Inputs: one or more JSONL datasets where each line contains fields produced by
training/eval_transition_mcq.py, including: state, action, choices,
correct_choice_index, label, etc.

Outputs: per-dataset detailed results JSON files and an overall summary JSON.
"""

import os
import argparse
import json
import gc
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# Must match training prompts used in training/eval_transition_mcq.py
TRANSITION_SYSTEM_PROMPT = (
    "You are a Minecraft transition model. Given a current state and ONE action, return the next state text in the exact "
    "format used by the data (same headers and lists). Do not output actions. Enforce environment rules: cannot move/smelt into [0]; "
    "crafting outputs appear at [0] and must be moved to an inventory slot to complete. Output only the next state text."
)

def load_model_and_tokenizer(
    base_model: str,
    adapter_dir: str,
    load_in_8bit: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    config = AutoConfig.from_pretrained(base_model)
    quant_config = getattr(config, "quantization_config", None)

    if load_in_8bit and not torch.cuda.is_available():
        print("Warning: 8-bit load requested but CUDA is unavailable; using full precision.")
        load_in_8bit = False

    if load_in_8bit and quant_config is not None and not is_bnb_quant_config(quant_config):
        print(
            "Warning: Model has a non-BitsAndBytes quantization config; "
            "skipping 8-bit load and using the model's native quantization."
        )
        load_in_8bit = False

    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "config": config,
    }
    if load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    try:
        if hasattr(model, "config"):
            model.config.use_cache = True
    except Exception:
        pass

    return model, tokenizer


def format_transition_user_content(state_text: str, action_text: str) -> str:
    return state_text.strip() + "\n\nAction:\n" + action_text.strip()


def build_transition_messages(state_text: str, action_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": TRANSITION_SYSTEM_PROMPT},
        {"role": "user", "content": format_transition_user_content(state_text, action_text)},
    ]


def get_model_input_device(model: AutoModelForCausalLM) -> torch.device:
    try:
        emb = model.get_input_embeddings()
        if hasattr(emb, "weight"):
            return emb.weight.device
    except Exception:
        pass
    for p in model.parameters():
        try:
            return p.device
        except Exception:
            continue
    return torch.device("cpu")


def build_prompt_and_full_text(
    tokenizer: AutoTokenizer,
    state_text: str,
    action_text: str,
    choice_text: str,
) -> Tuple[str, str]:
    """
    Returns (prompt_only, full_text) strings using the tokenizer's chat template.
    prompt_only ends with the assistant prefix (generation prompt), and full_text
    is prompt_only + choice_text.
    """
    messages = build_transition_messages(state_text, action_text)
    prompt_only = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = prompt_only + choice_text
    return prompt_only, full_text


def build_prompt_and_full_text_from_messages(
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    choice_text: str,
) -> Tuple[str, str]:
    """
    Returns (prompt_only, full_text) using provided chat messages.
    prompt_only ends with the assistant prefix (generation prompt), and full_text
    is prompt_only + choice_text.
    """
    prompt_only = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = prompt_only + choice_text
    return prompt_only, full_text


@torch.no_grad()
def score_choice_logprob(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    state_text: str,
    action_text: str,
    choice_text: str,
) -> float:
    prompt_only, full_text = build_prompt_and_full_text(tokenizer, state_text, action_text, choice_text)
    inputs = tokenizer(full_text, return_tensors="pt")
    device = get_model_input_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    prefix_tok = tokenizer(prompt_only, return_tensors="pt")
    prefix_len = int(prefix_tok["input_ids"].shape[1])

    outputs = model(**inputs)
    logits = outputs.logits[:, :-1]
    target_ids = inputs["input_ids"][..., 1:]

    seq_len = target_ids.shape[1]
    mask = torch.zeros_like(target_ids, dtype=torch.bool)
    # Start accumulating from the first token after the prompt
    mask[:, max(0, prefix_len - 1) :] = True

    logprobs = torch.log_softmax(logits, dim=-1)
    token_logprobs = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    masked = token_logprobs.masked_select(mask)
    return float(masked.sum().item())


@torch.no_grad()
def score_choice_logprob_from_messages(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    choice_text: str,
) -> float:
    prompt_only, full_text = build_prompt_and_full_text_from_messages(tokenizer, messages, choice_text)
    inputs = tokenizer(full_text, return_tensors="pt")
    device = get_model_input_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    prefix_tok = tokenizer(prompt_only, return_tensors="pt")
    prefix_len = int(prefix_tok["input_ids"].shape[1])

    outputs = model(**inputs)
    logits = outputs.logits[:, :-1]
    target_ids = inputs["input_ids"][..., 1:]

    seq_len = target_ids.shape[1]
    mask = torch.zeros_like(target_ids, dtype=torch.bool)
    mask[:, max(0, prefix_len - 1) :] = True

    logprobs = torch.log_softmax(logits, dim=-1)
    token_logprobs = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    masked = token_logprobs.masked_select(mask)
    return float(masked.sum().item())


def build_messages_for_row(row: Dict) -> List[Dict[str, str]]:
    """
    Build chat messages for a dataset row.
    - If the row provides 'system' and 'user' (ScienceWorld format), use them.
    - Otherwise, fall back to Minecraft format using state/action fields.
    """
    sys = row.get("system", "")
    usr = row.get("user", "")
    if isinstance(sys, str) and isinstance(usr, str) and sys.strip() and usr.strip():
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": usr},
        ]
    # Fallback to Minecraft transition prompt
    state_text = row.get("state", "")
    action_text = row.get("action", "")
    return build_transition_messages(state_text, action_text)


def ensure_dir_for_prefix(prefix: str) -> None:
    d = os.path.dirname(os.path.abspath(prefix))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def is_cuda_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "cuda out of memory" in msg


def clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def is_bnb_quant_config(quant_config: object) -> bool:
    if isinstance(quant_config, BitsAndBytesConfig):
        return True
    if isinstance(quant_config, dict):
        if "load_in_8bit" in quant_config or "load_in_4bit" in quant_config:
            return True
        for key in quant_config.keys():
            if key.startswith("bnb_"):
                return True
    return False


def extract_mcq_fields(row: Dict) -> Tuple[List[str], int]:
    choices = row.get("choices", None)
    if not choices:
        choices = row.get("options", [])
    correct_idx_val = row.get("correct_choice_index", None)
    if correct_idx_val is None:
        correct_idx_val = row.get("correct_index", -1)
    try:
        correct_idx = int(correct_idx_val)
    except Exception:
        correct_idx = -1
    return choices, correct_idx


def build_mcq_items(rows: List[Dict]) -> Tuple[List[Dict], int]:
    mcq_items: List[Dict] = []
    skipped_non_mcq = 0
    for row in rows:
        choices, correct_idx = extract_mcq_fields(row)
        if not choices or correct_idx < 0:
            skipped_non_mcq += 1
            continue
        mcq_items.append({"row": row, "choices": choices, "correct_idx": correct_idx})
    return mcq_items, skipped_non_mcq


def build_result_entry(
    item: Dict,
    mcq_index: int,
    scores: List[float],
    pred_idx: int,
    is_correct: bool,
) -> Dict:
    row = item["row"]
    return {
        "question_idx": mcq_index + 1,
        "id": row.get("id"),
        "file": row.get("file"),
        "step_idx": row.get("step_idx"),
        "input_state": row.get("state", ""),
        "input_action": row.get("action", ""),
        "input_system": row.get("system"),
        "input_user": row.get("user"),
        "choices": item["choices"],
        "choice_logprobs": scores,
        "predicted_idx": pred_idx,
        "correct_idx": item["correct_idx"],
        "is_correct": bool(is_correct),
        "label": row.get("label", ""),
        "topic": row.get("topic", ""),
    }


def build_skipped_oom_entry(item: Dict, mcq_index: int) -> Dict:
    row = item["row"]
    return {
        "question_idx": mcq_index + 1,
        "id": row.get("id"),
        "file": row.get("file"),
        "step_idx": row.get("step_idx"),
        "input_state": row.get("state", ""),
        "input_action": row.get("action", ""),
        "input_system": row.get("system"),
        "input_user": row.get("user"),
        "choices": item["choices"],
        "choice_logprobs": None,
        "predicted_idx": -1,
        "correct_idx": item["correct_idx"],
        "is_correct": None,
        "label": row.get("label", ""),
        "topic": row.get("topic", ""),
        "skipped_oom": True,
    }


def evaluate_mcq_indices(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    mcq_items: List[Dict],
    indices: List[int],
    desc: str,
) -> Tuple[Dict[int, Dict], List[int]]:
    results_by_idx: Dict[int, Dict] = {}
    oom_indices: List[int] = []
    for idx in tqdm(indices, desc=desc):
        item = mcq_items[idx]
        try:
            messages = build_messages_for_row(item["row"])
            scores: List[float] = []
            for choice in item["choices"]:
                score = score_choice_logprob_from_messages(model, tokenizer, messages, choice)
                scores.append(score)
            pred_idx = int(torch.tensor(scores).argmax().item()) if scores else -1
            is_correct = pred_idx == item["correct_idx"]
            results_by_idx[idx] = build_result_entry(item, idx, scores, pred_idx, is_correct)
        except Exception as exc:
            if is_cuda_oom_error(exc):
                print(f"CUDA OOM at MCQ index {idx + 1}; skipping for 8-bit retry.")
                clear_cuda_cache()
                oom_indices.append(idx)
                continue
            raise
    return results_by_idx, oom_indices


def evaluate_datasets(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_paths: List[str],
    output_prefix: str,
    max_eval: int = 0,
    base_model: str = "",
    adapter_dir: str = "",
) -> None:
    ensure_dir_for_prefix(output_prefix)

    dataset_records: List[Dict] = []
    total_oom = 0

    for i, dataset_path in enumerate(dataset_paths):
        print(f"\nEvaluating on dataset {i + 1}: {dataset_path}")

        rows: List[Dict] = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue

        if max_eval and max_eval > 0:
            rows = rows[: max_eval]

        mcq_items, skipped_non_mcq = build_mcq_items(rows)
        indices = list(range(len(mcq_items)))

        results_by_idx, oom_indices = evaluate_mcq_indices(
            model=model,
            tokenizer=tokenizer,
            mcq_items=mcq_items,
            indices=indices,
            desc=f"Dataset {i + 1}",
        )
        total_oom += len(oom_indices)

        dataset_records.append(
            {
                "dataset_index": i,
                "dataset_path": dataset_path,
                "mcq_items": mcq_items,
                "results_by_idx": results_by_idx,
                "skipped_non_mcq": skipped_non_mcq,
                "oom_indices": oom_indices,
            }
        )

    if total_oom > 0:
        if not torch.cuda.is_available():
            print("CUDA OOM detected but CUDA is unavailable; skipping 8-bit retry.")
        else:
            print(f"\nRetrying {total_oom} skipped questions with 8-bit model...")
            del model
            clear_cuda_cache()
            model_8bit, tokenizer = load_model_and_tokenizer(
                base_model=base_model,
                adapter_dir=adapter_dir,
                load_in_8bit=True,
            )
            for record in dataset_records:
                if not record["oom_indices"]:
                    continue
                retry_results, retry_oom = evaluate_mcq_indices(
                    model=model_8bit,
                    tokenizer=tokenizer,
                    mcq_items=record["mcq_items"],
                    indices=record["oom_indices"],
                    desc=f"Dataset {record['dataset_index'] + 1} (8-bit retry)",
                )
                record["results_by_idx"].update(retry_results)
                record["oom_indices"] = retry_oom
            remaining_oom = sum(len(r["oom_indices"]) for r in dataset_records)
            if remaining_oom:
                print(f"Warning: {remaining_oom} questions still failed after 8-bit retry.")

    all_accuracies: List[float] = []
    all_correct = 0
    all_total = 0
    total_skipped_oom = 0
    overall_topic_totals: Dict[str, int] = {}
    overall_topic_corrects: Dict[str, int] = {}

    for record in dataset_records:
        dataset_path = record["dataset_path"]
        mcq_items = record["mcq_items"]
        results_by_idx = record["results_by_idx"]
        skipped_non_mcq = record["skipped_non_mcq"]
        skipped_oom = len(record["oom_indices"])
        total_skipped_oom += skipped_oom

        correct = 0
        total = 0
        detailed_results: List[Dict] = []

        label_totals: Dict[str, int] = {}
        label_corrects: Dict[str, int] = {}
        topic_totals: Dict[str, int] = {}
        topic_corrects: Dict[str, int] = {}

        for idx, item in enumerate(mcq_items):
            result = results_by_idx.get(idx)
            if result is None:
                detailed_results.append(build_skipped_oom_entry(item, idx))
                continue

            detailed_results.append(result)
            total += 1
            if result["is_correct"]:
                correct += 1

            label = result.get("label", "")
            if label:
                label_totals[label] = label_totals.get(label, 0) + 1
                if result["is_correct"]:
                    label_corrects[label] = label_corrects.get(label, 0) + 1
            topic = result.get("topic", "")
            if topic:
                topic_totals[topic] = topic_totals.get(topic, 0) + 1
                if result["is_correct"]:
                    topic_corrects[topic] = topic_corrects.get(topic, 0) + 1

        acc = correct / max(1, total)
        all_accuracies.append(acc)
        all_correct += correct
        all_total += total

        for k, v in topic_totals.items():
            overall_topic_totals[k] = overall_topic_totals.get(k, 0) + v
        for k, v in topic_corrects.items():
            overall_topic_corrects[k] = overall_topic_corrects.get(k, 0) + v

        label_accuracy: Dict[str, float] = {}
        for k, v in label_totals.items():
            c = label_corrects.get(k, 0)
            label_accuracy[k] = c / max(1, v)
        topic_accuracy: Dict[str, float] = {}
        for k, v in topic_totals.items():
            c = topic_corrects.get(k, 0)
            topic_accuracy[k] = c / max(1, v)

        results_data = {
            "summary": {
                "accuracy": acc,
                "total": total,
                "correct": correct,
                "skipped_non_mcq": skipped_non_mcq,
                "skipped_oom": skipped_oom,
                "dataset_path": dataset_path,
                "label_totals": label_totals,
                "label_corrects": label_corrects,
                "label_accuracy": label_accuracy,
                "topic_totals": topic_totals,
                "topic_corrects": topic_corrects,
                "topic_accuracy": topic_accuracy,
            },
            "questions": detailed_results,
        }

        output_file = f"{output_prefix}_dataset_{record['dataset_index'] + 1}.json"
        with open(output_file, "w", encoding="utf-8") as f_out:
            json.dump(results_data, f_out, indent=2)

        print(f"Dataset {record['dataset_index'] + 1} results saved to {output_file}")
        print(f"Dataset {record['dataset_index'] + 1} accuracy: {acc:.4f} ({correct}/{total})")

    total_acc = all_correct / max(1, all_total)
    overall_topic_accuracy: Dict[str, float] = {}
    for k, v in overall_topic_totals.items():
        c = overall_topic_corrects.get(k, 0)
        overall_topic_accuracy[k] = c / max(1, v)
    overall_summary = {
        "total_accuracy": total_acc,
        "total_correct": all_correct,
        "total_questions": all_total,
        "dataset_accuracies": all_accuracies,
        "dataset_paths": dataset_paths,
        "topic_totals": overall_topic_totals,
        "topic_corrects": overall_topic_corrects,
        "topic_accuracy": overall_topic_accuracy,
        "total_skipped_oom": total_skipped_oom,
    }

    overall_file = f"{output_prefix}_summary.json"
    with open(overall_file, "w", encoding="utf-8") as f_sum:
        json.dump(overall_summary, f_sum, indent=2)

    print(f"\nOverall summary saved to {overall_file}")


def _hf_data_file(task: str, split: str) -> str:
    return f"data/{task}/{split}.jsonl"


def load_rows_from_hf_dataset(repo_id: str, task: str, split: str, max_eval: int = 0) -> Tuple[str, List[Dict]]:
    """
    Load JSONL rows from a Hugging Face dataset repo that stores files under:
      data/<task>/<split>.jsonl
    Returns (label, rows) where label is used in the output summary's dataset_paths.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency 'datasets'. Install it to load datasets from HF.") from e

    task = str(task).strip().lower()
    split = str(split).strip().lower()
    data_file = _hf_data_file(task, split)
    label = f"hf:{repo_id}/{data_file}"

    ds = load_dataset(repo_id, data_files={split: data_file}, split=split)
    rows: List[Dict] = []
    for i, row in enumerate(ds):
        if max_eval and max_eval > 0 and i >= max_eval:
            break
        # ensure plain dict (datasets returns dict-like objects)
        rows.append(dict(row))
    return label, rows


def evaluate_loaded_datasets(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    datasets: List[Tuple[str, List[Dict]]],
    output_prefix: str,
    max_eval: int = 0,
    base_model: str = "",
    adapter_dir: str = "",
) -> None:
    """
    Same as evaluate_datasets(), but takes already-loaded rows.
    """
    ensure_dir_for_prefix(output_prefix)

    dataset_records: List[Dict] = []
    total_oom = 0

    for i, (dataset_label, rows_in) in enumerate(datasets):
        print(f"\nEvaluating on dataset {i + 1}: {dataset_label}")
        rows = rows_in[: max_eval] if (max_eval and max_eval > 0) else rows_in

        mcq_items, skipped_non_mcq = build_mcq_items(rows)
        indices = list(range(len(mcq_items)))

        results_by_idx, oom_indices = evaluate_mcq_indices(
            model=model,
            tokenizer=tokenizer,
            mcq_items=mcq_items,
            indices=indices,
            desc=f"Dataset {i + 1}",
        )
        total_oom += len(oom_indices)

        dataset_records.append(
            {
                "dataset_index": i,
                "dataset_path": dataset_label,
                "mcq_items": mcq_items,
                "results_by_idx": results_by_idx,
                "skipped_non_mcq": skipped_non_mcq,
                "oom_indices": oom_indices,
            }
        )

    if total_oom > 0:
        if not torch.cuda.is_available():
            print("CUDA OOM detected but CUDA is unavailable; skipping 8-bit retry.")
        else:
            print(f"\nRetrying {total_oom} skipped questions with 8-bit model...")
            del model
            clear_cuda_cache()
            model_8bit, tokenizer = load_model_and_tokenizer(
                base_model=base_model,
                adapter_dir=adapter_dir,
                load_in_8bit=True,
            )
            for record in dataset_records:
                if not record["oom_indices"]:
                    continue
                retry_results, retry_oom = evaluate_mcq_indices(
                    model=model_8bit,
                    tokenizer=tokenizer,
                    mcq_items=record["mcq_items"],
                    indices=record["oom_indices"],
                    desc=f"Dataset {record['dataset_index'] + 1} (8-bit retry)",
                )
                record["results_by_idx"].update(retry_results)
                record["oom_indices"] = retry_oom
            remaining_oom = sum(len(r["oom_indices"]) for r in dataset_records)
            if remaining_oom:
                print(f"Warning: {remaining_oom} questions still failed after 8-bit retry.")

    all_accuracies: List[float] = []
    all_correct = 0
    all_total = 0
    total_skipped_oom = 0
    overall_topic_totals: Dict[str, int] = {}
    overall_topic_corrects: Dict[str, int] = {}

    dataset_paths: List[str] = []

    for record in dataset_records:
        dataset_label = record["dataset_path"]
        dataset_paths.append(dataset_label)

        mcq_items = record["mcq_items"]
        results_by_idx = record["results_by_idx"]
        skipped_non_mcq = record["skipped_non_mcq"]
        skipped_oom = len(record["oom_indices"])
        total_skipped_oom += skipped_oom

        correct = 0
        total = 0
        detailed_results: List[Dict] = []

        label_totals: Dict[str, int] = {}
        label_corrects: Dict[str, int] = {}
        topic_totals: Dict[str, int] = {}
        topic_corrects: Dict[str, int] = {}

        for idx, item in enumerate(mcq_items):
            result = results_by_idx.get(idx)
            if result is None:
                detailed_results.append(build_skipped_oom_entry(item, idx))
                continue

            detailed_results.append(result)
            total += 1
            if result["is_correct"]:
                correct += 1

            label = result.get("label", "")
            if label:
                label_totals[label] = label_totals.get(label, 0) + 1
                if result["is_correct"]:
                    label_corrects[label] = label_corrects.get(label, 0) + 1
            topic = result.get("topic", "")
            if topic:
                topic_totals[topic] = topic_totals.get(topic, 0) + 1
                if result["is_correct"]:
                    topic_corrects[topic] = topic_corrects.get(topic, 0) + 1

        acc = correct / max(1, total)
        all_accuracies.append(acc)
        all_correct += correct
        all_total += total

        for k, v in topic_totals.items():
            overall_topic_totals[k] = overall_topic_totals.get(k, 0) + v
        for k, v in topic_corrects.items():
            overall_topic_corrects[k] = overall_topic_corrects.get(k, 0) + v

        label_accuracy: Dict[str, float] = {}
        for k, v in label_totals.items():
            c = label_corrects.get(k, 0)
            label_accuracy[k] = c / max(1, v)
        topic_accuracy: Dict[str, float] = {}
        for k, v in topic_totals.items():
            c = topic_corrects.get(k, 0)
            topic_accuracy[k] = c / max(1, v)

        results_data = {
            "summary": {
                "accuracy": acc,
                "total": total,
                "correct": correct,
                "skipped_non_mcq": skipped_non_mcq,
                "skipped_oom": skipped_oom,
                "dataset_path": dataset_label,
                "label_totals": label_totals,
                "label_corrects": label_corrects,
                "label_accuracy": label_accuracy,
                "topic_totals": topic_totals,
                "topic_corrects": topic_corrects,
                "topic_accuracy": topic_accuracy,
            },
            "questions": detailed_results,
        }

        output_file = f"{output_prefix}_dataset_{record['dataset_index'] + 1}.json"
        with open(output_file, "w", encoding="utf-8") as f_out:
            json.dump(results_data, f_out, indent=2)

        print(f"Dataset {record['dataset_index'] + 1} results saved to {output_file}")
        print(f"Dataset {record['dataset_index'] + 1} accuracy: {acc:.4f} ({correct}/{total})")

    total_acc = all_correct / max(1, all_total)
    overall_topic_accuracy: Dict[str, float] = {}
    for k, v in overall_topic_totals.items():
        c = overall_topic_corrects.get(k, 0)
        overall_topic_accuracy[k] = c / max(1, v)
    overall_summary = {
        "total_accuracy": total_acc,
        "total_correct": all_correct,
        "total_questions": all_total,
        "dataset_accuracies": all_accuracies,
        "dataset_paths": dataset_paths,
        "topic_totals": overall_topic_totals,
        "topic_corrects": overall_topic_corrects,
        "topic_accuracy": overall_topic_accuracy,
        "total_skipped_oom": total_skipped_oom,
    }

    overall_file = f"{output_prefix}_summary.json"
    with open(overall_file, "w", encoding="utf-8") as f_sum:
        json.dump(overall_summary, f_sum, indent=2)

    print(f"\nOverall summary saved to {overall_file}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, required=True, help="Base model id or path")
    p.add_argument("--adapter", type=str, help="Path to LoRA adapter directory")
    p.add_argument(
        "--dataset_paths",
        type=str,
        nargs="+",
        default=None,
        help="Paths to one or more transition MCQ JSONL files (use this OR --dataset_repo_id/--task/--split).",
    )
    p.add_argument("--dataset_repo_id", type=str, default=None, help="HF dataset repo id, e.g. ORG/nesys-world-model-benchmark")
    p.add_argument("--task", type=str, default=None, help="Task name in HF dataset repo: plancraft|scienceworld|webshop")
    p.add_argument("--split", type=str, default=None, help="Split name in HF dataset repo: dev|test")
    p.add_argument("--max_eval", type=int, default=0, help="Optional cap on number of rows")
    p.add_argument(
        "--output_prefix",
        type=str,
        default="transition_mcq_eval_results",
        help="Prefix for output files (directory will be created if needed)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter)

    if args.dataset_repo_id and args.task and args.split:
        label, rows = load_rows_from_hf_dataset(
            repo_id=args.dataset_repo_id,
            task=args.task,
            split=args.split,
            max_eval=args.max_eval,
        )
        evaluate_loaded_datasets(
            model=model,
            tokenizer=tokenizer,
            datasets=[(label, rows)],
            output_prefix=args.output_prefix,
            max_eval=args.max_eval,
            base_model=args.base_model,
            adapter_dir=args.adapter or "",
        )
        return

    if not args.dataset_paths:
        raise SystemExit(
            "Must provide either --dataset_paths (local JSONL) OR (--dataset_repo_id and --task and --split)."
        )

    evaluate_datasets(
        model=model,
        tokenizer=tokenizer,
        dataset_paths=args.dataset_paths,
        output_prefix=args.output_prefix,
        max_eval=args.max_eval,
        base_model=args.base_model,
        adapter_dir=args.adapter,
    )


if __name__ == "__main__":
    main()


