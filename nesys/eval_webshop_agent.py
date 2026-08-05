#!/usr/bin/env python3
"""
Evaluate a WebShop agent across four NeSyS integration points:

  - naive          : pure agent baseline, no rules, no WM
  - rule_filter    : enumerate clickable candidates, drop those whose
                     symbolic-rule "fail" score is positive (no WM).
  - wm_rerank      : rule_filter + WM rerank, expose top-K (action,
                     predicted next state) to the agent as planning context.
  - wm_controller  : same scoring as wm_rerank, but execute the argmax
                     action directly (no second agent call).

This script reuses the NeSyS rule files, rule loaders, and the transition
model interface from `create_transition_mcq_rules.py` /
`eval_transition_mcq_logprob.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WEBSHOP_ROOT = Path(
    os.environ.get("WEBSHOP_ROOT", str(REPO_ROOT / "webshop"))
).expanduser().resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(WEBSHOP_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBSHOP_ROOT))

from eval_transition_mcq_logprob import (  # noqa: E402
    get_model_input_device,
    load_model_and_tokenizer,
    score_choice_logprob_from_messages,
)


WEBSHOP_TRANSITION_SYSTEM = (
    "You are a transition model for the WebShop text environment.\n"
    "Given the current page text (state) and the action taken, predict the "
    "next page text.\nYour answer must be ONLY the next state's text, with "
    "no extra commentary."
)


@dataclass
class WebshopCandidate:
    action: str
    predicted_state: str
    transition_logprob: float
    rule_score: float
    combined_score: float


def load_rule_weights_from_json(weights_path: str) -> List[float]:
    """Load either a JSON list or ``{"weights": [...]}``."""
    with open(weights_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("weights")
    if not isinstance(data, list):
        raise ValueError("Rule weights must be a list or a dict containing 'weights'.")
    return [float(weight) for weight in data]


def load_rules_from_file(file_path: str) -> List[str]:
    """Load repeated ``rule_reward`` programs from a released rule file."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    blocks = re.split(r"\n# Rule \d+\s*\n", content)
    return [
        block.strip()
        for block in blocks
        if block.strip().startswith("def ")
        or block.strip().startswith("# Task group:")
    ]


# -------- argument parsing ---------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent_model",
        type=str,
        default="Jianwen/Webshop-7B-SFT",
        help="HF model id/path for the acting agent.",
    )
    parser.add_argument(
        "--transition_base_model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Base model id for the neural transition model.",
    )
    parser.add_argument(
        "--transition_adapter",
        type=str,
        default="cindermond/world-model-webshop-llama3-2-1b-instruct-filtered",
        help="Optional LoRA adapter for the transition model.",
    )
    parser.add_argument(
        "--rule_file",
        type=str,
        default=str(
            SCRIPT_DIR
            / "final_rules"
            / "rules_webshop_transition_qa_sft_llama3-2-1b-instruct_final.py"
        ),
        help="Python rule file containing repeated `rule_reward` functions.",
    )
    parser.add_argument(
        "--rule_weights_json",
        type=str,
        default="",
        help="Optional JSON file with per-rule weights.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["naive", "rule_filter", "wm_rerank", "wm_controller"],
        choices=["naive", "rule_filter", "wm_rerank", "wm_controller"],
        help="Evaluation modes to run sequentially.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results_webshop_agent",
        help="Output directory for per-mode JSONL trajectories and summary.",
    )
    parser.add_argument(
        "--num_sessions",
        type=int,
        default=100,
        help="Number of WebShop sessions to evaluate.",
    )
    parser.add_argument(
        "--session_offset",
        type=int,
        default=0,
        help="Starting session index (sessions are deterministic per seed).",
    )
    parser.add_argument(
        "--max_steps", type=int, default=15, help="Max env steps per episode."
    )
    parser.add_argument(
        "--top_k_candidates",
        type=int,
        default=10,
        help="Max clickable candidates to score per step.",
    )
    parser.add_argument(
        "--top_k_states_context",
        type=int,
        default=3,
        help="Top-K (action, predicted state) tuples to show the agent in advisor mode.",
    )
    parser.add_argument(
        "--rule_filter_fail_threshold",
        type=float,
        default=0.0,
        help=(
            "Candidates whose aggregated rule fail-score is STRICTLY GREATER "
            "than this are dropped before agent selection."
        ),
    )
    parser.add_argument(
        "--rule_scale",
        type=float,
        default=1.0,
        help="Scale factor for rule_score in the WM combined ranking.",
    )
    parser.add_argument("--agent_temperature", type=float, default=0.2)
    parser.add_argument("--agent_max_new_tokens", type=int, default=128)
    parser.add_argument("--transition_temperature", type=float, default=0.8)
    parser.add_argument("--transition_top_p", type=float, default=0.95)
    parser.add_argument("--transition_max_new_tokens", type=int, default=256)
    parser.add_argument("--webshop_observation_mode", type=str, default="text")
    parser.add_argument(
        "--use_spacy_shim",
        action="store_true",
        help=(
            "Install a lightweight spaCy shim before importing WebShop. This is "
            "only a fallback for broken spaCy/pydantic environments; by default "
            "the script uses the real installed spaCy package."
        ),
    )
    parser.add_argument(
        "--num_products",
        type=int,
        default=0,
        help=(
            "Pass through to the WebShop SimServer to cap product set "
            "(0 = use the default subset configured by webshop)."
        ),
    )
    parser.add_argument(
        "--cache_dir", type=str, default="", help="Shared HF cache directory."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing per-mode JSONL files.",
    )
    parser.add_argument(
        "--max_history_chars",
        type=int,
        default=4000,
        help="Max characters of observation history kept in the agent prompt.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for any local sampling."
    )
    return parser.parse_args()


# -------- rule / model loading ----------------------------------------------


def compile_rule_functions(rule_file: str) -> List[Callable[[str, str, str], float]]:
    compiled: List[Callable[[str, str, str], float]] = []
    for rule_code in load_rules_from_file(rule_file):
        local_env: Dict[str, Any] = {}
        try:
            exec(rule_code, local_env)
            func = local_env.get("rule_reward")
        except Exception:
            func = None
        if callable(func):
            compiled.append(func)
    return compiled


def aggregate_rule_score(
    state_text: str,
    action_text: str,
    predicted_state: str,
    rule_funcs: Sequence[Callable[[str, str, str], float]],
    rule_weights: Sequence[float],
) -> float:
    total = 0.0
    for weight, func in zip(rule_weights, rule_funcs):
        try:
            total += float(weight) * float(func(state_text, action_text, predicted_state))
        except Exception:
            continue
    return total


def set_optional_cache_dir(cache_dir: str) -> None:
    if not cache_dir:
        return
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)


def load_agent_model_and_tokenizer(
    model_name: str, cache_dir: str = ""
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, cache_dir=cache_dir or None
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir or None,
    )
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    return model, tokenizer


# -------- WebShop env --------------------------------------------------------


def install_spacy_shim() -> None:
    """
    WebShop imports spaCy only to split product titles and keep noun-like tokens
    for the reward's type match. Some cluster envs have a spaCy/pydantic combo
    that fails during import before WebShop can even start. This small shim
    provides the tiny subset of spaCy used by `web_agent_site.engine.goal`:
    `spacy.load(...)` returning a callable whose tokens have `text` and `pos_`.
    """

    class _SimpleToken:
        def __init__(self, text: str):
            self.text = text
            self.pos_ = "NOUN"

    class _SimpleNlp:
        def __call__(self, text: str) -> List[_SimpleToken]:
            return [_SimpleToken(tok) for tok in re.findall(r"[A-Za-z0-9]+", text or "")]

    def _load(_: str) -> _SimpleNlp:
        return _SimpleNlp()

    sys.modules["spacy"] = types.SimpleNamespace(load=_load)


def make_env(args: argparse.Namespace):
    if not (WEBSHOP_ROOT / "web_agent_site").is_dir():
        raise FileNotFoundError(
            "WebShop was not found at "
            f"{WEBSHOP_ROOT}. Clone the WebShop repository there or set "
            "WEBSHOP_ROOT to the directory containing web_agent_site/."
        )
    if args.use_spacy_shim:
        install_spacy_shim()
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

    kwargs: Dict[str, Any] = {
        "observation_mode": args.webshop_observation_mode,
    }
    if args.num_products > 0:
        kwargs["num_products"] = args.num_products
    return WebAgentTextEnv(**kwargs)


def env_observation_text(env) -> str:
    """Return the textual observation (page text) for the current state."""
    if hasattr(env, "browser") and hasattr(env, "convert_html_to_text"):
        try:
            return env.convert_html_to_text(env.browser.page_source)
        except Exception:
            pass
    obs = getattr(env, "observation", "")
    return obs if isinstance(obs, str) else str(obs)


def get_current_goal(env) -> Dict[str, Any]:
    try:
        return env.server.user_sessions[env.session]["goal"]
    except Exception:
        return {}


def build_search_candidates(env) -> List[str]:
    """
    Match the original WebShop baseline action space on the search page:
    search by the product query, by each attribute plus query, and by the
    full instruction. Free-form LLM search queries are usually too verbose
    for WebShop's Lucene index and often return zero products.
    """
    goal = get_current_goal(env)
    query = str(goal.get("query", "")).strip().lower()
    instruction = str(goal.get("instruction_text", "")).strip().lower()
    attributes = goal.get("attributes", []) or []

    texts: List[str] = []
    if query:
        # Prefer WebShop's exact-query operator. The repo's Lucene indexes are
        # often absent/empty on clusters; `search[<q> ...]` bypasses Lucene and
        # returns products whose product['query'] matches the goal query.
        texts.append(f"<q> {query}")
        texts.append(query)
        texts.extend(f"{str(att).strip().lower()} {query}" for att in attributes if str(att).strip())
    if instruction:
        texts.append(instruction)

    candidates: List[str] = []
    seen = set()
    for text in texts:
        text = re.sub(r"\s+", " ", text).strip()
        if not text or text in seen:
            continue
        candidates.append(f"search[{text}]")
        seen.add(text)
    return candidates


def list_action_candidates(env, include_search: bool) -> List[str]:
    """Return valid action candidates for the current WebShop page."""
    avail = env.get_available_actions()
    candidates: List[str] = []
    for name in avail.get("clickables", []) or []:
        if not name:
            continue
        candidates.append(f"click[{name}]")
    if include_search and avail.get("has_search_bar", False):
        candidates.extend(build_search_candidates(env))
    # de-dup while preserving order
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            unique.append(c)
            seen.add(c)
    return unique


# -------- agent prompting ----------------------------------------------------


SYSTEM_AGENT_PROMPT = (
    "You are a WebShop shopping agent. Your goal is to buy a product that "
    "satisfies the user's instruction. At every step, choose exactly one "
    "admissible action from the provided action list.\n\n"
    "WebShop strategy:\n"
    "- On the search page, use the provided search action.\n"
    "- On a results page, click a product that best matches the instruction; "
    "do not loop on Next/Prev unless the current page has no plausible product.\n"
    "- On a product page, select required options such as color, size, style, "
    "or pack count when they are available.\n"
    "- After selecting required options, click[buy now].\n"
    "- Prefer valid listed actions exactly as written.\n\n"
    "Respond in this exact format:\n"
    "<reasoning>brief reason for the selected action</reasoning>\n"
    "<action>one admissible action, e.g. click[buy now]</action>"
)


def truncate_history(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def build_agent_messages(
    instruction: str,
    observation: str,
    history: List[Dict[str, str]],
    candidates: Optional[Sequence[str]] = None,
    advisor_block: Optional[str] = None,
    max_history_chars: int = 4000,
) -> List[Dict[str, str]]:
    """Construct the chat messages for the agent."""
    msgs: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_AGENT_PROMPT},
    ]
    if history:
        joined = "\n".join(
            f"[step {i}] action={item['action']} -> reward={item.get('reward', 0)}"
            for i, item in enumerate(history)
        )
        joined = truncate_history(joined, max_history_chars)
        msgs.append({"role": "user", "content": f"Past actions so far:\n{joined}"})
        msgs.append({"role": "assistant", "content": "ok"})

    parts = [f"Instruction: {instruction}", "", "Current page:", observation]
    if candidates:
        parts += ["", "Admissible actions (choose exactly one):"]
        parts += [f" - {c}" for c in candidates]
    if advisor_block:
        parts += ["", advisor_block]
    parts += [
        "",
        "Return exactly one <reasoning> tag and one <action> tag. The action must match one admissible action exactly.",
    ]
    msgs.append({"role": "user", "content": "\n".join(parts)})
    return msgs


def sanitize_generation(text: str) -> str:
    cleaned = text.replace("<|eot_id|>", "").strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()
    return cleaned


def render_chat_prompt_fallback(messages: List[Dict[str, str]]) -> str:
    """Render messages in a Qwen/ChatML-compatible format if no chat template exists."""
    chunks: List[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    chunks.append("<|im_start|>assistant\n")
    return "\n".join(chunks)


def build_generation_inputs(
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
) -> Dict[str, torch.Tensor]:
    if getattr(tokenizer, "chat_template", None):
        try:
            prompt_inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if isinstance(prompt_inputs, torch.Tensor):
                return {
                    "input_ids": prompt_inputs,
                    "attention_mask": torch.ones_like(prompt_inputs),
                }
        except ValueError:
            pass

    prompt = render_chat_prompt_fallback(messages)
    encoded = tokenizer(prompt, return_tensors="pt")
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])),
    }


def generate_chat_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
) -> str:
    inputs = build_generation_inputs(tokenizer, messages)
    device = get_model_input_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    eos_token_ids = [tokenizer.eos_token_id]
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is not None:
            eos_token_ids.append(eot_id)
    except Exception:
        pass
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=list({tid for tid in eos_token_ids if tid is not None}),
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    return sanitize_generation(text)


# -------- WM transition utilities -------------------------------------------


def build_webshop_transition_messages(state_text: str, action_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": WEBSHOP_TRANSITION_SYSTEM},
        {
            "role": "user",
            "content": (
                "Current state (page text):\n"
                f"{state_text}\n\n"
                "Action taken:\n"
                f"{action_text}\n\n"
                "Question: What is the next state's page text? "
                "Answer with ONLY the next state's text."
            ),
        },
    ]


def generate_transition_next_state(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    state_text: str,
    action_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    messages = build_webshop_transition_messages(state_text, action_text)
    prompt_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    device = get_model_input_device(model)
    prompt_inputs = prompt_inputs.to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=prompt_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(outputs[0][prompt_inputs.shape[-1] :], skip_special_tokens=False)
    return text.replace("<|eot_id|>", "").strip()


def score_transition_logprob(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    state_text: str,
    action_text: str,
    predicted_state: str,
) -> float:
    messages = build_webshop_transition_messages(state_text, action_text)
    try:
        return score_choice_logprob_from_messages(model, tokenizer, messages, predicted_state)
    except Exception:
        return float("-inf")


# -------- rule-only filtering ------------------------------------------------


_FAIL_PROBE_CHOICES = ("fail", "Fail", "FAIL")


def fail_score(
    state_text: str,
    action_text: str,
    rule_funcs: Sequence[Callable[[str, str, str], float]],
    rule_weights: Sequence[float],
) -> float:
    """
    Aggregate rule_reward(state, action, choice="fail") -- if positive, the
    rules predict the action will fail. Used by the rule_filter mode (and as
    a first-stage prune in wm_rerank / wm_controller).
    """
    best = float("-inf")
    for choice in _FAIL_PROBE_CHOICES:
        score = aggregate_rule_score(state_text, action_text, choice, rule_funcs, rule_weights)
        if score > best:
            best = score
    return best if best != float("-inf") else 0.0


def filter_candidates_by_rules(
    state_text: str,
    candidates: Sequence[str],
    rule_funcs: Sequence[Callable[[str, str, str], float]],
    rule_weights: Sequence[float],
    threshold: float,
) -> List[str]:
    kept: List[str] = []
    for action_text in candidates:
        if fail_score(state_text, action_text, rule_funcs, rule_weights) > threshold:
            continue
        kept.append(action_text)
    return kept


# -------- WM rerank ---------------------------------------------------------


def rerank_with_wm(
    state_text: str,
    candidates: Sequence[str],
    transition_model: AutoModelForCausalLM,
    transition_tokenizer: AutoTokenizer,
    rule_funcs: Sequence[Callable[[str, str, str], float]],
    rule_weights: Sequence[float],
    rule_scale: float,
    args: argparse.Namespace,
) -> List[WebshopCandidate]:
    ranked: List[WebshopCandidate] = []
    for action_text in candidates:
        try:
            predicted_state = generate_transition_next_state(
                transition_model,
                transition_tokenizer,
                state_text,
                action_text,
                max_new_tokens=args.transition_max_new_tokens,
                temperature=args.transition_temperature,
                top_p=args.transition_top_p,
            )
        except Exception:
            continue
        logprob = score_transition_logprob(
            transition_model,
            transition_tokenizer,
            state_text,
            action_text,
            predicted_state,
        )
        rule_score = aggregate_rule_score(
            state_text, action_text, predicted_state, rule_funcs, rule_weights
        )
        combined = logprob + rule_scale * rule_score
        ranked.append(
            WebshopCandidate(
                action=action_text,
                predicted_state=predicted_state,
                transition_logprob=logprob,
                rule_score=rule_score,
                combined_score=combined,
            )
        )
    ranked.sort(key=lambda x: x.combined_score, reverse=True)
    return ranked


def advisor_block(candidates: Sequence[WebshopCandidate]) -> str:
    if not candidates:
        return ""
    lines = ["Predicted next states (top candidates from world model):"]
    for i, c in enumerate(candidates, start=1):
        snippet = c.predicted_state[:300].replace("\n", " ")
        lines.append(
            f"  {i}. action={c.action}  "
            f"(logprob={c.transition_logprob:.2f}, rule={c.rule_score:.2f})"
        )
        lines.append(f"     predicted: {snippet}")
    return "\n".join(lines)


# -------- evaluation loop ----------------------------------------------------


def evaluate_mode(
    mode: str,
    args: argparse.Namespace,
    env,
    agent_model: AutoModelForCausalLM,
    agent_tokenizer: AutoTokenizer,
    transition_model: Optional[AutoModelForCausalLM],
    transition_tokenizer: Optional[AutoTokenizer],
    rule_funcs: Sequence[Callable[[str, str, str], float]],
    rule_weights: Sequence[float],
    progress: Optional[tqdm],
    jsonl_path: Path,
    completed_session_ids: set,
) -> List[Dict[str, Any]]:
    sessions = list(
        range(args.session_offset, args.session_offset + args.num_sessions)
    )
    results: List[Dict[str, Any]] = []

    for session_id in sessions:
        if session_id in completed_session_ids:
            continue
        try:
            env.reset(session=session_id)
        except Exception as exc:
            results.append(
                {
                    "mode": mode,
                    "session_id": session_id,
                    "error": f"reset failed: {exc}",
                    "reward": 0.0,
                    "done": False,
                }
            )
            continue

        instruction = env.instruction_text
        observation = env_observation_text(env)
        steps: List[Dict[str, Any]] = []
        done = False
        reward = 0.0
        info = None
        history: List[Dict[str, Any]] = []

        for step_idx in range(args.max_steps):
            candidates_full = list_action_candidates(env, include_search=True)
            search_candidates = [c for c in candidates_full if c.startswith("search[")]
            clickable_candidates = [c for c in candidates_full if c.startswith("click[")]

            # Rules/WM operate on click transitions. On the initial search page,
            # keep the original WebShop baseline's guided search candidates.
            filtered_candidates: List[str] = (
                search_candidates if search_candidates else clickable_candidates
            )
            mpc_candidates: List[WebshopCandidate] = []
            advisor_text = ""

            if mode in {"rule_filter", "wm_rerank", "wm_controller"} and clickable_candidates:
                filtered_candidates = filter_candidates_by_rules(
                    observation,
                    clickable_candidates,
                    rule_funcs,
                    rule_weights,
                    args.rule_filter_fail_threshold,
                )
                filtered_candidates = filtered_candidates[: args.top_k_candidates]

            if mode in {"wm_rerank", "wm_controller"} and clickable_candidates:
                if transition_model is None or transition_tokenizer is None:
                    raise RuntimeError(
                        f"Mode {mode} requires a transition model"
                    )
                mpc_candidates = rerank_with_wm(
                    observation,
                    filtered_candidates,
                    transition_model,
                    transition_tokenizer,
                    rule_funcs,
                    rule_weights,
                    args.rule_scale,
                    args,
                )
                if mpc_candidates:
                    advisor_text = advisor_block(mpc_candidates[: args.top_k_states_context])

            agent_action: Optional[str] = None
            if mode == "wm_controller" and mpc_candidates:
                agent_action = mpc_candidates[0].action
                raw_response = agent_action
            else:
                show_candidates: Optional[List[str]] = candidates_full[: args.top_k_candidates]
                if mode in {"rule_filter", "wm_rerank"}:
                    show_candidates = (
                        filtered_candidates
                        if mode == "rule_filter"
                        else (
                            [c.action for c in mpc_candidates[: args.top_k_states_context]]
                            if mpc_candidates
                            else filtered_candidates
                        )
                    )
                messages = build_agent_messages(
                    instruction=instruction,
                    observation=observation,
                    history=history,
                    candidates=show_candidates,
                    advisor_block=advisor_text or None,
                    max_history_chars=args.max_history_chars,
                )
                raw_response = generate_chat_response(
                    agent_model,
                    agent_tokenizer,
                    messages,
                    args.agent_max_new_tokens,
                    args.agent_temperature,
                )
                agent_action = extract_action(raw_response)

            # On the initial search page, use the first guided search candidate
            # deterministically. It is `search[<q> goal_query]`, which bypasses
            # Lucene and stays usable even when local WebShop indexes are empty.
            # This matches the original baseline's closed search action space
            # while avoiding accidental free-form queries from the LLM.
            if search_candidates:
                agent_action = search_candidates[0]

            if agent_action is None:
                steps.append(
                    {
                        "step_idx": step_idx,
                        "observation_excerpt": observation[:300],
                        "raw_response": raw_response,
                        "action": None,
                        "filtered_candidates": filtered_candidates,
                        "mpc_candidates": [c.__dict__ for c in mpc_candidates[: args.top_k_states_context]],
                        "reward": reward,
                        "done": done,
                    }
                )
                continue

            if agent_action.startswith("think:"):
                history.append({"action": agent_action, "reward": 0.0})
                steps.append(
                    {
                        "step_idx": step_idx,
                        "observation_excerpt": observation[:300],
                        "raw_response": raw_response,
                        "action": agent_action,
                        "filtered_candidates": filtered_candidates,
                        "mpc_candidates": [c.__dict__ for c in mpc_candidates[: args.top_k_states_context]],
                        "reward": reward,
                        "done": done,
                    }
                )
                continue

            try:
                _, step_reward, done, info = env.step(agent_action)
            except Exception as exc:
                step_reward = 0.0
                done = True
                info = {"error": str(exc)}
            reward = float(step_reward) if step_reward is not None else reward
            history.append({"action": agent_action, "reward": float(step_reward or 0.0)})

            steps.append(
                {
                    "step_idx": step_idx,
                    "observation_excerpt": observation[:300],
                    "raw_response": raw_response,
                    "action": agent_action,
                    "filtered_candidates": filtered_candidates,
                    "mpc_candidates": [c.__dict__ for c in mpc_candidates[: args.top_k_states_context]],
                    "reward": float(step_reward or 0.0),
                    "done": done,
                }
            )

            if done:
                break
            observation = env_observation_text(env)

        result = {
            "mode": mode,
            "session_id": session_id,
            "instruction": instruction,
            "num_steps": len(steps),
            "trajectory": steps,
            "reward": float(reward),
            "done": bool(done),
            "info": info,
        }
        results.append(result)
        append_result(jsonl_path, result)
        if progress is not None:
            progress.update(1)
            progress.set_postfix(
                avg_reward=f"{sum(r['reward'] for r in results)/max(1,len(results)):.3f}",
                solved=f"{sum(1 for r in results if r['reward']>=1.0)}/{len(results)}",
            )

    return results


def extract_action(raw_response: str) -> Optional[str]:
    """
    Pull an action string out of the agent's free-form response.
    Prefer SkillRL-style `<action>...</action>`, then fall back to the first
    `search[..]`, `click[..]`, or `think: ...` token group.
    """
    text = raw_response.strip()
    if not text:
        return None
    tag_match = re.search(r"<action>\s*(.*?)\s*</action>", text, flags=re.I | re.S)
    if tag_match:
        text = tag_match.group(1).strip()
    # Bracketed actions first.
    for prefix in ("search", "click"):
        m = re.search(prefix + r"\[(.*?)\]", text, flags=re.S)
        if m:
            return f"{prefix}[{m.group(1).strip()}]"
    if text.lower().startswith("think:"):
        return "think: " + text.split(":", 1)[1].strip()
    return None


# -------- summary ------------------------------------------------------------


def summarize_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    avg_reward = sum(item["reward"] for item in results) / max(1, total)
    solved = sum(1 for item in results if item["reward"] >= 1.0)
    avg_steps = sum(item["num_steps"] for item in results) / max(1, total)
    return {
        "total_examples": total,
        "average_reward": avg_reward,
        "success_rate": solved / max(1, total),
        "average_steps": avg_steps,
        "successes": solved,
    }


# -------- IO -----------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_existing_results(jsonl_path: Path) -> List[Dict[str, Any]]:
    if not jsonl_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_result(jsonl_path: Path, result: Dict[str, Any]) -> None:
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


# -------- main ---------------------------------------------------------------


def main() -> None:
    args = parse_args()
    set_optional_cache_dir(args.cache_dir)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    needs_transition_model = any(m in {"wm_rerank", "wm_controller"} for m in args.modes)
    needs_rules = any(m in {"rule_filter", "wm_rerank", "wm_controller"} for m in args.modes)

    print(f"[setup] loading agent model: {args.agent_model}")
    agent_model, agent_tokenizer = load_agent_model_and_tokenizer(
        args.agent_model, cache_dir=args.cache_dir
    )

    transition_model = None
    transition_tokenizer = None
    if needs_transition_model:
        print(
            f"[setup] loading transition model: base={args.transition_base_model}"
            f" adapter={args.transition_adapter}"
        )
        transition_model, transition_tokenizer = load_model_and_tokenizer(
            args.transition_base_model, args.transition_adapter
        )

    rule_funcs: List[Callable[[str, str, str], float]] = []
    rule_weights: List[float] = []
    if needs_rules:
        rule_funcs = compile_rule_functions(args.rule_file)
        if not rule_funcs:
            raise RuntimeError(f"No rule functions loaded from {args.rule_file}")
        if args.rule_weights_json:
            rule_weights = load_rule_weights_from_json(args.rule_weights_json)
            if len(rule_weights) != len(rule_funcs):
                raise ValueError(
                    f"Loaded {len(rule_weights)} weights for {len(rule_funcs)} rules from {args.rule_weights_json}"
                )
        else:
            rule_weights = [1.0] * len(rule_funcs)

    print("[setup] launching WebShop SimServer (this loads product data once)")
    env = make_env(args)

    overall_summary: Dict[str, Any] = {
        "agent_model": args.agent_model,
        "transition_base_model": args.transition_base_model if needs_transition_model else None,
        "transition_adapter": args.transition_adapter if needs_transition_model else None,
        "rule_file": args.rule_file if needs_rules else None,
        "num_sessions": args.num_sessions,
        "session_offset": args.session_offset,
        "max_steps": args.max_steps,
        "modes": {},
    }

    for mode in args.modes:
        jsonl_path = output_dir / f"{mode}.jsonl"
        if not args.resume and jsonl_path.exists():
            jsonl_path.unlink()
        existing = load_existing_results(jsonl_path) if args.resume else []
        completed = {item["session_id"] for item in existing if "session_id" in item}

        progress = tqdm(
            total=args.num_sessions,
            initial=len(existing),
            desc=f"{mode}",
        )
        new_results = evaluate_mode(
            mode=mode,
            args=args,
            env=env,
            agent_model=agent_model,
            agent_tokenizer=agent_tokenizer,
            transition_model=transition_model,
            transition_tokenizer=transition_tokenizer,
            rule_funcs=rule_funcs,
            rule_weights=rule_weights,
            progress=progress,
            jsonl_path=jsonl_path,
            completed_session_ids=completed,
        )
        progress.close()

        all_results = list(existing) + new_results
        overall_summary["modes"][mode] = {
            "trajectory_file": str(jsonl_path),
            "summary": summarize_results(all_results),
        }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)
    print(json.dumps(overall_summary, indent=2))


if __name__ == "__main__":
    main()
