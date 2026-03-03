try:
    import openai  # type: ignore
except Exception:
    openai = None  # type: ignore
import json
import os
import sys
import argparse
import random
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import OPTICS
from sentence_transformers import SentenceTransformer
import umap
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Global switch so CREATE/IMPROVE helpers (which don't thread flags through every call)
# can still respect the CLI option.
_GLOBAL_SCALE_RULES_BY_SPREAD: bool = False


def _rule_scale_from_logprobs(logprobs: np.ndarray, scale_by_spread: bool) -> float:
    """
    If scale_by_spread is True, scale rule outputs by (logprob_spread + 1.0)
    where spread = max(logprobs) - min(logprobs).
    """
    if not scale_by_spread:
        return 1.0
    try:
        spread = float(np.max(logprobs) - np.min(logprobs))
        return spread + 1.0
    except Exception:
        return 1.0

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# Initialize preprocessing tools (used by CREATE/IMPROVE modes). Evaluation-only
# usage should not require NLTK downloads.
lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words("english"))
except Exception:
    stop_words = set()

_OPENAI_CLIENT = None


def _get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT
    if openai is None:
        raise RuntimeError(
            "OpenAI client not available. Install the 'openai' package and set OPENAI_API_KEY "
            "to use CREATE/IMPROVE modes."
        )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Set it to use CREATE/IMPROVE modes.")
    _OPENAI_CLIENT = openai.OpenAI(api_key=api_key)
    return _OPENAI_CLIENT

prompt_template_case = """
State: {state_content}
Action: {action_content}
Correct choice: {correct_choice_content}
Model's choice: {wrong_choice_content}
"""

prompt_template = """
Analyze the following model error cases and summarize one actionable improvement rule. Follow these guidelines:

[Error Cases]
{cases}

[Analysis Requirements]
1. Try to find patterns in the questions, what do they have in common? Is the action of the same type? Is the current state has some similarities?
2. Try to find patterns in the correct answers, what are the shared characteristics of the correct answers? 
3. Try to find patterns in the incorrect answers, what makes them incorrect?
4. Formulate one generalizable rule for the presented error cases, the rule should be detailed enough to be programmed. It will be used to score each candidate choice. The rule should only be used when the shared patterns of these questions are observed, and should encourage the patterns we see in the correct answers, and disencourage the patterns of the incorrect answers.
5. If a detailed rule cannot be found for all the presented error cases, describe a rule for fewer error cases instead of saying vague things.
6. Phrase the rule after "### Rule ### "
7. Write a Python program after "### Program ### ", it should be a function named "rule_reward" with "state", "action" and "choice" as input, and output a float number between -1 to 1, indicating how likely the choice is correct (positive) or wrong(negative). The state parameter is the last state, the action parameter is the action, and choice is one of the answer choices (predicted next state).

[Example Rule and Programs]
1. Example rule: For a move action "move: from [I?] to [A/B/C?] with quantity q", the next state should reflect only the intended move: the moved item name (from the source slot in Current state) must increase by q at the destination slot and decrease by q at the source slot. Penalize choices that change counts of unrelated items.
Example Program: ```python
def rule_reward(state, action, choice):
    import re

    def parse_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, 0
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns list of (name, slot, qty) with slot like [I17], [A1], [0]
        items = []
        for name, slot, qty in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            items.append((name.strip(), f'[{slot}]', int(qty)))
        return items

    src_slot, dst_slot, qty = parse_action(action)
    if not src_slot:
        return 0.0

    q_items = parse_items(state)
    c_items = parse_items(choice)

    slot_to_item_q = {slot: (name, count) for (name, slot, count) in q_items}
    slot_to_item_c = {slot: (name, count) for (name, slot, count) in c_items}

    moved = slot_to_item_q.get(src_slot)
    if moved is None:
        return 0.0
    moved_name, src_prev = moved
    dst_prev = slot_to_item_q.get(dst_slot, (None, 0))[1]
    src_new_name, src_new = slot_to_item_c.get(src_slot, (moved_name, 0))
    dst_new_name, dst_new = slot_to_item_c.get(dst_slot, (moved_name, 0))

    checks = 0
    # destination quantity increased by q for same item
    if dst_new_name == moved_name and (dst_new - dst_prev) == qty:
        checks += 1
    # source quantity decreased by q for same item (or source disappears when reaches 0)
    if src_new_name in (moved_name, None) and (src_prev - src_new) == qty:
        checks += 1
    # penalize unrelated item total-count changes (ignore moved item and output slot [0])
    def totals(items):
        d = {}
        for name, slot, count in items:
            if name == moved_name:
                continue
            if slot == '[0]':
                continue
            d[name] = d.get(name, 0) + count
        return d

    tq, tc = totals(q_items), totals(c_items)
    if all(tq.get(k, 0) == tc.get(k, 0) for k in set(tq) | set(tc)):
        checks += 1

    # modest bonus for not changing unrelated slots
    return checks / 3.0  # maps to [0, 1]
```

2. Example rule: For illegal actions other than move or smelt, the next state should not change.
Example Program: ```python
def rule_reward(state, action, choice):
    import re

    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, 0
        return m.group(1), m.group(2), int(m.group(3))

    def parse_smelt_action(a):
        m = re.search(r'smelt:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, 0
        return m.group(1), m.group(2), int(m.group(3))        
        
    def parse_items(s):
        # returns list of (name, slot, qty) with slot like [I17], [A1], [0]
        items = []
        for name, slot, qty in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            items.append((name.strip(), f'[{slot}]', int(qty)))
        return items

    src_slot, dst_slot, qty = parse_move_action(action)
    if src_slot:
        return 0.0
    src_slot, dst_slot, qty = parse_smelt_action(action)
    if src_slot:
        return 0.0

    q_items = parse_items(state)
    c_items = parse_items(choice)

    if q_items == c_items:
        return 1.0
    else:
        return -1.0
```
"""

prompt_template_sw = """
Analyze the following ScienceWorld model error cases and summarize one actionable improvement rule. Follow these guidelines:

[Error Cases]
{cases}

[ScienceWorld Format]
- The 'state' text is a compressed history including Task Description, optional Memory, a per-step History with 'action', 'observation', 'reward', and 'inventory' or 'inventory_diff' snippets, and ends with 'current_step_action: <action>'.
- The 'choice' is the model's continuation with EXACTLY these lines (order fixed):
  predicted_observation: <text>
  predicted_reward: <float in [-1,1]>
  predicted_inventory_diff: <zero or more +/- lines>

[Analysis Requirements]
1. Identify common action types (e.g., 'open X', 'go to Y', 'pick up Z', 'use thermometer in inventory on W', 'move A to B', 'activate X').
2. Infer what a consistent predicted_observation and predicted_inventory_diff should look like for those actions.
3. Design a rule that checks for these consistencies by parsing the choice lines and, if needed, extracting the current step action from the state (line starting with 'current_step_action:').
4. The rule should be specific (applies only when it matches the action pattern) and returns a score in [-1, 1].
5. Phrase the rule after "### Rule ### "
6. Write the Python program after "### Program ### " defining 'rule_reward(state, action, choice)' returning a float in [-1, 1].

[Example Rules and Programs]
1) Example rule (Pick up): If action starts with 'pick up <obj>', then:
   - predicted_observation contains 'You move the <obj> to the inventory.'
   - predicted_inventory_diff contains a line starting with '+ ' and mentioning <obj>.
Example Program: ```python
def rule_reward(state, action, choice):
    import re
    # Extract action if not provided
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''
    # Parse choice into fields
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)\n', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)
    if not (obs_m and rew_m and diff_m is not None):
        return -0.5
    obs = obs_m.group(1).strip().lower()
    inv = diff_m.group(1)
    # Match 'pick up <obj>'
    m = re.match(r'(?i)\s*pick up\s+(.+)$', action.strip())
    if not m:
        return 0.0
    obj = m.group(1).strip().lower()
    score = 0.0
    if f'move the {obj} to the inventory' in obs:
        score += 0.5
    # Check a '+ ' line referencing the object
    has_plus = any(ln.strip().startswith('+ ') and (obj.split()[0] in ln.lower()) for ln in inv.splitlines())
    if has_plus:
        score += 0.5
    return max(-1.0, min(1.0, score))
```

2) Example rule (Open): If action matches 'open <obj>', require:
   - predicted_observation contains 'The <obj> is now open.' (or 'already open')
   - predicted_inventory_diff is empty or contains no changes
Example Program: ```python
def rule_reward(state, action, choice):
    import re
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)\n', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)
    if not (obs_m and diff_m is not None):
        return -0.5
    obs = obs_m.group(1).strip().lower()
    inv = diff_m.group(1)
    m = re.match(r'(?i)\s*open\s+(.+)$', action.strip())
    if not m:
        return 0.0
    obj = m.group(1).strip().lower()
    ok_obs = (f'the {obj} is now open' in obs) or ('already open' in obs)
    inv_lines = [ln.strip() for ln in inv.splitlines() if ln.strip()]
    no_change = len(inv_lines) == 0
    return 1.0 if ok_obs and no_change else -0.2
```

3) Example rule (Thermometer): If action contains 'use thermometer' and 'on <target>', then:
   - predicted_observation contains 'the thermometer measures a temperature of'
   - predicted_inventory_diff should be empty
Example Program: ```python
def rule_reward(state, action, choice):
    import re
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)\n', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)
    if not (obs_m and diff_m is not None):
        return -0.5
    obs = obs_m.group(1).strip().lower()
    inv = diff_m.group(1)
    if not re.search(r'(?i)use\s+thermometer', action) or not re.search(r'(?i)\bon\b', action):
        return 0.0
    inv_lines = [ln.strip() for ln in inv.splitlines() if ln.strip()]
    ok_obs = 'the thermometer measures a temperature of' in obs
    return 1.0 if ok_obs and len(inv_lines) == 0 else -0.2
```
"""

prompt_template_webshop = """
Analyze the following WebShop transition model error cases and summarize one actionable improvement rule. Follow these guidelines:

[Error Cases]
{cases}

[WebShop Format]
- The 'state' text is the current WebShop page text. It often contains lines like:
  - Instruction:
  - [button] ... [button_]
- The 'action' is the environment action taken, typically in forms like:
  - click[buy now]
  - click[< prev]
  - click[description] / click[features] / click[reviews]
  - search[...]
- The 'choice' is one candidate next-state. For most steps it's the next page text; for terminal steps it may be exactly one token:
  - Success
  - Fail

[Analysis Requirements]
1. Identify the shared action type(s) across the cases (e.g., clicking a button that exists/doesn't exist).
2. Infer what a consistent next page should look like, or when the correct result should be the terminal token 'Fail'/'Success'.
3. Formulate one generalizable and checkable rule that returns a score in [-1, 1].
4. The rule should be specific: only apply when its conditions match; otherwise return 0.0.
5. Phrase the rule after "### Rule ### "
6. Write the Python program after "### Program ### " defining 'rule_reward(state, action, choice)' returning a float in [-1, 1].

[Example Rules and Programs]
1) Example rule (Buy Now missing => Fail): If action is exactly click[buy now] but the current page text does NOT contain a Buy Now button,
   then the correct terminal result is Fail. Return 1 if choice is Fail and -1 otherwise.
Example Program: ```python
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = norm(action)
    if act != "click[buy now]":
        return 0.0

    # Accept both "[button] Buy Now [button_]" and minor formatting variants.
    has_buy_now = re.search(r"(?i)\\[button\\]\\s*buy now\\s*\\[button_\\]", st) is not None
    if has_buy_now:
        return 0.0

    return 1.0 if norm(choice) == "fail" else -1.0
```

2) Example rule (< Prev => back to results): If action is click[< prev] and the current page text contains a < Prev button,
   then the next page is usually a search results list page that contains 'Total results' and a 'Next >' button.
Example Program: ```python
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = norm(action)
    if act != "click[< prev]":
        return 0.0

    has_prev = re.search(r"(?i)\\[button\\]\\s*<\\s*prev\\s*\\[button_\\]", st) is not None
    if not has_prev:
        return 0.0

    ch = choice or ""
    looks_results = ("Total results" in ch) and (re.search(r"(?i)\\[button\\]\\s*next\\s*>\\s*\\[button_\\]", ch) is not None)
    return 1.0 if looks_results else -0.5
```
"""

def extract_python_programs(text):
    """
    Extracts Python code blocks from text containing Markdown-style code fences.
    
    Args:
        text (str): Input text containing Python code blocks wrapped in ```python...```
        
    Returns:
        list: List of Python programs as strings
    """
    pattern = r'```python\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Clean up leading/trailing whitespace and empty lines
    programs = []
    for code in matches:
        cleaned = '\n'.join([line.rstrip() for line in code.split('\n')])
        programs.append(cleaned)
        
    return programs

def load_evaluation_results(eval_results_path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(eval_results_path, 'r') as f:
        return json.load(f)

def load_rule_weights_from_json(weights_path: str) -> List[float]:
    """
    Load per-rule weights from a JSON file.

    Supported formats:
      1) {"weights": [1.0, -0.5, ...]}  (recommended / default output of this script)
      2) [1.0, -0.5, ...]
    """
    with open(weights_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("weights", None)
    if not isinstance(data, list):
        raise ValueError("Weights JSON must be a list or a dict containing key 'weights'.")
    weights: List[float] = []
    for i, w in enumerate(data):
        try:
            weights.append(float(w))
        except Exception as e:
            raise ValueError(f"Weight at index {i} is not a number: {w}") from e
    return weights

def load_rules_from_file(file_path: str) -> List[str]:
    """Loads a list of rule function strings from a Python file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Rules are expected to be separated by a comment like '# Rule X' (exact match, nothing after the number)
    # Use word boundary and end of line to avoid matching '# Rule 1: description'
    rule_blocks = re.split(r'\n# Rule \d+\s*\n', content)
    
    rules = [block.strip() for block in rule_blocks if block.strip().startswith('def ') or block.strip().startswith('# Task group:')]
    return rules

def load_multi_dataset_results(eval_summary_path: str) -> Dict[str, Any]:
    """Load multi-dataset evaluation results from summary file and individual dataset files."""
    with open(eval_summary_path, 'r') as f:
        summary = json.load(f)

    # Load individual dataset results
    dataset_results = {}
    base_path = eval_summary_path.replace('_summary.json', '')

    for i, dataset_path in enumerate(summary['dataset_paths']):
        dataset_file = f"{base_path}_dataset_{i+1}.json"
        with open(dataset_file, 'r') as f:
            dataset_results[i] = json.load(f)

    return {
        'summary': summary,
        'dataset_results': dataset_results
    }

def _extract_state_and_action_from_question(q: Dict[str, Any]) -> (str, str):
    """
    Normalize state/action for both PlanCraft and ScienceWorld style questions.
    - PlanCraft: use 'input_state' and 'input_action'
    - ScienceWorld: use 'input_user' as state; extract 'current_step_action:' line; fall back to 'input_action'
    """
    def _try_parse_webshop_user(user_text: str) -> (str, str):
        """
        Parse WebShop evaluator 'input_user' text into (state_page_text, action_text).
        Expected structure (from build_webshop_transition_qa.py):
          Current state (page text):
          ...
          Action taken:
          ...
          Question: ...
        """
        ut = user_text or ""
        if "Current state (page text):" not in ut or "Action taken:" not in ut:
            return "", ""
        try:
            # Extract state between markers.
            _, rest = ut.split("Current state (page text):", 1)
            state_part, rest2 = rest.split("Action taken:", 1)
            # Extract action until Question: (if present).
            action_part = rest2
            if "Question:" in rest2:
                action_part, _ = rest2.split("Question:", 1)
            return state_part.strip(), action_part.strip()
        except Exception:
            return "", ""

    state = q.get("input_state") or ""
    action = q.get("input_action") or ""
    if state and action:
        return state, action
    # ScienceWorld fallback
    user = q.get("input_user") or ""
    st = user or state
    act = action
    if not act and user:
        try:
            m = re.search(r'(?mi)^current_step_action:\s*(.+)$', user)
            if m:
                act = m.group(1).strip()
        except Exception:
            act = action or ""
    # WebShop fallback: parse action/state from input_user formatting
    if user:
        ws_state, ws_action = _try_parse_webshop_user(user)
        if ws_state:
            st = ws_state
        if not act and ws_action:
            act = ws_action
    return st, act

def extract_wrong_answers(eval_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract questions that the model answered incorrectly."""
    wrong_answers = []
    questions = eval_results["questions"] if "questions" in eval_results else eval_results["dataset_results"][0]["questions"]

    for question_result in questions:
        if not question_result["is_correct"]:
            state_text, action_text = _extract_state_and_action_from_question(question_result)
            wrong_data = {
                "state": state_text,
                "action": action_text,
                "choices": question_result["choices"],
                "correct_choice": question_result["correct_idx"],
                "wrong_choice": question_result["predicted_idx"],
                "choice_logprobs": question_result["choice_logprobs"],
                "label": question_result.get("label"),
                "topic": question_result.get("topic"),
                "task_name": question_result.get("task_name"),
            }
            wrong_answers.append(wrong_data)
    return wrong_answers

def extract_multi_dataset_wrong_answers(multi_results: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    """Extract wrong answers from each dataset in multi-dataset results."""
    wrong_answers_per_dataset = {}

    for dataset_idx, dataset_result in multi_results['dataset_results'].items():
        wrong_answers = []
        for question_result in dataset_result["questions"]:
            if not question_result["is_correct"]:
                state_text, action_text = _extract_state_and_action_from_question(question_result)
                wrong_data = {
                    "state": state_text,
                    "action": action_text,
                    "choices": question_result["choices"],
                    "correct_choice": question_result["correct_idx"],
                    "wrong_choice": question_result["predicted_idx"],
                    "choice_logprobs": question_result["choice_logprobs"],
                    "label": question_result.get("label"),
                    "topic": question_result.get("topic"),
                    "task_name": question_result.get("task_name"),
                }
                wrong_answers.append(wrong_data)
        wrong_answers_per_dataset[dataset_idx] = wrong_answers

    return wrong_answers_per_dataset

def advanced_clean(text):
    """Enhanced text cleaning pipeline."""
    text = re.sub(r'\d+', '[NUM]', text)  # Normalize numbers
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words
    cleaned: List[str] = []
    for w in text.split():
        if len(w) <= 2:
            continue
        if w.lower() in stop_words:
            continue
        try:
            cleaned.append(lemmatizer.lemmatize(w))
        except Exception:
            cleaned.append(w)
    return " ".join(cleaned)

def tfidf_feature_extraction(texts):
    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1,3),
        analyzer='word',
        min_df=2
    )
    return vectorizer.fit_transform(texts).toarray()

def semantic_feature_extraction(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(texts)

def dimension_reduction(features):
    # Two-stage dimensionality reduction
    svd = TruncatedSVD(n_components=50)
    reduced = svd.fit_transform(features)
    return umap.UMAP(n_components=5).fit_transform(reduced)

def adaptive_clustering(data):
    optics = OPTICS(
        min_samples=3,
        xi=0.05,
        min_cluster_size=0.1
    ).fit(data)
    return optics.labels_

def extract_newly_appearing_items(question, correct_answer):
    """
    Extract features about inventory changes for clustering.
    - For PlanCraft (Minecraft): items that newly appear in the correct answer vs question inventory.
    - For ScienceWorld: parse predicted_inventory_diff lines from the correct answer.
    
    Args:
        question (str): The question text containing initial inventory
        correct_answer (str): The correct answer choice containing the updated inventory
        
    Returns:
        str: A text representation for clustering
    """
    # ScienceWorld detection: presence of predicted_observation/predicted_inventory_diff headers
    if ("predicted_observation:" in correct_answer) or ("predicted_inventory_diff" in correct_answer):
        difflines = []
        try:
            body = correct_answer.split("predicted_inventory_diff", 1)[1]
            if body.startswith(":"):
                body = body[1:]
            for ln in body.splitlines():
                t = ln.strip()
                if not t:
                    continue
                if t.startswith("+ ") or t.startswith("- "):
                    difflines.append(t)
        except Exception:
            difflines = []
        return " ".join(difflines) if difflines else "no_inv_diff"
    def parse_inventory(text):
        """Parse inventory from text format into a dict of slot -> (item_name, quantity)"""
        inventory = {}
        # Regex to find items, their slots, and quantities
        pattern = r'-\s+([^\[\n]+?)\s+\[([A-Z0-9]+)\]\s+quantity\s+(\d+)'
        matches = re.findall(pattern, text)
        for name, slot, qty in matches:
            inventory[f'[{slot}]'] = (name.strip(), int(qty))
        return inventory
    
    question_inventory = parse_inventory(question)
    correct_inventory = parse_inventory(correct_answer)
    
    # Find items that exist in correct answer but not in question
    newly_appearing = []
    
    # Get all item names from question inventory for comparison
    question_items = set()
    for slot, (item_name, qty) in question_inventory.items():
        question_items.add(item_name)
    
    # Check each item in correct answer
    for slot, (item_name, qty) in correct_inventory.items():
        if item_name not in question_items:
            # This item newly appears in the correct answer
            newly_appearing.append(f"{item_name} in {slot}")
    
    # Also check for items that appear in new slots (moved items)
    for slot, (item_name, qty) in correct_inventory.items():
        if slot not in question_inventory:
            # This slot is new - item appeared in a new location
            newly_appearing.append(f"{item_name} new_in {slot}")
    
    # Return a string representation for clustering
    if newly_appearing:
        return " ".join(newly_appearing)
    else:
        return "no_new_items"

def dynamic_clustering_analysis(all_data, cluster_key="question"):
    """Cluster the wrong answers based on different aspects."""
    
    if cluster_key == "question":
        all_keys = [item['state'] for item in all_data]
    elif cluster_key == "correct_answer":
        all_keys = [item['choices'][item['correct_choice']] for item in all_data]
    elif cluster_key == "wrong_answer":
        all_keys = [item['choices'][item['wrong_choice']] for item in all_data]
    elif cluster_key == "label":
        all_keys = [item['label'] for item in all_data]
    elif cluster_key == "topic":
        all_keys = [item.get('topic', 'UNTOPICED') for item in all_data]
    elif cluster_key in ("task", "task_name"):
        all_keys = [item.get('task_name', 'UNTASKED') for item in all_data]
    elif cluster_key == "action":
        all_keys = [item['action'] for item in all_data]
    elif cluster_key == "inventory_diff":
        all_keys = [extract_newly_appearing_items(item['state'], item['choices'][item['correct_choice']]) for item in all_data]
    else:
        raise ValueError(f"Unknown cluster_key: {cluster_key}")
    
    processed_questions = [advanced_clean(q) for q in all_keys]
    
    # Feature extraction
    tfidf_matrix = tfidf_feature_extraction(processed_questions)
    semantic_embeddings = semantic_feature_extraction(processed_questions)
    
    # Feature fusion and dimensionality reduction
    combined_features = np.hstack([tfidf_matrix, semantic_embeddings])
    reduced_features = dimension_reduction(combined_features)
    
    # Adaptive clustering
    cluster_labels = adaptive_clustering(reduced_features)
    
    return cluster_labels, processed_questions

def analyze_clusters(cluster_labels, texts):
    """Analyze clusters and return valid cluster IDs."""
    cluster_ids = []
    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:  # Skip noise points
            continue
        cluster_size = np.sum(cluster_labels == cluster_id)
        if cluster_size >= 2:  # Only consider clusters with at least 2 samples
            cluster_ids.append(cluster_id)
    return cluster_ids

def hierarchical_clustering_indices(all_data, primary_key: str, secondary_key: str):
    """Two-stage clustering: first by primary_key, then within each cluster by secondary_key.

    Returns a list of lists, where each inner list contains absolute indices
    (w.r.t. all_data) that belong to a valid secondary cluster (size >= 2)
    inside a valid primary cluster.
    """
    try:
        primary_labels, _ = dynamic_clustering_analysis(all_data, primary_key)
    except Exception:
        return []

    # Identify valid primary clusters
    primary_cluster_ids = analyze_clusters(primary_labels, all_data)
    nested_cluster_indices = []

    for primary_id in primary_cluster_ids:
        primary_member_indices = [i for i, lab in enumerate(primary_labels) if lab == primary_id]
        if len(primary_member_indices) < 2:
            continue
        subset = [all_data[i] for i in primary_member_indices]

        try:
            secondary_labels, _ = dynamic_clustering_analysis(subset, secondary_key)
        except Exception:
            continue

        secondary_cluster_ids = analyze_clusters(secondary_labels, subset)
        for secondary_id in secondary_cluster_ids:
            rel_indices = [j for j, lab in enumerate(secondary_labels) if lab == secondary_id]
            abs_indices = [primary_member_indices[j] for j in rel_indices]
            if len(abs_indices) >= 2:
                nested_cluster_indices.append(abs_indices)

    return nested_cluster_indices

# Default clustering strategies
# - Strings indicate single-level clustering by that key
# - 2-tuples indicate hierarchical clustering: first by primary, then by secondary
DEFAULT_CREATE_CLUSTER_STRATEGIES = [
    "question",
    "correct_answer",
    "wrong_answer",
    "action",
    "label",
    ("label", "question"),
    ("label", "correct_answer"),
    ("label", "wrong_answer"),
    ("label", "action"),
    ("action", "question")
]

DEFAULT_IMPROVE_CLUSTER_STRATEGIES = [
    "question",
    "correct_answer",
    "wrong_answer",
    "action",
    "label",
    "inventory_diff",
    #"topic",
    #"task_name",
    # Example hierarchical strategies (uncomment or edit as needed):
    # ("action", "inventory_diff"),
]

def build_prompt(samples):
    """Build prompt for GPT-4 analysis."""
    cases = []
    for sample in samples:
        case = prompt_template_case.format(
            state_content=sample['state'],
            action_content=sample['action'],
            correct_choice_content=sample['choices'][sample['correct_choice']],
            wrong_choice_content=sample['choices'][sample['wrong_choice']]
        )
        cases.append(case)
    cases_text = "\n".join(cases)
    # Choose template (ScienceWorld vs WebShop vs PlanCraft) and escape braces except {cases}
    looks_sw = any(
        ("predicted_observation:" in sample['choices'][sample['correct_choice']])
        or ("Task Description:" in sample['state'])
        for sample in samples
    )
    looks_webshop = any(
        (
            ("[button]" in (sample.get("state") or "")) and ("Instruction:" in (sample.get("state") or ""))
        )
        or ("click[" in (sample.get("action") or "").lower())
        or ("search[" in (sample.get("action") or "").lower())
        or any((c or "").strip() in ("Fail", "Success") for c in (sample.get("choices") or []))
        for sample in samples
    )
    if looks_sw:
        base_template = prompt_template_sw
    elif looks_webshop:
        base_template = prompt_template_webshop
    else:
        base_template = prompt_template
    safe_template = (
        base_template
        .replace("{", "{{")
        .replace("}", "}}")
        .replace("{{cases}}", "{cases}")
    )
    return safe_template.format(cases=cases_text)

def gpt4_analyze_cluster(samples):
    """Call OpenAI API to analyze a cluster of samples."""
    prompt = build_prompt(samples)
    client = _get_openai_client()
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],

    )
    return response.choices[0].message.content

def identify_negatively_impacted_questions(dev_eval_summary: str, baseline_rules: List[str], new_rules: List[str]) -> List[Dict[str, Any]]:
    """Identify questions that flipped from correct (under baseline_rules) to wrong (under new_rules)."""
    multi_results = load_multi_dataset_results(dev_eval_summary)
    negatively_impacted = []

    for _, dataset_result in multi_results['dataset_results'].items():
        for question_result in dataset_result["questions"]:
            state, action = _extract_state_and_action_from_question(question_result)
            choices = question_result["choices"]
            original_logprobs = question_result["choice_logprobs"]
            correct_idx = question_result["correct_idx"]

            # Compute baseline prediction (apply baseline_rules)
            baseline_modified = original_logprobs.copy()
            for rule_code in baseline_rules:
                try:
                    exec(rule_code, globals())
                    rule_func = globals()['rule_reward']
                    for i, choice in enumerate(choices):
                        baseline_modified[i] += rule_func(state, action, choice)
                except Exception:
                    continue
            baseline_pred_idx = int(np.argmax(baseline_modified))

            # Compute new prediction (apply new_rules)
            new_modified = original_logprobs.copy()
            for rule_code in new_rules:
                try:
                    exec(rule_code, globals())
                    rule_func = globals()['rule_reward']
                    for i, choice in enumerate(choices):
                        new_modified[i] += rule_func(state, action, choice)
                except Exception:
                    continue
            new_pred_idx = int(np.argmax(new_modified))

            if baseline_pred_idx == correct_idx and new_pred_idx != correct_idx:
                negatively_impacted.append({
                    "state": state,
                    "action": action,
                    "choices": choices,
                    "correct_choice": correct_idx,
                    "wrong_choice": new_pred_idx,
                    "choice_logprobs": new_modified
                })

    return negatively_impacted

def sample_negative_impact_cases(negatively_impacted: List[Dict[str, Any]], current_rule: str, num_samples: int = 3) -> List[Dict[str, Any]]:
    """Sample cases where the current rule had negative impact for refinement."""
    if len(negatively_impacted) <= num_samples:
        return negatively_impacted
    
    return random.sample(negatively_impacted, num_samples)

def build_refinement_prompt(negative_cases: List[Dict[str, Any]], current_rule: str, original_rule_response: str) -> str:
    """Build prompt for GPT-4 to refine a rule based on negative impact cases."""
    
    # Extract the rule description from the original response
    rule_description = "No rule description available"
    if '### Rule ###' in original_rule_response:
        rule_description = original_rule_response.split('### Rule ###')[1].split('### Program ###')[0].strip()
    
    cases_text = ""
    for i, case in enumerate(negative_cases):
        case_text = f"""
Case {i+1}:
State: {case['state']} \n\n Action: {case['action']}
Correct choice: {case['choices'][case['correct_choice']]}
Model's choice: {case['choices'][case['wrong_choice']]}
Issue: The rule incorrectly changed this from correct to wrong.
"""
        cases_text += case_text
    
    refinement_template = f"""
The following rule is causing negative impacts on some questions that were originally answered correctly. Please refine the rule to avoid these negative impacts while maintaining its beneficial effects.

[Current Rule Description]
{rule_description}

[Current Rule Program]
```python
{current_rule}
```

[Negative Impact Cases]
{cases_text}

[Refinement Requirements]
1. Analyze why the current rule is causing these originally correct answers to become wrong
2. Identify the specific conditions or patterns that are causing the negative impact
3. Refine the rule to be more precise and avoid these false positives
4. Maintain the beneficial effects the rule was designed to achieve
5. The refined rule should be more conservative to avoid breaking correct predictions
6. Write the refined rule after "### Rule ### "
7. Write the refined Python program after "### Program ### ", it should be a function named "rule_reward" with "state", "action" and "choice" as input, and output a float number between -1 to 1

[Focus Areas for Refinement]
- Add more specific conditions to prevent false positives
- Consider edge cases that might be incorrectly handled
- Make the rule more conservative in its judgments
- Add additional validation checks before applying penalties/rewards
"""
    
    return refinement_template

def gpt4_refine_rule(negative_cases: List[Dict[str, Any]], current_rule: str, original_rule_response: str) -> str:
    """Call OpenAI API to refine a rule based on negative impact cases."""
    prompt = build_refinement_prompt(negative_cases, current_rule, original_rule_response)
    client = _get_openai_client()
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def extract_remaining_wrong_answers(dev_eval_summary: str, existing_rules: List[str]) -> Dict[int, List[Dict[str, Any]]]:
    """Extract wrong answers that are still wrong after applying existing rules."""
    multi_results = load_multi_dataset_results(dev_eval_summary)
    remaining_wrong_answers_per_dataset = {}

    for dataset_idx, dataset_result in multi_results['dataset_results'].items():
        remaining_wrong_answers = []

        for question_result in dataset_result["questions"]:
            # Use robust state/action extraction (supports ScienceWorld-style questions)
            state, action = _extract_state_and_action_from_question(question_result)
            choices = question_result["choices"]
            original_logprobs = question_result["choice_logprobs"]
            correct_idx = question_result["correct_idx"]
            # Preserve metadata for clustering strategies like label/topic/task_name
            label = question_result.get("label")
            topic = question_result.get("topic")
            task_name = question_result.get("task_name")

            modified_logprobs = original_logprobs.copy()

            for rule_code in existing_rules:
                try:
                    exec(rule_code, globals())
                    rule_func = globals()['rule_reward']
                    for i, choice in enumerate(choices):
                        modified_logprobs[i] += rule_func(state, action, choice)
                except Exception:
                    continue

            new_pred_idx = int(np.argmax(modified_logprobs))
            if new_pred_idx != correct_idx:
                remaining_wrong_answers.append({
                    "state": state,
                    "action": action,
                    "choices": choices,
                    "correct_choice": correct_idx,
                    "wrong_choice": new_pred_idx,
                    "choice_logprobs": modified_logprobs,
                    "label": label,
                    "topic": topic,
                    "task_name": task_name,
                })

        remaining_wrong_answers_per_dataset[dataset_idx] = remaining_wrong_answers

    return remaining_wrong_answers_per_dataset

def identify_problematic_rules(rules: List[str], dev_eval_summary: str) -> List[int]:
    """Identify individual rules that are causing negative impacts on dev set."""
    problematic_indices = []
    
    # Baseline performance (no rules)
    baseline_results = evaluate_on_multi_dataset(dev_eval_summary, [])
    baseline_accuracy = baseline_results["total_accuracy"]
    
    eprint(f"Baseline accuracy (no rules): {baseline_accuracy:.2%}")
    
    # Test each rule individually
    for i, rule in enumerate(rules):
        try:
            single_rule_results = evaluate_on_multi_dataset(dev_eval_summary, [rule])
            single_rule_accuracy = single_rule_results["total_accuracy"]
            
            # If this rule alone performs worse than baseline, it's problematic
            if single_rule_accuracy < baseline_accuracy - 0.001:
                problematic_indices.append(i)
                eprint(f"Rule {i+1}: PROBLEMATIC - Accuracy: {single_rule_accuracy:.2%} (vs baseline {baseline_accuracy:.2%})")
            else:
                eprint(f"Rule {i+1}: OK - Accuracy: {single_rule_accuracy:.2%}")
                
        except Exception as e:
            eprint(f"Rule {i+1}: ERROR during evaluation - {e}")
            problematic_indices.append(i)  # Consider it problematic if it causes errors
    
    return problematic_indices

def improve_existing_rules(rules: List[str], dev_eval_summary: str, max_reflection_attempts: int = 3, max_new_rules: int = 10) -> List[str]:
    """Improve existing rules and learn new rules based on dev set performance.
    Uses strategies defined in DEFAULT_IMPROVE_CLUSTER_STRATEGIES for clustering.
    """
    improved_rules = rules.copy()
    
    # Step 1: Identify and fix problematic existing rules
    eprint("Step 1: Identifying and fixing problematic existing rules...")
    problematic_indices = identify_problematic_rules(rules, dev_eval_summary)
    
    if problematic_indices:
        eprint(f"Found {len(problematic_indices)} problematic rules to improve")
        
        # Try to improve each problematic rule
        for rule_idx in problematic_indices:
            eprint(f"\nImproving rule {rule_idx + 1}...")
            
            current_rule = rules[rule_idx]
            
            # Test this rule alone to identify negative impact cases
            baseline_results = evaluate_on_multi_dataset(dev_eval_summary, [])
            rule_results = evaluate_on_multi_dataset(dev_eval_summary, [current_rule])
            
            # Find questions where this rule made things worse (true flips)
            negatively_impacted = identify_negatively_impacted_questions(
                dev_eval_summary,
                baseline_rules=[],
                new_rules=[current_rule]
            )
            
            if len(negatively_impacted) == 0:
                eprint(f"  No negatively impacted questions found for rule {rule_idx + 1}")
                continue
                
            eprint(f"  Found {len(negatively_impacted)} negatively impacted questions")
            
            # Try to improve the rule through reflection
            best_rule = current_rule
            best_accuracy = rule_results["total_accuracy"]
            
            for reflection_attempt in range(max_reflection_attempts):
                eprint(f"  Improvement attempt {reflection_attempt + 1}/{max_reflection_attempts}")
                
                try:
                    # Sample negative cases for refinement
                    negative_cases = sample_negative_impact_cases(negatively_impacted, best_rule)
                    
                    # Use GPT-4 to refine the rule (we need to create a fake rule response for this)
                    fake_rule_response = f"### Rule ###\nRule {rule_idx + 1}\n### Program ###\n```python\n{best_rule}\n```"
                    refined_rule_response = gpt4_refine_rule(negative_cases, best_rule, fake_rule_response)
                    refined_programs = extract_python_programs(refined_rule_response)
                    
                    if not refined_programs:
                        eprint(f"    Attempt {reflection_attempt + 1}: No valid refined program found")
                        continue
                        
                    refined_rule = refined_programs[0]
                    
                    # Test the refined rule
                    refined_results = evaluate_on_multi_dataset(dev_eval_summary, [refined_rule])
                    refined_accuracy = refined_results["total_accuracy"]
                    
                    eprint(f"    Attempt {reflection_attempt + 1}: Refined accuracy: {refined_accuracy:.2%}")
                    
                    # Check if refinement improved things
                    if refined_accuracy > best_accuracy:
                        best_rule = refined_rule
                        best_accuracy = refined_accuracy
                        eprint(f"    Attempt {reflection_attempt + 1}: Improvement found!")
                    else:
                        eprint(f"    Attempt {reflection_attempt + 1}: No improvement")
                        
                except Exception as e:
                    eprint(f"    Attempt {reflection_attempt + 1}: ERROR - {e}")
                    continue
            
            # Update the rule if we found an improvement
            if best_rule != current_rule:
                improved_rules[rule_idx] = best_rule
                eprint(f"  Rule {rule_idx + 1}: IMPROVED - New accuracy: {best_accuracy:.2%}")
            else:
                eprint(f"  Rule {rule_idx + 1}: NO IMPROVEMENT found")
    else:
        eprint("No problematic rules found. All existing rules appear to be working well individually.")
    
    # Step 2: Learn new rules for remaining wrong answers
    eprint(f"\nStep 2: Learning new rules for remaining wrong answers...")
    
    # Get current performance with improved rules
    current_results = evaluate_on_multi_dataset(dev_eval_summary, improved_rules)
    current_accuracy = current_results["total_accuracy"]
    eprint(f"Current accuracy with existing rules: {current_accuracy:.2%}")
    
    # Extract remaining wrong answers that current rules don't handle well
    remaining_wrong_answers = extract_remaining_wrong_answers(dev_eval_summary, improved_rules)
    
    if not remaining_wrong_answers:
        eprint("No remaining wrong answers found - all questions are correctly handled by existing rules.")
        return improved_rules
    
    total_remaining = sum(len(wrong_list) for wrong_list in remaining_wrong_answers.values())
    eprint(f"Found {total_remaining} remaining wrong answers across datasets")
    
    # Generate new rules using clustering on remaining wrong answers
    new_rules_generated = 0
    for strategy in DEFAULT_IMPROVE_CLUSTER_STRATEGIES:
        if new_rules_generated >= max_new_rules:
            break
        # Hierarchical strategy: (primary, secondary)
        if isinstance(strategy, tuple) and len(strategy) == 2:
            primary_key, secondary_key = strategy
            eprint(f"\nGenerating new rules by hierarchical clustering {primary_key} -> {secondary_key}...")
            for dataset_idx, wrong_answers in remaining_wrong_answers.items():
                if new_rules_generated >= max_new_rules:
                    break
                if len(wrong_answers) < 2:
                    continue
                eprint(f"  Processing dataset {dataset_idx+1} with {len(wrong_answers)} remaining wrong answers")
                nested_clusters = hierarchical_clustering_indices(wrong_answers, primary_key, secondary_key)
                eprint(f"  Found {len(nested_clusters)} secondary clusters in dataset {dataset_idx+1}")
                for sub_indices in nested_clusters:
                    if new_rules_generated >= max_new_rules:
                        break
                    cluster_samples = [wrong_answers[i] for i in sub_indices]
                    if len(cluster_samples) < 2:
                        continue
                    for attempt in range(3):
                        new_rules_generated += 1
                        selected_samples = random.sample(cluster_samples, min(3, len(cluster_samples)))
                        try:
                            rule_response = gpt4_analyze_cluster(selected_samples)
                            programs = extract_python_programs(rule_response)
                            if not programs:
                                eprint(f"  New rule {new_rules_generated}: No valid program found")
                                continue
                            program = programs[0]
                            test_rules = improved_rules + [program]
                            new_results = evaluate_on_multi_dataset(dev_eval_summary, test_rules)
                            new_accuracy = new_results["total_accuracy"]
                            if new_accuracy > current_accuracy + 0.001:
                                improved_rules.append(program)
                                current_accuracy = new_accuracy
                                eprint(f"  New rule {new_rules_generated}: ACCEPTED - New total accuracy: {current_accuracy:.2%}")
                                eprint(f"    Rule: {rule_response.split('### Rule ###')[1].split('### Program ###')[0].strip() if '### Rule ###' in rule_response else 'N/A'}")
                            else:
                                eprint(f"  New rule {new_rules_generated}: REJECTED - No improvement (accuracy: {new_accuracy:.2%})")
                        except Exception as e:
                            eprint(f"  New rule {new_rules_generated}: ERROR - {e}")
                            continue
                    if new_rules_generated >= max_new_rules:
                        break
            continue
        
        # Single-level strategy: string
        if isinstance(strategy, str):
            cluster_key = strategy
            eprint(f"\nGenerating new rules by clustering {cluster_key}...")
            for dataset_idx, wrong_answers in remaining_wrong_answers.items():
                if new_rules_generated >= max_new_rules:
                    break
                if len(wrong_answers) < 2:
                    continue
                eprint(f"  Processing dataset {dataset_idx+1} with {len(wrong_answers)} remaining wrong answers")
                try:
                    cluster_labels, processed_questions = dynamic_clustering_analysis(wrong_answers, cluster_key)
                except Exception as e:
                    eprint(f"  Skipping clustering {cluster_key} for dataset {dataset_idx+1} due to error: {e}")
                    continue
                cluster_ids = analyze_clusters(cluster_labels, processed_questions)
                eprint(f"  Found {len(cluster_ids)} valid clusters in dataset {dataset_idx+1}")
                for cluster_id in cluster_ids:
                    if new_rules_generated >= max_new_rules:
                        break
                    cluster_samples = [wrong_answers[i] for i in range(len(wrong_answers)) if cluster_labels[i] == cluster_id]
                    if len(cluster_samples) < 2:
                        continue
                    for attempt in range(3):
                        new_rules_generated += 1
                        selected_samples = random.sample(cluster_samples, min(3, len(cluster_samples)))
                        try:
                            rule_response = gpt4_analyze_cluster(selected_samples)
                            programs = extract_python_programs(rule_response)
                            if not programs:
                                eprint(f"  New rule {new_rules_generated}: No valid program found")
                                continue
                            program = programs[0]
                            test_rules = improved_rules + [program]
                            new_results = evaluate_on_multi_dataset(dev_eval_summary, test_rules)
                            new_accuracy = new_results["total_accuracy"]
                            if new_accuracy > current_accuracy + 0.001:
                                improved_rules.append(program)
                                current_accuracy = new_accuracy
                                eprint(f"  New rule {new_rules_generated}: ACCEPTED - New total accuracy: {current_accuracy:.2%}")
                                eprint(f"    Rule: {rule_response.split('### Rule ###')[1].split('### Program ###')[0].strip() if '### Rule ###' in rule_response else 'N/A'}")
                            else:
                                eprint(f"  New rule {new_rules_generated}: REJECTED - No improvement (accuracy: {new_accuracy:.2%})")
                        except Exception as e:
                            eprint(f"  New rule {new_rules_generated}: ERROR - {e}")
                            continue
                    if new_rules_generated >= max_new_rules:
                        break
    
    total_added = len(improved_rules) - len(rules)
    eprint(f"\nCompleted rule improvement: added {total_added} new rules")
    
    return improved_rules

def apply_rules_to_logprobs(wrong_answers: List[Dict], rules: List[str]) -> float:
    """Apply rules to modify logprobs and calculate new accuracy on wrong answers."""
    correct_predictions = 0
    total_predictions = len(wrong_answers)

    for sample in wrong_answers:
        state = sample['state']
        action = sample['action']
        choices = sample['choices']
        original_logprobs = sample['choice_logprobs']
        correct_idx = sample['correct_choice']

        # Apply rules to modify logprobs
        modified_logprobs = original_logprobs.copy()
        scale = _rule_scale_from_logprobs(np.array(modified_logprobs, dtype=float), _GLOBAL_SCALE_RULES_BY_SPREAD)

        for rule_code in rules:
            try:
                # Execute the rule function
                exec(rule_code, globals())
                rule_func = globals()['rule_reward']

                # Apply rule to each choice
                for i, choice in enumerate(choices):
                    rule_score = rule_func(state, action, choice)
                    modified_logprobs[i] += float(rule_score) * scale

            except Exception as e:
                eprint(f"Error applying rule: {e}")
                continue

        # Check if the rule improves prediction
        new_pred_idx = np.argmax(modified_logprobs)
        if new_pred_idx == correct_idx:
            correct_predictions += 1

    return correct_predictions / total_predictions if total_predictions > 0 else 0.0

def apply_rules_to_multi_dataset_logprobs(wrong_answers_per_dataset: Dict[int, List[Dict]], rules: List[str]) -> Dict[str, float]:
    """Apply rules to modify logprobs across multiple datasets and calculate accuracies."""
    total_correct = 0
    total_predictions = 0
    dataset_accuracies = {}

    for dataset_idx, wrong_answers in wrong_answers_per_dataset.items():
        correct_predictions = 0
        dataset_total = len(wrong_answers)

        for sample in wrong_answers:
            state = sample['state']
            action = sample['action']
            choices = sample['choices']
            original_logprobs = sample['choice_logprobs']
            correct_idx = sample['correct_choice']

            # Apply rules to modify logprobs
            modified_logprobs = original_logprobs.copy()
            scale = _rule_scale_from_logprobs(np.array(modified_logprobs, dtype=float), _GLOBAL_SCALE_RULES_BY_SPREAD)

            for rule_code in rules:
                try:
                    # Execute the rule function
                    exec(rule_code, globals())
                    rule_func = globals()['rule_reward']

                    # Apply rule to each choice
                    for i, choice in enumerate(choices):
                        rule_score = rule_func(state, action, choice)
                        modified_logprobs[i] += float(rule_score) * scale

                except Exception as e:
                    eprint(f"Error applying rule: {e}")
                    continue

            # Check if the rule improves prediction
            new_pred_idx = np.argmax(modified_logprobs)
            if new_pred_idx == correct_idx:
                correct_predictions += 1

        dataset_accuracy = correct_predictions / dataset_total if dataset_total > 0 else 0.0
        dataset_accuracies[dataset_idx] = dataset_accuracy

        total_correct += correct_predictions
        total_predictions += dataset_total

    overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0

    return {
        'dataset_accuracies': dataset_accuracies,
        'total_accuracy': overall_accuracy,
        'total_correct': total_correct,
        'total_questions': total_predictions
    }

def evaluate_on_full_dataset(eval_results_path: str, rules: List[str], scale_by_spread: bool = False) -> Dict[str, float]:
    """Evaluate rules on the full dataset."""
    eval_results = load_evaluation_results(eval_results_path)

    correct_predictions = 0
    total_predictions = len(eval_results["questions"])

    for question_result in eval_results["questions"]:
        state, action = _extract_state_and_action_from_question(question_result)
        choices = question_result["choices"]
        original_logprobs = question_result["choice_logprobs"]
        correct_idx = question_result["correct_idx"]

        # Apply rules to modify logprobs
        modified_logprobs = original_logprobs.copy()
        scale = _rule_scale_from_logprobs(np.array(modified_logprobs, dtype=float), bool(scale_by_spread or _GLOBAL_SCALE_RULES_BY_SPREAD))

        for rule_code in rules:
            try:
                # Execute the rule function
                exec(rule_code, globals())
                rule_func = globals()['rule_reward']

                # Apply rule to each choice
                for i, choice in enumerate(choices):
                    rule_score = rule_func(state, action, choice)
                    modified_logprobs[i] += float(rule_score) * scale

            except Exception as e:
                continue

        # Check prediction
        new_pred_idx = np.argmax(modified_logprobs)
        if new_pred_idx == correct_idx:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return {
        "accuracy": accuracy,
        "total": total_predictions,
        "correct": correct_predictions
    }

def evaluate_on_multi_dataset(eval_summary_path: str, rules: List[str], scale_by_spread: bool = False) -> Dict[str, Any]:
    """Evaluate rules on multiple datasets."""
    multi_results = load_multi_dataset_results(eval_summary_path)

    dataset_results = {}
    total_correct = 0
    total_predictions = 0
    # Aggregate per-label and per-topic stats across all datasets
    per_label_counts = {}
    per_topic_counts = {}

    for dataset_idx, dataset_result in multi_results['dataset_results'].items():
        correct_predictions = 0
        dataset_total = len(dataset_result["questions"])

        for question_result in dataset_result["questions"]:
            state, action = _extract_state_and_action_from_question(question_result)
            choices = question_result["choices"]
            original_logprobs = question_result["choice_logprobs"]
            correct_idx = question_result["correct_idx"]
            label = question_result.get("label", "UNLABELED")
            topic = question_result.get("topic", "UNTOPICED")

            # Apply rules to modify logprobs
            modified_logprobs = original_logprobs.copy()
            scale = _rule_scale_from_logprobs(np.array(modified_logprobs, dtype=float), bool(scale_by_spread or _GLOBAL_SCALE_RULES_BY_SPREAD))

            for rule_code in rules:
                try:
                    # Execute the rule function
                    exec(rule_code, globals())
                    rule_func = globals()['rule_reward']

                    # Apply rule to each choice
                    for i, choice in enumerate(choices):
                        rule_score = rule_func(state, action, choice)
                        modified_logprobs[i] += float(rule_score) * scale

                except Exception as e:
                    continue

            # Check prediction
            new_pred_idx = np.argmax(modified_logprobs)
            if new_pred_idx == correct_idx:
                correct_predictions += 1

            # Update per-label aggregates
            counts = per_label_counts.get(label)
            if counts is None:
                counts = {"correct": 0, "total": 0}
                per_label_counts[label] = counts
            counts["total"] += 1
            if new_pred_idx == correct_idx:
                counts["correct"] += 1
            # Update per-topic aggregates
            tc = per_topic_counts.get(topic)
            if tc is None:
                tc = {"correct": 0, "total": 0}
                per_topic_counts[topic] = tc
            tc["total"] += 1
            if new_pred_idx == correct_idx:
                tc["correct"] += 1

        dataset_accuracy = correct_predictions / dataset_total if dataset_total > 0 else 0.0
        dataset_results[dataset_idx] = {
            "accuracy": dataset_accuracy,
            "total": dataset_total,
            "correct": correct_predictions,
            "dataset_path": multi_results['summary']['dataset_paths'][dataset_idx]
        }

        total_correct += correct_predictions
        total_predictions += dataset_total

    overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0

    # Prepare per-label accuracy summary
    per_label_summary = {
        label: {
            "accuracy": (cnts["correct"] / cnts["total"]) if cnts["total"] > 0 else 0.0,
            "total": cnts["total"],
            "correct": cnts["correct"]
        }
        for label, cnts in per_label_counts.items()
    }
    per_topic_summary = {
        topic: {
            "accuracy": (cnts["correct"] / cnts["total"]) if cnts["total"] > 0 else 0.0,
            "total": cnts["total"],
            "correct": cnts["correct"]
        }
        for topic, cnts in per_topic_counts.items()
    }

    return {
        "dataset_results": dataset_results,
        "total_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_questions": total_predictions,
        "per_label": per_label_summary,
        "per_topic": per_topic_summary
    }

def prune_rules_by_removal(dev_eval_summary: str, rules: List[str], improvement_threshold: float = 0.0001) -> Dict[str, Any]:
    """
    Greedy backward elimination of rules on the dev set. Repeatedly removes a single rule
    if doing so increases accuracy on the dev set.

    Returns a dict with keys:
      - pruned_rules: List[str]
      - removed_rule_indices: List[int]  # 0-based indices in the original combined rule list
      - baseline_accuracy: float
      - final_accuracy: float
      - steps: List[Dict[str, Any]]  # each step has removed_index (original), new_accuracy
    """
    # Evaluate baseline with all rules
    try:
        baseline_results = evaluate_on_multi_dataset(dev_eval_summary, rules)
        current_accuracy = baseline_results["total_accuracy"]
    except Exception:
        return {
            "pruned_rules": rules,
            "removed_rule_indices": [],
            "baseline_accuracy": 0.0,
            "final_accuracy": 0.0,
            "steps": []
        }

    current_rules = rules.copy()
    index_map = list(range(len(rules)))  # maps current index -> original index
    removed_original_indices: List[int] = []
    steps: List[Dict[str, Any]] = []

    while True:
        best_accuracy = current_accuracy
        best_remove_idx = None

        for i in range(len(current_rules)):
            candidate_rules = current_rules[:i] + current_rules[i+1:]
            try:
                candidate_results = evaluate_on_multi_dataset(dev_eval_summary, candidate_rules)
                candidate_accuracy = candidate_results["total_accuracy"]
            except Exception:
                continue

            if candidate_accuracy > best_accuracy + improvement_threshold:
                best_accuracy = candidate_accuracy
                best_remove_idx = i

        if best_remove_idx is None:
            break

        removed_original_idx = index_map[best_remove_idx]
        removed_original_indices.append(removed_original_idx)
        steps.append({
            "removed_index": removed_original_idx,
            "new_accuracy": best_accuracy
        })

        # Apply removal
        current_rules.pop(best_remove_idx)
        index_map.pop(best_remove_idx)
        current_accuracy = best_accuracy

    return {
        "pruned_rules": current_rules,
        "removed_rule_indices": removed_original_indices,
        "baseline_accuracy": baseline_results["total_accuracy"],
        "final_accuracy": current_accuracy,
        "steps": steps
    }

def _compile_rule_functions(rules: List[str]) -> List[Any]:
    """Compile rule code strings into callable functions without polluting globals."""
    compiled_funcs: List[Any] = []
    for code in rules:
        try:
            local_env: Dict[str, Any] = {}
            exec(code, local_env)
            func = local_env.get('rule_reward', None)
            if callable(func):
                compiled_funcs.append(func)
            else:
                compiled_funcs.append(lambda state, action, choice: 0.0)
        except Exception:
            compiled_funcs.append(lambda state, action, choice: 0.0)
    return compiled_funcs

def _flatten_multi_results_for_samples(
    multi_results: Dict[str, Any],
    rule_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    Flatten multi-dataset results into a list of samples with required fields.
    Uses robust state/action extraction so ScienceWorld actions are available
    to rule functions during routing/weight optimization.
    """
    samples: List[Dict[str, Any]] = []
    for _, dataset_result in multi_results['dataset_results'].items():
        for q in dataset_result["questions"]:
            # Fall back to parsing current_step_action from input_user when input_action is empty
            state, action = _extract_state_and_action_from_question(q)
            logprobs = np.array(q["choice_logprobs"], dtype=float)
            if rule_only:
                logprobs = np.zeros_like(logprobs, dtype=float)
            samples.append({
                "state": state,
                "action": action,
                "choices": q["choices"],
                "logprobs": logprobs,
                "correct_idx": int(q["correct_idx"]),
            })
    return samples

def _precompute_rule_choice_scores(
    samples: List[Dict[str, Any]],
    rule_funcs: List[Any],
    scale_by_spread: bool = False,
) -> List[List[np.ndarray]]:
    """For each sample and each rule, precompute the per-choice score vector.

    If scale_by_spread is True, multiply each rule score by (logprob_spread + 1.0)
    to match the scaling used by training/create_scienceworld_task_rules.py.
    """
    precomputed: List[List[np.ndarray]] = []
    for s in samples:
        per_rule: List[np.ndarray] = []
        scale = 1.0
        if scale_by_spread:
            try:
                spread = float(np.max(s["logprobs"]) - np.min(s["logprobs"]))
                scale = spread + 1.0
            except Exception:
                scale = 1.0
        for func in rule_funcs:
            try:
                scores = [float(func(s["state"], s["action"], choice)) * scale for choice in s["choices"]]
            except Exception:
                scores = [0.0 for _ in s["choices"]]
            per_rule.append(np.array(scores, dtype=float))
        precomputed.append(per_rule)
    return precomputed

def _accuracy_from_weights(
    samples: List[Dict[str, Any]],
    precomputed: List[List[np.ndarray]],
    weights: List[float],
) -> float:
    correct = 0
    total = len(samples)
    for idx, s in enumerate(samples):
        combined = s["logprobs"].copy()
        per_rule = precomputed[idx]
        for j, w in enumerate(weights):
            if w == 0.0:
                continue
            combined = combined + w * per_rule[j]
        if int(np.argmax(combined)) == s["correct_idx"]:
            correct += 1
    return (correct / total) if total > 0 else 0.0

def optimize_rule_weights_on_dev(
    dev_eval_summary: str,
    rules: List[str],
    weight_grid: List[float] = None,
    max_passes: int = 10,
    improvement_threshold: float = 0.0001,
    scale_by_spread: bool = False,
    rule_only: bool = False,
) -> Dict[str, Any]:
    """
    Coordinate-ascent search for per-rule weights maximizing dev accuracy.

    Returns dict with:
      - weights: List[float]
      - baseline_accuracy: float  # with all weights = 1.0
      - final_accuracy: float
      - passes: int
      - changes: List[Dict[str, Any]] with keys (rule_index, old_weight, new_weight, new_accuracy)
    """
    if weight_grid is None:
        weight_grid = [-2.0, -1.5, -1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0, 1.25, 1.5, 2.0, 3.0]

    multi_results = load_multi_dataset_results(dev_eval_summary)
    samples = _flatten_multi_results_for_samples(multi_results, rule_only=rule_only)
    rule_funcs = _compile_rule_functions(rules)
    precomputed = _precompute_rule_choice_scores(samples, rule_funcs, scale_by_spread=scale_by_spread)

    weights = [1.0 for _ in rules]
    baseline_accuracy = _accuracy_from_weights(samples, precomputed, weights)

    changes: List[Dict[str, Any]] = []
    current_accuracy = baseline_accuracy
    num_passes = 0

    while num_passes < max_passes:
        num_passes += 1
        improved_any = False

        for r in range(len(weights)):
            best_w = weights[r]
            best_acc = current_accuracy

            for w in weight_grid:
                if w == weights[r]:
                    continue
                trial = list(weights)
                trial[r] = w
                acc = _accuracy_from_weights(samples, precomputed, trial)
                if acc > best_acc + improvement_threshold:
                    best_acc = acc
                    best_w = w

            if best_w != weights[r]:
                changes.append({
                    "rule_index": r,
                    "old_weight": weights[r],
                    "new_weight": best_w,
                    "new_accuracy": best_acc
                })
                weights[r] = best_w
                current_accuracy = best_acc
                improved_any = True

        if not improved_any:
            break

    return {
        "weights": weights,
        "baseline_accuracy": baseline_accuracy,
        "final_accuracy": current_accuracy,
        "passes": num_passes,
        "changes": changes
    }

def evaluate_on_multi_dataset_with_weights(
    eval_summary_path: str,
    rules: List[str],
    weights: List[float],
    scale_by_spread: bool = False,
    rule_only: bool = False,
) -> Dict[str, Any]:
    """Evaluate rules with per-rule weights on multiple datasets.

    If scale_by_spread is True, multiply each rule score by (logprob_spread + 1.0)
    per question, to match the scaling used by training/create_scienceworld_task_rules.py.
    """
    multi_results = load_multi_dataset_results(eval_summary_path)

    dataset_results: Dict[int, Any] = {}
    total_correct = 0
    total_predictions = 0
    per_label_counts: Dict[str, Dict[str, int]] = {}
    per_topic_counts: Dict[str, Dict[str, int]] = {}

    rule_funcs = _compile_rule_functions(rules)

    for dataset_idx, dataset_result in multi_results['dataset_results'].items():
        correct_predictions = 0
        dataset_total = len(dataset_result["questions"])

        for q in dataset_result["questions"]:
            state, action = _extract_state_and_action_from_question(q)
            choices = q["choices"]
            original_logprobs = q["choice_logprobs"]
            correct_idx = q["correct_idx"]
            label = q.get("label", "UNLABELED")
            topic = q.get("topic", "UNTOPICED")

            if rule_only:
                modified_logprobs = np.zeros(len(original_logprobs), dtype=float)
            else:
                modified_logprobs = np.array(original_logprobs, dtype=float)
            scale = 1.0
            if scale_by_spread and not rule_only:
                try:
                    spread = float(np.max(modified_logprobs) - np.min(modified_logprobs))
                    scale = spread + 1.0
                except Exception:
                    scale = 1.0

            for w, func in zip(weights, rule_funcs):
                if w == 0.0:
                    continue
                try:
                    for i, choice in enumerate(choices):
                        modified_logprobs[i] += float(w) * float(func(state, action, choice)) * scale
                except Exception:
                    continue

            new_pred_idx = int(np.argmax(modified_logprobs))
            if new_pred_idx == correct_idx:
                correct_predictions += 1

            counts = per_label_counts.get(label)
            if counts is None:
                counts = {"correct": 0, "total": 0}
                per_label_counts[label] = counts
            counts["total"] += 1
            if new_pred_idx == correct_idx:
                counts["correct"] += 1
            tcounts = per_topic_counts.get(topic)
            if tcounts is None:
                tcounts = {"correct": 0, "total": 0}
                per_topic_counts[topic] = tcounts
            tcounts["total"] += 1
            if new_pred_idx == correct_idx:
                tcounts["correct"] += 1

        dataset_accuracy = correct_predictions / dataset_total if dataset_total > 0 else 0.0
        dataset_results[dataset_idx] = {
            "accuracy": dataset_accuracy,
            "total": dataset_total,
            "correct": correct_predictions,
            "dataset_path": multi_results['summary']['dataset_paths'][dataset_idx]
        }

        total_correct += correct_predictions
        total_predictions += dataset_total

    overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0
    per_label_summary = {
        label: {
            "accuracy": (cnts["correct"] / cnts["total"]) if cnts["total"] > 0 else 0.0,
            "total": cnts["total"],
            "correct": cnts["correct"]
        }
        for label, cnts in per_label_counts.items()
    }
    per_topic_summary = {
        topic: {
            "accuracy": (cnts["correct"] / cnts["total"]) if cnts["total"] > 0 else 0.0,
            "total": cnts["total"],
            "correct": cnts["correct"]
        }
        for topic, cnts in per_topic_counts.items()
    }

    return {
        "dataset_results": dataset_results,
        "total_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_questions": total_predictions,
        "per_label": per_label_summary,
        "per_topic": per_topic_summary
    }

def main():
    parser = argparse.ArgumentParser(
        description="WMQA Rule Manager - Create, evaluate, or improve WMQA rules",
        epilog="""
Three operation modes:
1. CREATE MODE (default): Generate new rules from scratch using dev set wrong answers
   Requires: --dev_eval_summary
   
2. EVALUATE MODE: Evaluate existing rules on test set only
   Requires: --evaluate_rules_file
   Optional: --weights_file to load an existing weights JSON (skips weight learning),
             or --dev_eval_summary to learn weights on dev.
   
3. IMPROVE MODE: Improve existing rules and learn new rules using dev set feedback
   Requires: --improve_rules_file and --dev_eval_summary

 Clustering strategies are defined in-code. You can edit DEFAULT_CREATE_CLUSTER_STRATEGIES
 and DEFAULT_IMPROVE_CLUSTER_STRATEGIES to use single-level (string) or hierarchical
 (tuple of two keys) clustering. Supported keys include: question, correct_answer,
 wrong_answer, action, label, inventory_diff.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--eval_summary", type=str, required=True,
                       help="Path to evaluation summary JSON file (from multi-dataset eval)")
    parser.add_argument("--dev_eval_summary", type=str,
                       help="Path to dev set evaluation summary JSON file (required for rule generation and improvement)")
    parser.add_argument("--weights_file", type=str, default=None,
                       help="Path to a JSON file containing per-rule weights (e.g., {'weights': [...]}) to use in EVALUATE MODE. If provided, skips dev-set weight optimization.")
    parser.add_argument("--max_rules", type=int, default=100000,
                       help="Maximum number of rules to generate")
    parser.add_argument("--max_new_rules", type=int, default=100000,
                       help="Maximum number of new rules to generate in improve mode")
    parser.add_argument("--max_reflection_attempts", type=int, default=3,
                       help="Maximum number of reflection attempts when a rule has negative impact")
    parser.add_argument("--evaluate_rules_file", type=str, nargs='+', default=None,
                       help="Paths to one or more Python files containing rules to evaluate. Enables EVALUATE MODE.")
    parser.add_argument("--improve_rules_file", type=str, nargs='+', default=None,
                       help="Paths to one or more Python files containing rules to improve. Enables IMPROVE MODE.")
    parser.add_argument("--output_rules_file", type=str, default=None,
                       help="Path to a Python file to save the rules.")
    parser.add_argument(
        "--scale_rules_by_spread",
        action="store_true",
        help="Scale each rule score by (logprob_spread + 1.0) per question (matches create_scienceworld_task_rules.py).",
    )
    parser.add_argument(
        "--rule_only",
        action="store_true",
        help="Evaluate mode only: ignore logprobs from the summary (treat as 0) and score only by rules.",
    )
    args = parser.parse_args()

    global _GLOBAL_SCALE_RULES_BY_SPREAD
    _GLOBAL_SCALE_RULES_BY_SPREAD = bool(args.scale_rules_by_spread)

    # Validate mode arguments
    mode_count = sum([
        bool(args.evaluate_rules_file),
        bool(args.improve_rules_file)
    ])
    
    if mode_count > 1:
        eprint("Error: Cannot specify multiple modes. Choose one of --evaluate_rules_file or --improve_rules_file.")
        sys.exit(1)

    if args.rule_only and not args.evaluate_rules_file:
        eprint("Error: --rule_only can only be used in evaluate mode (--evaluate_rules_file).")
        sys.exit(1)

    if args.rule_only and args.scale_rules_by_spread:
        eprint("Error: --rule_only is not compatible with --scale_rules_by_spread.")
        sys.exit(1)

    random.seed(42)
    np.random.seed(42)

    # --- NEW: Evaluation-only mode ---
    if args.evaluate_rules_file:
        rule_files = args.evaluate_rules_file if isinstance(args.evaluate_rules_file, list) else [args.evaluate_rules_file]
        eprint(f"Evaluation mode: Loading rules from {len(rule_files)} file(s)")
        combined_rules = []
        for rf in rule_files:
            if not os.path.exists(rf):
                eprint(f"Error: Rule file not found at {rf}")
                sys.exit(1)
            file_rules = load_rules_from_file(rf)
            if not file_rules:
                eprint(f"Warning: No rules found in {rf}")
            else:
                eprint(f"Loaded {len(file_rules)} rules from {rf}")
                combined_rules.extend(file_rules)
        if not combined_rules:
            eprint("Error: No rules found in the specified files.")
            sys.exit(1)
        eprint(f"Total rules loaded: {len(combined_rules)}")

        # Load the dataset for evaluation
        multi_eval_results = load_multi_dataset_results(args.eval_summary)
        baseline_total_accuracy = multi_eval_results['summary']['total_accuracy']

        # Dev-based rule weighting (coordinate ascent)
        weights: List[float] = [1.0 for _ in combined_rules]
        if args.weights_file:
            if not os.path.exists(args.weights_file):
                eprint(f"Error: weights file not found at {args.weights_file}")
                sys.exit(1)
            try:
                loaded_weights = load_rule_weights_from_json(args.weights_file)
            except Exception as e:
                eprint(f"Error: failed to load weights from {args.weights_file} - {e}")
                sys.exit(1)
            if len(loaded_weights) != len(combined_rules):
                eprint(
                    "Error: weights length mismatch: "
                    f"got {len(loaded_weights)} weights but loaded {len(combined_rules)} rules. "
                    "Ensure the weights correspond to the same rule file(s) and ordering."
                )
                sys.exit(1)
            weights = loaded_weights
            eprint(f"\nUsing provided weights from {args.weights_file} (n={len(weights)}). Skipping dev-set weight optimization.")
        elif args.dev_eval_summary:
            eprint("\nOptimizing per-rule weights on the dev set (coordinate ascent)...")
            opt_info = optimize_rule_weights_on_dev(
                args.dev_eval_summary,
                combined_rules,
                improvement_threshold=0.0001,
                scale_by_spread=args.scale_rules_by_spread,
                rule_only=args.rule_only,
            )
            weights = opt_info["weights"]
            eprint(f"Dev accuracy with unit weights: {opt_info['baseline_accuracy']:.2%}")
            eprint(f"Dev accuracy after weighting: {opt_info['final_accuracy']:.2%} (passes: {opt_info['passes']})")

            # Save weights to JSON next to rules
            if args.output_rules_file:
                weights_filename = args.output_rules_file.replace('.py', '_weights.json') if args.output_rules_file.endswith('.py') else args.output_rules_file + '_weights.json'
            elif len(rule_files) == 1 and rule_files[0].endswith('.py'):
                weights_filename = rule_files[0].replace('.py', '_weights.json')
            else:
                base_dir = os.path.dirname(rule_files[0]) if rule_files else '.'
                weights_filename = os.path.join(base_dir, 'combined_rule_weights.json')

            try:
                with open(weights_filename, 'w') as wf:
                    json.dump({"weights": weights}, wf, indent=2)
                eprint(f"Saved learned weights to {weights_filename}")
            except Exception as e:
                eprint(f"Warning: failed to save weights JSON - {e}")

        eprint("\nEvaluating weighted rules on the test set...")
        final_results = evaluate_on_multi_dataset_with_weights(
            args.eval_summary,
            combined_rules,
            weights,
            scale_by_spread=args.scale_rules_by_spread,
            rule_only=args.rule_only,
        )
        
        eprint(f"\n--- Evaluation Results ---")
        eprint(f"Baseline total accuracy: {baseline_total_accuracy:.2%}")
        
        eprint(f"\nPost-rule accuracies:")
        for dataset_idx, dataset_result in final_results["dataset_results"].items():
            baseline_dataset_acc = multi_eval_results['summary']['dataset_accuracies'][dataset_idx]
            improvement = dataset_result['accuracy'] - baseline_dataset_acc
            eprint(f"  Dataset {dataset_idx+1}: {dataset_result['accuracy']:.2%} ({dataset_result['correct']}/{dataset_result['total']}) | Improvement: {improvement:+.2%}")
        
        eprint(f"\nFinal total accuracy: {final_results['total_accuracy']:.2%}")
        eprint(f"Total improvement: {final_results['total_accuracy'] - baseline_total_accuracy:+.2%}")

        # New: print per-label accuracies aggregated across all datasets
        if 'per_label' in final_results and final_results['per_label']:
            eprint("\nPer-label accuracies (all datasets):")
            # Sort labels alphabetically for stable output
            for label in sorted(final_results['per_label'].keys()):
                stats = final_results['per_label'][label]
                eprint(f"  {label}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

        # Print per-topic accuracies aggregated across all datasets
        if 'per_topic' in final_results and final_results['per_topic']:
            eprint("\nPer-topic accuracies (all datasets):")
            # Sort topics alphabetically for stable output
            for topic in sorted(final_results['per_topic'].keys()):
                stats = final_results['per_topic'][topic]
                eprint(f"  {topic}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

        return # End execution

    # --- NEW: Improve-rules mode ---
    if args.improve_rules_file:
        rule_files = args.improve_rules_file if isinstance(args.improve_rules_file, list) else [args.improve_rules_file]
        eprint(f"Improve mode: Loading rules from {len(rule_files)} file(s)")
        combined_rules = []
        for rf in rule_files:
            if not os.path.exists(rf):
                eprint(f"Error: Rule file not found at {rf}")
                sys.exit(1)
            file_rules = load_rules_from_file(rf)
            if not file_rules:
                eprint(f"Warning: No rules found in {rf}")
            else:
                eprint(f"Loaded {len(file_rules)} rules from {rf}")
                combined_rules.extend(file_rules)
        if not combined_rules:
            eprint("Error: No rules found in the specified file(s).")
            sys.exit(1)
        eprint(f"Total rules loaded for improvement: {len(combined_rules)}")
        
        if not args.dev_eval_summary:
            eprint("Error: --dev_eval_summary is required for improve mode.")
            sys.exit(1)
        
        # Use the combined rules from all provided files
        rules = combined_rules

        # Evaluate original rules on dev set
        eprint("\nEvaluating original rules on dev set...")
        original_results = evaluate_on_multi_dataset(args.dev_eval_summary, rules)
        original_accuracy = original_results["total_accuracy"]
        
        eprint(f"Original rule accuracies on dev set:")
        for dataset_idx, dataset_result in original_results["dataset_results"].items():
            eprint(f"  Dataset {dataset_idx+1}: {dataset_result['accuracy']:.2%} ({dataset_result['correct']}/{dataset_result['total']})")
        eprint(f"  Total accuracy: {original_accuracy:.2%}")

        # Improve the rules
        eprint(f"\nImproving rules using dev set feedback...")
        improved_rules = improve_existing_rules(
            rules,
            args.dev_eval_summary,
            args.max_reflection_attempts,
            args.max_new_rules,
        )
        
        # Evaluate improved rules on dev set
        eprint(f"\nEvaluating improved rules on dev set...")
        improved_results = evaluate_on_multi_dataset(args.dev_eval_summary, improved_rules)
        improved_accuracy = improved_results["total_accuracy"]
        
        eprint(f"Improved rule accuracies on dev set:")
        for dataset_idx, dataset_result in improved_results["dataset_results"].items():
            original_dataset_acc = original_results["dataset_results"][dataset_idx]['accuracy']
            improvement = dataset_result['accuracy'] - original_dataset_acc
            eprint(f"  Dataset {dataset_idx+1}: {dataset_result['accuracy']:.2%} ({dataset_result['correct']}/{dataset_result['total']}) | Improvement: {improvement:+.2%}")
        eprint(f"  Total accuracy: {improved_accuracy:.2%}")
        eprint(f"  Total improvement: {improved_accuracy - original_accuracy:+.2%}")

        # Optimize per-rule weights on dev (same approach as evaluation mode)
        eprint("\nOptimizing per-rule weights on the dev set (coordinate ascent)...")
        opt_info = optimize_rule_weights_on_dev(
            args.dev_eval_summary,
            improved_rules,
            improvement_threshold=0.0001,
            scale_by_spread=args.scale_rules_by_spread,
        )
        weights = opt_info["weights"]
        eprint(f"Dev accuracy with unit weights: {opt_info['baseline_accuracy']:.2%}")
        eprint(f"Dev accuracy after weighting: {opt_info['final_accuracy']:.2%} (passes: {opt_info['passes']})")

        # Evaluate weighted rules on dev (for reporting)
        eprint("\nEvaluating improved rules on dev set with learned weights...")
        weighted_dev_results = evaluate_on_multi_dataset_with_weights(
            args.dev_eval_summary,
            improved_rules,
            weights,
            scale_by_spread=args.scale_rules_by_spread,
        )
        weighted_dev_accuracy = weighted_dev_results["total_accuracy"]
        eprint(f"Weighted improved rule accuracies on dev set:")
        for dataset_idx, dataset_result in weighted_dev_results["dataset_results"].items():
            unit_acc = improved_results["dataset_results"][dataset_idx]['accuracy']
            delta = dataset_result['accuracy'] - unit_acc
            eprint(f"  Dataset {dataset_idx+1}: {dataset_result['accuracy']:.2%} ({dataset_result['correct']}/{dataset_result['total']}) | Δ vs unit: {delta:+.2%}")
        eprint(f"  Total accuracy: {weighted_dev_accuracy:.2%} | Δ vs unit: {weighted_dev_accuracy - improved_accuracy:+.2%}")

        # Final evaluation on test set (weighted)
        eprint(f"\nEvaluating improved rules on test set with learned weights...")
        final_results = evaluate_on_multi_dataset_with_weights(
            args.eval_summary,
            improved_rules,
            weights,
            scale_by_spread=args.scale_rules_by_spread,
        )
        
        # Load test baseline for comparison
        multi_eval_results = load_multi_dataset_results(args.eval_summary)
        baseline_total_accuracy = multi_eval_results['summary']['total_accuracy']
        
        eprint(f"\nTest set results:")
        eprint(f"Baseline total accuracy: {baseline_total_accuracy:.2%}")
        
        eprint(f"\nPost-improvement accuracies:")
        for dataset_idx, dataset_result in final_results["dataset_results"].items():
            baseline_dataset_acc = multi_eval_results['summary']['dataset_accuracies'][dataset_idx]
            improvement = dataset_result['accuracy'] - baseline_dataset_acc
            eprint(f"  Dataset {dataset_idx+1}: {dataset_result['accuracy']:.2%} ({dataset_result['correct']}/{dataset_result['total']}) | Improvement: {improvement:+.2%}")
        
        eprint(f"\nFinal total accuracy: {final_results['total_accuracy']:.2%}")
        eprint(f"Total improvement: {final_results['total_accuracy'] - baseline_total_accuracy:+.2%}")

        # Save improved rules (honor --output_rules_file if provided; otherwise derive from inputs)
        if args.output_rules_file:
            improved_filename = args.output_rules_file
        elif len(rule_files) == 1:
            src = rule_files[0]
            if src.endswith('.py'):
                improved_filename = src.replace('.py', '_improved.py')
            else:
                improved_filename = src + '_improved.py'
        else:
            base_dir = os.path.dirname(rule_files[0]) if rule_files else '.'
            improved_filename = os.path.join(base_dir, 'combined_rules_improved.py')
        with open(improved_filename, "w") as f:
            f.write("# WMQA Improved Rules\n")
            if len(rule_files) == 1:
                f.write(f"# Improved from: {rule_files[0]}\n")
            else:
                f.write(f"# Improved from ({len(rule_files)} files):\n")
                for rf in rule_files:
                    f.write(f"#   - {rf}\n")
            f.write(f"# Dev unit-weight improvement vs original: {improved_accuracy - original_accuracy:+.2%}\n")
            f.write(f"# Dev unit-weight accuracy (improved rules): {improved_accuracy:.2%}\n")
            f.write(f"# Dev weighted accuracy (learned on dev): {weighted_dev_accuracy:.2%}\n")
            f.write(f"# Test baseline accuracy: {baseline_total_accuracy:.2%}\n")
            f.write(f"# Test weighted accuracy: {final_results['total_accuracy']:.2%}\n")
            f.write(f"# Test weighted improvement: {final_results['total_accuracy'] - baseline_total_accuracy:+.2%}\n\n")
            for i, rule in enumerate(improved_rules):
                f.write(f"# Rule {i+1}\n")
                f.write(rule)
                f.write("\n\n")

        eprint(f"Saved {len(improved_rules)} improved rules to {improved_filename}")

        # Save learned weights next to the improved rules file for reproducibility
        try:
            weights_filename = improved_filename.replace('.py', '_weights.json') if improved_filename.endswith('.py') else (improved_filename + '_weights.json')
            with open(weights_filename, 'w') as wf:
                json.dump({"weights": weights}, wf, indent=2)
            eprint(f"Saved learned weights to {weights_filename}")
        except Exception as e:
            eprint(f"Warning: failed to save weights JSON - {e}")
        
        return # End execution

    # --- Original rule generation logic ---
    if not args.dev_eval_summary:
        eprint("Error: --dev_eval_summary is required for rule generation mode.")
        sys.exit(1)
    
    if not args.output_rules_file:
        eprint("Error: --output_rules_file is required for rule generation mode.")
        sys.exit(1)

    # Load DEV multi-dataset evaluation results and extract wrong answers (cluster on DEV only)
    dev_multi_eval_results = load_multi_dataset_results(args.dev_eval_summary)
    wrong_answers_per_dataset = extract_multi_dataset_wrong_answers(dev_multi_eval_results)

    total_wrong_answers = sum(len(wrong_list) for wrong_list in wrong_answers_per_dataset.values())
    total_questions = sum(len(dataset['questions']) for dataset in dev_multi_eval_results['dataset_results'].values())

    eprint(f"Found {total_wrong_answers} wrong answers out of {total_questions} total questions")

    # Initial evaluation on dev set
    baseline_results = evaluate_on_multi_dataset(args.dev_eval_summary, [])
    current_accuracy = baseline_results["total_accuracy"]

    eprint(f"Baseline accuracies:")
    for dataset_idx, dataset_result in baseline_results["dataset_results"].items():
        eprint(f"  Dataset {dataset_idx+1}: {dataset_result['accuracy']:.2%} ({dataset_result['correct']}/{dataset_result['total']})")
    eprint(f"  Total accuracy: {current_accuracy:.2%}")

    final_rules = []
    rule_count = 0
    
    # Clustering and rule generation loop - process each dataset using strategies
    valid_keys = {"question", "correct_answer", "wrong_answer", "inventory_diff", "action", "label"}
    for strategy in DEFAULT_CREATE_CLUSTER_STRATEGIES:
        if rule_count >= args.max_rules:
            break
        # Hierarchical strategy
        if isinstance(strategy, tuple) and len(strategy) == 2:
            primary_key, secondary_key = strategy
            if primary_key not in valid_keys or secondary_key not in valid_keys:
                continue
            eprint(f"\nHierarchical clustering by: {primary_key} -> {secondary_key}")
            for dataset_idx, wrong_answers in wrong_answers_per_dataset.items():
                if rule_count >= args.max_rules:
                    break
                if len(wrong_answers) < 2:
                    continue
                eprint(f"Processing dataset {dataset_idx+1} with {len(wrong_answers)} wrong answers")
                nested_clusters = hierarchical_clustering_indices(wrong_answers, primary_key, secondary_key)
                eprint(f"Found {len(nested_clusters)} secondary clusters in dataset {dataset_idx+1}")
                for sub_indices in nested_clusters:
                    if rule_count >= args.max_rules:
                        break
                    cluster_samples = [wrong_answers[i] for i in sub_indices]
                    if len(cluster_samples) < 2:
                        continue
                    for attempt in range(3):
                        rule_count += 1
                        selected_samples = random.sample(cluster_samples, min(3, len(cluster_samples)))
                        try:
                            rule_response = gpt4_analyze_cluster(selected_samples)
                            programs = extract_python_programs(rule_response)
                            if not programs:
                                eprint(f"Rule {rule_count}: No valid program found")
                                continue
                            program = programs[0]
                            test_rules = final_rules + [program]
                            new_results = evaluate_on_multi_dataset(args.dev_eval_summary, test_rules)
                            new_accuracy = new_results["total_accuracy"]
                            if new_accuracy > current_accuracy + 0.0001:
                                final_rules.append(program)
                                current_accuracy = new_accuracy
                                eprint(f"Rule {rule_count}: ACCEPTED - New total accuracy: {current_accuracy:.2%}")
                                eprint(f"Rule: {rule_response.split('### Rule ###')[1].split('### Program ###')[0].strip() if '### Rule ###' in rule_response else 'N/A'}")
                            elif new_accuracy < current_accuracy - 0.0001:
                                eprint(f"Rule {rule_count}: NEGATIVE IMPACT detected - Total accuracy: {new_accuracy:.2%} (vs {current_accuracy:.2%})")
                                negatively_impacted = identify_negatively_impacted_questions(
                                    args.dev_eval_summary,
                                    baseline_rules=final_rules,
                                    new_rules=final_rules + [program]
                                )
                                if len(negatively_impacted) > 0:
                                    eprint(f"Found {len(negatively_impacted)} negatively impacted questions")
                                    max_reflection_attempts = args.max_reflection_attempts
                                    refined_program = program
                                    refined_rule_response = rule_response
                                    best_accuracy = new_accuracy
                                    refined_accuracy = new_accuracy
                                    reflection_succeeded = False
                                    for reflection_attempt in range(max_reflection_attempts):
                                        eprint(f"Reflection attempt {reflection_attempt + 1}/{max_reflection_attempts}")
                                        try:
                                            negative_cases = sample_negative_impact_cases(negatively_impacted, refined_program)
                                            refined_rule_response = gpt4_refine_rule(negative_cases, refined_program, refined_rule_response)
                                            refined_programs = extract_python_programs(refined_rule_response)
                                            if not refined_programs:
                                                eprint(f"  Reflection {reflection_attempt + 1}: No valid refined program found")
                                                continue
                                            refined_program = refined_programs[0]
                                            refined_test_rules = final_rules + [refined_program]
                                            refined_results = evaluate_on_multi_dataset(args.dev_eval_summary, refined_test_rules)
                                            refined_accuracy = refined_results["total_accuracy"]
                                            eprint(f"  Reflection {reflection_attempt + 1}: Refined accuracy: {refined_accuracy:.2%}")
                                            if refined_accuracy > current_accuracy + 0.0001:
                                                final_rules.append(refined_program)
                                                current_accuracy = refined_accuracy
                                                reflection_succeeded = True
                                                eprint(f"  Reflection {reflection_attempt + 1}: SUCCESS! Rule refined and accepted")
                                                break
                                            elif refined_accuracy > best_accuracy:
                                                best_accuracy = refined_accuracy
                                                eprint(f"  Reflection {reflection_attempt + 1}: Partial improvement, continuing...")
                                            else:
                                                eprint(f"  Reflection {reflection_attempt + 1}: No improvement, continuing...")
                                        except Exception as e:
                                            eprint(f"  Reflection {reflection_attempt + 1}: ERROR - {e}")
                                            continue
                                    if not reflection_succeeded:
                                        eprint(f"Rule {rule_count}: REJECTED after {max_reflection_attempts} reflection attempts")
                                else:
                                    eprint(f"Rule {rule_count}: REJECTED - No negatively impacted questions found for reflection")
                            else:
                                eprint(f"Rule {rule_count}: REJECTED - Total accuracy: {new_accuracy:.2%} (no improvement)")
                        except Exception as e:
                            eprint(f"Rule {rule_count}: ERROR - {e}")
                            continue
                # end nested clusters loop
            # end datasets loop
            continue
        
        # Single-level strategy
        if isinstance(strategy, str):
            cluster_key = strategy
            if cluster_key not in valid_keys:
                continue
            eprint(f"\nClustering by: {cluster_key}")
            for dataset_idx, wrong_answers in wrong_answers_per_dataset.items():
                if rule_count >= args.max_rules:
                    break
                if len(wrong_answers) < 2:
                    continue
                eprint(f"Processing dataset {dataset_idx+1} with {len(wrong_answers)} wrong answers")
                cluster_labels, processed_questions = dynamic_clustering_analysis(wrong_answers, cluster_key)
                cluster_ids = analyze_clusters(cluster_labels, processed_questions)
                eprint(f"Found {len(cluster_ids)} valid clusters in dataset {dataset_idx+1}")
                for cluster_id in cluster_ids:
                    if rule_count >= args.max_rules:
                        break
                    cluster_samples = [wrong_answers[i] for i in range(len(wrong_answers)) if cluster_labels[i] == cluster_id]
                    if len(cluster_samples) < 2:
                        continue
                    for attempt in range(3):
                        rule_count += 1
                        selected_samples = random.sample(cluster_samples, min(3, len(cluster_samples)))
                        try:
                            rule_response = gpt4_analyze_cluster(selected_samples)
                            programs = extract_python_programs(rule_response)
                            if not programs:
                                eprint(f"Rule {rule_count}: No valid program found")
                                continue
                            program = programs[0]
                            test_rules = final_rules + [program]
                            new_results = evaluate_on_multi_dataset(args.dev_eval_summary, test_rules)
                            new_accuracy = new_results["total_accuracy"]
                            if new_accuracy > current_accuracy + 0.0001:
                                final_rules.append(program)
                                current_accuracy = new_accuracy
                                eprint(f"Rule {rule_count}: ACCEPTED - New total accuracy: {current_accuracy:.2%}")
                                eprint(f"Rule: {rule_response.split('### Rule ###')[1].split('### Program ###')[0].strip() if '### Rule ###' in rule_response else 'N/A'}")
                            elif new_accuracy < current_accuracy - 0.0001:
                                eprint(f"Rule {rule_count}: NEGATIVE IMPACT detected - Total accuracy: {new_accuracy:.2%} (vs {current_accuracy:.2%})")
                                negatively_impacted = identify_negatively_impacted_questions(
                                    args.dev_eval_summary,
                                    baseline_rules=final_rules,
                                    new_rules=final_rules + [program]
                                )
                                if len(negatively_impacted) > 0:
                                    eprint(f"Found {len(negatively_impacted)} negatively impacted questions")
                                    max_reflection_attempts = args.max_reflection_attempts
                                    refined_program = program
                                    refined_rule_response = rule_response
                                    best_accuracy = new_accuracy
                                    refined_accuracy = new_accuracy
                                    reflection_succeeded = False
                                    for reflection_attempt in range(max_reflection_attempts):
                                        eprint(f"Reflection attempt {reflection_attempt + 1}/{max_reflection_attempts}")
                                        try:
                                            negative_cases = sample_negative_impact_cases(negatively_impacted, refined_program)
                                            refined_rule_response = gpt4_refine_rule(negative_cases, refined_program, refined_rule_response)
                                            refined_programs = extract_python_programs(refined_rule_response)
                                            if not refined_programs:
                                                eprint(f"  Reflection {reflection_attempt + 1}: No valid refined program found")
                                                continue
                                            refined_program = refined_programs[0]
                                            refined_test_rules = final_rules + [refined_program]
                                            refined_results = evaluate_on_multi_dataset(args.dev_eval_summary, refined_test_rules)
                                            refined_accuracy = refined_results["total_accuracy"]
                                            eprint(f"  Reflection {reflection_attempt + 1}: Refined accuracy: {refined_accuracy:.2%}")
                                            if refined_accuracy > current_accuracy + 0.0001:
                                                final_rules.append(refined_program)
                                                current_accuracy = refined_accuracy
                                                reflection_succeeded = True
                                                eprint(f"  Reflection {reflection_attempt + 1}: SUCCESS! Rule refined and accepted")
                                                break
                                            elif refined_accuracy > best_accuracy:
                                                best_accuracy = refined_accuracy
                                                eprint(f"  Reflection {reflection_attempt + 1}: Partial improvement, continuing...")
                                            else:
                                                eprint(f"  Reflection {reflection_attempt + 1}: No improvement, continuing...")
                                        except Exception as e:
                                            eprint(f"  Reflection {reflection_attempt + 1}: ERROR - {e}")
                                            continue
                                    if not reflection_succeeded:
                                        eprint(f"Rule {rule_count}: REJECTED after {max_reflection_attempts} reflection attempts")
                                else:
                                    eprint(f"Rule {rule_count}: REJECTED - No negatively impacted questions found for reflection")
                            else:
                                eprint(f"Rule {rule_count}: REJECTED - Total accuracy: {new_accuracy:.2%} (no improvement)")
                        except Exception as e:
                            eprint(f"Rule {rule_count}: ERROR - {e}")
                            continue
                # end clusters within dataset
            # end datasets loop
        # end single-level strategy
    
    # Final evaluation
    if final_rules:
        final_results = evaluate_on_multi_dataset(args.eval_summary, final_rules)
        # Load TEST baseline for comparison (TEST may have different number of datasets)
        test_multi_eval_results = load_multi_dataset_results(args.eval_summary)
        baseline_total_accuracy = test_multi_eval_results['summary']['total_accuracy']

        eprint(f"\nFinal evaluation on test set:")
        eprint(f"Test accuracies:")
        for dataset_idx, dataset_result in final_results["dataset_results"].items():
            baseline_dataset_acc = test_multi_eval_results['summary']['dataset_accuracies'][dataset_idx]
            improvement = dataset_result['accuracy'] - baseline_dataset_acc
            eprint(f"  Dataset {dataset_idx+1}: {dataset_result['accuracy']:.2%} ({dataset_result['correct']}/{dataset_result['total']}) - Improvement: {improvement:.2%}")
        eprint(f"Total accuracy: {final_results['total_accuracy']:.2%}")
        eprint(f"Total improvement: {final_results['total_accuracy'] - baseline_total_accuracy:.2%}")

        # Save rules
        with open(args.output_rules_file, "w") as f:
            f.write("# WMQA Improvement Rules\n\n")
            for i, rule in enumerate(final_rules):
                f.write(f"# Rule {i+1}\n")
                f.write(rule)
                f.write("\n\n")

        eprint(f"Saved {len(final_rules)} rules to wmqa_train_3b_rules.py")
    else:
        eprint("No beneficial rules found")

if __name__ == '__main__':
    main()
