# WMQA Improved Rules
# Improved from (2 files):
#   - transition_mcq/rules_scienceworld_llama3-2-1b-instruct.py
#   - transition_mcq/scienceworld_task_combined_rules_llama3-2-1b-instruct.py
# Dev unit-weight improvement vs original: +4.02%
# Dev unit-weight accuracy (improved rules): 75.45%
# Dev weighted accuracy (learned on dev): 80.28%
# Test baseline accuracy: 61.67%
# Test weighted accuracy: 68.29%
# Test weighted improvement: +6.61%

# Rule 1
def rule_reward(state, action, choice):
    import re
    # If action not provided, extract from state
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''
    # Only apply to 'open ...' actions
    if not re.match(r'(?i)^\s*open\s+.+', action):
        return 0.0

    # Parse predicted_observation and predicted_inventory_diff from choice
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)(?:\npredicted_reward:|\Z)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)
    if not obs_m or diff_m is None:
        # Missing expected fields -> negative but not extreme
        return -0.5

    obs = obs_m.group(1).strip().lower()
    diff = diff_m.group(1)

    # Condition 1: observation should mention the object is now open or already open
    cond_obs = bool(re.search(r'\b(is now open|now open|already open)\b', obs))

    # Condition 2: inventory diff should be empty (no non-empty +/- lines)
    inv_lines = [ln.strip() for ln in diff.splitlines() if ln.strip()]
    # Consider lines starting with '+' or '-' as inventory changes
    inv_changes = [ln for ln in inv_lines if ln.startswith('+') or ln.startswith('-')]
    cond_inv_empty = (len(inv_changes) == 0)

    # Scoring
    if cond_obs and cond_inv_empty:
        return 1.0
    if cond_obs and not cond_inv_empty:
        # Good observation but incorrectly claims inventory changes
        return 0.0
    if not cond_obs and cond_inv_empty:
        # No proper "is now open" wording but at least no bogus inventory change
        return -0.2
    # Neither condition satisfied
    return -0.8

# Rule 2
def rule_reward(state, action, choice):
    import re
    # If action not provided, extract from state
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''

    # Parse predicted_observation, predicted_reward (unused), predicted_inventory_diff
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)\npredicted_reward:', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)

    # If parsing fails, return a neutral/weak penalty
    if not obs_m or diff_m is None:
        return -0.5

    obs = obs_m.group(1).strip().lower()
    inv = diff_m.group(1) or ''
    # Normalize action
    act = action.strip().lower()

    # Match focus action: "focus on X" or "focus X"
    mact = re.match(r'(?i)^\s*focus(?:\s+on)?\s+(.+?)\s*$', act)
    if not mact:
        return 0.0  # Rule not applicable

    obj_raw = mact.group(1).strip()
    # Tokenize object, remove articles
    obj_tokens = [t for t in re.findall(r'\w+', obj_raw.lower()) if t not in ('the', 'a', 'an')]

    # Check observation: must mention 'focus' and all object tokens
    obs_ok = False
    if 'focus' in obs:
        if obj_tokens:
            if all(tok in obs for tok in obj_tokens):
                obs_ok = True
        else:
            # no tokens to check, presence of 'focus' is enough
            obs_ok = True

    # Check inventory diff: should contain no '+' or '-' change lines
    inv_lines = [ln.strip() for ln in inv.splitlines() if ln.strip()]
    has_changes = any(re.match(r'^[\+\-]\s+', ln) for ln in inv_lines)
    inv_ok = not has_changes

    # Scoring: reward full if both ok; partial if one ok; penalize if neither
    score = 0.0
    if obs_ok:
        score += 0.6
    else:
        score -= 0.6
    if inv_ok:
        score += 0.4
    else:
        score -= 0.4

    # Clamp to [-1, 1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return float(score)

# Rule 3
def rule_reward(state, action, choice):
    import re
    # If action not provided, extract it from state
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''
    # Only apply rule to 'open <obj>' actions
    m_act = re.match(r'(?i)\s*open\s+(.+)$', action.strip())
    if not m_act:
        return 0.0

    obj = m_act.group(1).strip().lower()

    # Parse choice fields robustly
    obs_m = re.search(r'(?is)predicted_observation:\s*(.*?)(?:\npredicted_reward:|\Z)', choice)
    rew_m = re.search(r'(?m)predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?is)predicted_inventory_diff\s*:\s*(.*)$', choice)

    if not (obs_m and rew_m and diff_m is not None):
        # malformed choice
        return -0.5

    obs = obs_m.group(1).strip().lower()
    try:
        rew = float(rew_m.group(1))
    except:
        return -0.5
    inv = diff_m.group(1)

    score = 0.0
    # Observation check: should indicate open or already open and mention object
    ok_open_phrase = ('now open' in obs) or ('already open' in obs) or ('is now open' in obs) or ('is already open' in obs)
    mentions_obj = obj in obs  # simple containment check
    if ok_open_phrase and mentions_obj:
        score += 0.6
    elif ok_open_phrase:
        # open phrase present but object not explicitly mentioned
        score += 0.3

    # Inventory diff should be empty (no + or - product lines)
    inv_lines = [ln.strip() for ln in inv.splitlines() if ln.strip()]
    has_change = any(ln.startswith('+') or ln.startswith('-') for ln in inv_lines)
    if not inv_lines or not has_change:
        score += 0.2
    else:
        # penalize if inventory changes are present for an open action
        score -= 0.3

    # Reward should be exactly 0.0 for a typical 'open' step
    if abs(rew - 0.0) < 1e-6:
        score += 0.2
    else:
        score -= 0.2

    # clamp to [-1, 1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return score

# Rule 4
def rule_reward(state, action, choice):
    import re
    # Helper to clamp
    def clamp(x):
        return max(-1.0, min(1.0, x))

    # Extract action from state if not provided
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state or '')
        action = m.group(1).strip() if m else ''

    # Only apply to connect actions
    if not re.search(r'(?i)^\s*connect\b', action):
        return 0.0

    # Parse choice fields robustly
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)\s*\n\s*predicted_reward:', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)

    if not (obs_m and rew_m and diff_m is not None):
        return -0.5

    obs = obs_m.group(1).strip().lower()
    inv = diff_m.group(1)

    # Extract endpoints from the action: try "connect <A> to <B>"
    m = re.search(r'(?i)^\s*connect\s+(.+?)\s+to\s+(.+)$', action.strip())
    if not m:
        # action is connect but not in expected simple form; penalize mildly
        return -0.3

    left = m.group(1).strip().lower()
    right = m.group(2).strip().lower()

    # Build token sets of relevant keywords from each endpoint
    def extract_tokens(s):
        # keep words and terminal numbers, anode/cathode keywords
        toks = re.findall(r'\b(?:terminal\s*\d+|anode|cathode|\w+)\b', s)
        # normalize "terminal 1" -> "terminal 1"
        toks = [re.sub(r'\s+', ' ', t) for t in toks]
        return set(toks)

    left_toks = extract_tokens(left)
    right_toks = extract_tokens(right)
    combined_toks = list(left_toks | right_toks)

    # Check observation contains connection indicator
    obs_ok = bool(re.search(r'\bconnected\b|\bnow connected\b|\bis now connected\b|\bnow\s+connected\b', obs))
    score = 0.0

    if not obs_ok:
        score -= 0.5
    else:
        # require that observation mentions at least two important tokens from action endpoints
        match_count = 0
        for tok in combined_toks:
            # match tokens loosely in obs
            if tok and re.search(re.escape(tok), obs):
                match_count += 1
        # Consider also matching the raw device names if tokenization split them
        # If we have at least two matches, consider observation consistent
        if match_count >= 2:
            score += 0.7
        else:
            # partial credit if at least one token matched
            if match_count == 1:
                score += 0.2
            else:
                score -= 0.5

    # Inventory diff should not have +/- lines for connect
    inv_lines = [ln for ln in (inv or '').splitlines() if ln.strip()]
    has_changes = any(ln.strip().startswith(('+', '-')) for ln in inv_lines)
    if not has_changes:
        score += 0.3
    else:
        score -= 0.3

    return clamp(score)

# Rule 5
def rule_reward(state, action, choice):
    import re
    def clamp(x):
        return max(-1.0, min(1.0, x))

    # If action not provided, try to extract from state
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''

    # Parse predicted fields from choice
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)\n(?=predicted_reward:)', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff:\s*(.*)$', choice)

    # If parsing failed, give a moderate negative signal
    if not obs_m or diff_m is None:
        return -0.5

    obs = obs_m.group(1).strip().lower()
    inv_diff = diff_m.group(1).strip()

    # Only apply rule for 'focus on' actions
    mact = re.match(r'(?i)^\s*focus on\s+(.+?)(?:\s+in\s+.+)?\s*$', action.strip())
    if not mact:
        return 0.0

    obj_raw = mact.group(1).strip().lower()

    # Normalize object string: remove leading articles 'the', 'a', 'an'
    obj_norm = re.sub(r'^(the|a|an)\s+', '', obj_raw).strip()

    score = 0.0

    # Require the observation to contain the phrase 'you focus on'
    if 'you focus on' in obs:
        score += 0.6
    else:
        # if observation uses slightly different but acceptable phrasing, still allow small credit
        if re.search(r'\bfocus(es)? on\b', obs):
            score += 0.3

    # Require the observation to mention the object (substring match of normalized object)
    # Split object into words and check that at least one significant token appears (but prefer full substring)
    if obj_norm and obj_norm in obs:
        score += 0.4
    else:
        # fallback: check if at least one non-trivial token from obj appears
        tokens = [t for t in re.split(r'\s+', obj_norm) if len(t) > 2]
        if tokens and any(tok in obs for tok in tokens):
            score += 0.15

    # Inventory diff should be empty for focus actions; penalize non-empty diffs
    if inv_diff and len(inv_diff.strip()) > 0:
        # Strong penalty because focus should not change inventory
        score -= 0.9

    return clamp(score)

# Rule 6
def rule_reward(state, action, choice):
    import re
    # Extract action if not provided
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''

    # Parse choice fields
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)\n', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)
    if not (obs_m and rew_m and diff_m is not None):
        return -0.5  # malformed choice

    obs = obs_m.group(1).strip().lower()
    try:
        rew = float(rew_m.group(1))
    except:
        rew = 0.0
    inv_diff = diff_m.group(1)

    # Apply only to movement actions: "go to <loc>" or "move to <loc>"
    mact = re.match(r'(?i)^\s*(go to|move to)\s+(.+)$', action.strip())
    if not mact:
        return 0.0

    loc = mact.group(2).strip().lower()
    # Normalize location for simple matching (allow optional leading "the")
    loc_no_the = re.sub(r'^(the\s+)', '', loc).strip()

    score = 0.0

    # Check observation mentions movement
    move_phrase_ok = False
    if re.search(r'\byou (move to|move into|move)\b', obs):
        # check that the location appears (allow with/without "the")
        if loc in obs or loc_no_the in obs or re.search(r'\bthe\s+' + re.escape(loc_no_the) + r'\b', obs):
            move_phrase_ok = True

    if move_phrase_ok:
        score += 0.6
    else:
        score -= 0.6

    # Check inventory diff is empty (no non-blank lines)
    inv_lines = [ln for ln in inv_diff.splitlines() if ln.strip()]
    if len(inv_lines) == 0:
        score += 0.4
    else:
        score -= 0.4

    # Slightly penalize clearly negative reward for a movement prediction
    if rew < 0:
        score -= 0.2
    # Cap final score into [-1,1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return score

# Rule 7
def rule_reward(state, action, choice):
    import re
    # Extract action from state if not provided
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state or '')
        action = m.group(1).strip() if m else ''
    # Parse choice into fields
    obs_m = re.search(r'(?si)predicted_observation:\s*(.*?)\s*predicted_reward:', choice or '')
    rew_m = re.search(r'(?i)predicted_reward:\s*([-+]?\d*\.?\d+)', choice or '')
    diff_m = re.search(r'(?si)predicted_inventory_diff\s*:\s*(.*)$', choice or '')
    if not (obs_m and rew_m and diff_m is not None):
        return -0.5
    obs = obs_m.group(1).strip().lower()
    inv = diff_m.group(1) if diff_m else ''
    # Check if action is a focus action and extract object
    m = re.match(r'(?i)\s*focus on\s+(.+?)(?:\s+in\s+.+)?\s*$', action.strip())
    if not m:
        return 0.0  # rule not applicable
    obj = m.group(1).strip().lower()
    # Normalize whitespace/punctuation in object for matching
    obj_norm = re.sub(r'[^\w\s]', '', obj)
    obs_norm = re.sub(r'[^\w\s]', '', obs)
    score = 0.0
    # Check for any "you focus on" phrase
    if 'you focus on' in obs_norm:
        # Check exact object mention after phrase
        # Accept either "you focus on {obj}" or "you focus on the {obj}"
        if (f'you focus on {obj_norm}' in obs_norm) or (f'you focus on the {obj_norm}' in obs_norm):
            score += 1.0
        else:
            # Focus phrase present but object mismatches
            score -= 0.6
    else:
        # No focus phrase present
        score -= 0.8
    # Inventory diff should be empty for focus actions
    inv_lines = [ln for ln in (inv.splitlines() if inv is not None else []) if ln.strip() != '']
    if len(inv_lines) > 0:
        score -= 0.4
    # Clamp to [-1, 1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return float(score)

# Rule 8
def rule_reward(state, action, choice):
    import re
    # Extract action from state if not provided
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''
    # Parse choice into predicted_observation, predicted_reward, predicted_inventory_diff
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)\n(?=predicted_reward:)', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)
    if not (obs_m and rew_m and diff_m is not None):
        return -0.5  # malformed choice
    obs = obs_m.group(1).strip().lower()
    inv = diff_m.group(1)
    # Check if action is an 'open ...' action
    m = re.match(r'(?i)\s*open\s+(.+)$', action.strip())
    if not m:
        return 0.0  # rule not applicable
    obj = m.group(1).strip().lower()
    # Build token list from object for simple mention checks
    tokens = [t for t in re.split(r'[\s,._/]+', obj) if t]
    # Determine if observation uses canonical wording
    has_stateful_phrase = bool(re.search(r'\b(is now open|already open)\b', obs))
    # Determine if observation mentions the object (any token appears)
    mentions_obj = False
    for tok in tokens:
        if len(tok) >= 2 and re.search(r'\b' + re.escape(tok) + r'\b', obs):
            mentions_obj = True
            break
    # Accept generic "the door is now open" even if obj contains location
    if 'the door is now open' in obs:
        mentions_obj = True
        has_stateful_phrase = True
    # Partial-accept alternative phrasing like "you open the <obj>"
    has_you_open = bool(re.search(r'\byou (open|opened)\b', obs))
    # Inventory diff: consider non-empty lines as changes
    inv_lines = [ln for ln in inv.splitlines() if ln.strip()]
    no_change = len(inv_lines) == 0
    # Scoring logic
    if has_stateful_phrase and mentions_obj and no_change:
        return 1.0
    if has_you_open and mentions_obj and no_change:
        return 0.2
    if has_stateful_phrase and mentions_obj and not no_change:
        return -0.2
    # If inventory is unchanged but wording is non-canonical and doesn't mention object => mild penalty
    if no_change and (has_stateful_phrase or has_you_open):
        return 0.0
    # Otherwise, negative score for wrong phrasing or spurious inventory diffs
    return -0.5

# Rule 9
def rule_reward(state, action, choice):
    import re
    # Extract action from state if not provided
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''
    # Parse predicted_observation, predicted_reward, predicted_inventory_diff
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)\npredicted_reward:', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)
    if not (obs_m and rew_m and diff_m is not None):
        return -0.5
    obs = obs_m.group(1).strip().lower()
    try:
        reward = float(rew_m.group(1))
    except:
        return -0.5
    inv = diff_m.group(1)
    # Only apply this rule to "open <obj>" actions
    if not re.match(r'(?i)^\s*open\s+.+', action):
        return 0.0
    # Check observation phrasing
    ok_obs = bool(re.search(r'\b(is now open|already open)\b', obs))
    # Check inventory diff is empty (no non-blank lines)
    inv_lines = [ln for ln in inv.splitlines() if ln.strip()]
    no_inv_change = len(inv_lines) == 0
    # Reward closeness to zero
    reward_ok = abs(reward - 0.0) <= 0.02
    # Scoring
    if ok_obs and no_inv_change and reward_ok:
        return 1.0
    if ok_obs and no_inv_change and not reward_ok:
        return 0.5
    if ok_obs and not no_inv_change:
        return -0.5
    if not ok_obs:
        return -0.6
    # Fallback
    return -0.5

# Rule 10
def rule_reward(state, action, choice):
    import re
    # Extract action from state if not provided
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''
    # Parse predicted_observation, predicted_reward, predicted_inventory_diff
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)\npredicted_reward:', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)
    if not (obs_m and rew_m and diff_m is not None):
        return -0.5  # can't parse, mild penalty
    obs = obs_m.group(1).strip().lower()
    try:
        reward_val = float(rew_m.group(1))
    except:
        return -0.5
    inv = diff_m.group(1) or ""
    # Only apply rule for 'open <obj>' actions
    am = re.match(r'(?i)\s*open\s+(.+)$', action.strip())
    if not am:
        return 0.0  # rule not applicable
    obj = am.group(1).strip().lower()
    # If the observation simply echoes the command, strong negative score
    if obs == action.strip().lower() or obs.startswith(action.strip().lower()):
        return -1.0
    # Check for acceptable observation phrases
    obs_ok = ('the door is now open' in obs) or ('already open' in obs) or ('is already open' in obs)
    # Evaluate inventory diff: consider empty if no non-blank lines or only whitespace
    inv_lines = [ln for ln in (inv.splitlines() if inv else []) if ln.strip()]
    inv_empty = len(inv_lines) == 0
    # Scoring
    score = 0.0
    if obs_ok:
        score += 0.5
    else:
        # observation is wrong (but not an echo)
        score -= 0.8
    # Reward should be exactly 0.0
    if abs(reward_val - 0.0) < 1e-6:
        score += 0.3
    else:
        score -= 0.2
    # Inventory diff should be empty
    if inv_empty:
        score += 0.2
    else:
        score -= 0.2
    # Clip to [-1, 1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return score

# Rule 11
def rule_reward(state, action, choice):
    import re

    # Helper: safe lower and normalize
    def norm(s):
        return s.lower().strip() if s is not None else ''

    # If action not provided, try to extract from state
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state or '')
        action = m.group(1).strip() if m else ''

    # Parse predicted_observation, predicted_reward, predicted_inventory_diff
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)\n\s*predicted_reward:', choice or '')
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice or '')
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice or '')

    # Conservative: if we can't reliably parse required fields, do not apply rule
    if not (obs_m and rew_m and diff_m is not None):
        return 0.0

    obs_raw = obs_m.group(1).strip()
    obs = obs_raw.lower()
    inv = diff_m.group(1) or ''  # may be empty or multi-line

    # Match 'focus on <obj>' optionally followed by 'in inventory'
    m_action = re.match(r'(?i)^\s*focus on\s+(.+?)(?:\s+in\s+inventory)?\s*$', (action or '').strip())
    if not m_action:
        return 0.0  # rule does not apply

    obj = m_action.group(1).strip().lower()
    # Normalize object by removing leading determiners
    obj_noship = re.sub(r'^\s*(the|a|an)\s+', '', obj).strip()
    if not obj_noship:
        # If object becomes empty after normalization, be conservative
        return 0.0

    # Check predicted observation starts with canonical prefix "you focus on"
    if not re.match(r'(?i)^\s*you focus on\b', obs):
        # Do not apply the rule if the canonical prefix is missing (be conservative)
        return 0.0

    # Extract scene object names from the state so we can validate referents.
    # We'll look for common list lines like "a ...", "an ...", "the ..." in the state text.
    state_lower = (state or '').lower()
    scene_objects = set()
    # Try to capture item lines (lines that start with whitespace and an article)
    for line in state_lower.splitlines():
        line_s = line.strip()
        # Accept lines that start with an article or contain "you see:" list entries
        m = re.match(r'^(?:a|an|the)\s+(.+)$', line_s)
        if m:
            # Trim trailing qualifiers (like ", which is ..." or ". In it is ..." etc.)
            name = re.split(r',|;|\.|\(|:| which\b| that\b| in\b', m.group(1).strip(), 1)[0].strip()
            if name:
                scene_objects.add(name)
        else:
            # Also try to find "you see:" blocks: after "you see:" following tokens like "a baby ant"
            # find all occurrences "a <name>", "an <name>", "the <name>" in the line
            for m2 in re.finditer(r'\b(?:a|an|the)\s+([a-z0-9][a-z0-9 \-\']*)', line_s):
                name = m2.group(1).strip()
                # stop at punctuation if present later in the phrase
                name = re.split(r',|;|\.|\(|:| which\b| that\b| in\b', name, 1)[0].strip()
                if name:
                    scene_objects.add(name)

    # Build list of scene object names that contain the action headword as a whole word
    head = re.escape(obj_noship)
    target_candidates = [s for s in scene_objects if re.search(r'\b' + re.escape(obj_noship) + r'\b', s)]
    # If we cannot find any scene object containing the headword, be conservative -> do not apply rule
    if not target_candidates:
        return 0.0

    # Check exact match: observation contains one of the full target candidate phrases
    def contains_phrase(text, phrase):
        # Use word-boundary aware search
        return re.search(r'\b' + re.escape(phrase) + r'\b', text) is not None

    obs_exact = any(contains_phrase(obs, cand) for cand in target_candidates)
    obs_partial = re.search(r'\b' + re.escape(obj_noship) + r'\b', obs) is not None

    # Detect mentions of other scene objects (those that don't contain the headword)
    other_scene_objects = [s for s in scene_objects if s not in target_candidates]
    mentions_other = any(contains_phrase(obs, other) for other in other_scene_objects)

    # Check inventory diff lines for any + or - change lines
    inv_lines = [ln.strip() for ln in inv.splitlines() if ln.strip()]
    has_change = any(ln.startswith('+') or ln.startswith('-') for ln in inv_lines)

    # Scoring (more conservative)
    # Exact full-name match is best
    if obs_exact and not has_change:
        return 1.0
    if obs_exact and has_change:
        return -0.5

    # If the observation explicitly mentions a different scene object, strong negative
    if mentions_other:
        return -0.7

    # Partial (headword-only) matches are acceptable but weaker
    if obs_partial and not has_change:
        return 0.5
    if obs_partial and has_change:
        # mild penalty for inventory changes combined with partial match
        return -0.25

    # Observation starts with canonical prefix but mentions none of the scene objects -> mild penalty
    return -0.5

# Rule 12
def rule_reward(state, action, choice):
    import re
    # Extract action from state if not provided
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''

    # Parse predicted_observation, predicted_reward, predicted_inventory_diff from choice
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)\npredicted_reward:', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)

    # If required fields missing, return a moderate negative score (rule cannot verify)
    if not obs_m or diff_m is None:
        return -0.5

    obs = obs_m.group(1).strip().lower()
    inv_diff = diff_m.group(1).strip().lower()

    # Check if action is a move action and extract object and container
    m = re.match(r'(?i)\s*move\s+(.+?)\s+(?:in inventory\s+)?to\s+(.+)$', action.strip())
    if not m:
        # Rule not applicable
        return 0.0

    obj_raw = m.group(1).strip()
    container_raw = m.group(2).strip()

    # Normalize by removing parentheses content and extra whitespace
    def normalize(s):
        s = re.sub(r'\(.*?\)', '', s)  # remove parenthetical notes
        s = re.sub(r'[^a-z0-9\s]', ' ', s.lower())
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    obj = normalize(obj_raw)
    container = normalize(container_raw)

    # Helper: check whether a short substring of obj appears in a text
    def appears(text, phrase):
        if not phrase:
            return False
        return phrase in text

    # For matching, try full obj, then first two words, then first word
    obj_tokens = obj.split()
    obj_keys = []
    if obj:
        obj_keys.append(obj)
    if len(obj_tokens) >= 2:
        obj_keys.append(' '.join(obj_tokens[:2]))
    if len(obj_tokens) >= 1:
        obj_keys.append(obj_tokens[0])

    # Check observation mentions "you move" and contains both object and container tokens
    ok_obs = False
    if 'you move' in obs:
        has_obj = any(k in obs for k in obj_keys)
        has_container = container in obs
        if has_obj and has_container:
            ok_obs = True

    # Check inventory diff contains a removal line for the object
    ok_diff = False
    # Split lines and look for a '- ' line that mentions the object
    for ln in inv_diff.splitlines():
        ln_stripped = ln.strip()
        if ln_stripped.startswith('- '):
            ln_lower = ln_stripped[2:].lower()
            if any(k in ln_lower for k in obj_keys):
                ok_diff = True
                break

    # Scoring as specified
    if ok_obs and ok_diff:
        return 1.0
    if ok_obs and not ok_diff:
        return -0.2
    if not ok_obs and ok_diff:
        return -0.2
    return -0.5

# Rule 13
def rule_reward(state, action, choice):
    import re
    # Helper to clamp
    def clamp(x):
        return max(-1.0, min(1.0, x))

    # If action not provided, extract from state
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''

    # Parse choice fields
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)(?:\npredicted_reward:|\Z)', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)

    if not (obs_m and rew_m and diff_m is not None):
        # Malformed continuation
        return -0.6

    obs = obs_m.group(1).strip().lower()
    inv_block = diff_m.group(1).strip()
    inv_lines = [ln.strip() for ln in inv_block.splitlines() if ln.strip()]

    act = action.strip().lower()

    score = 0.0

    # Pattern 1: focus on <obj>
    m_focus = re.match(r'(?i)^\s*focus on\s+(.+)$', act)
    if m_focus:
        obj = m_focus.group(1).strip().lower()
        # Normalize some common word orders: if begins with 'egg ' and object has two tokens, try swap
        tok = obj.split()
        obj_variants = {obj}
        if len(tok) == 2 and tok[0] == 'egg':
            obj_variants.add(f"{tok[1]} egg")
        # Observation should mention focusing and the object
        ok_obs = ('you focus on' in obs) and any(v in obs for v in obj_variants)
        if ok_obs:
            score += 0.6
        else:
            # allow partial: mention object or mention "focus" phrase
            if ('you focus on' in obs) or any(v in obs for v in obj_variants):
                score += 0.2
        # Inventory diff should be empty for focus
        if len(inv_lines) == 0:
            score += 0.4
        else:
            # penalize if any +/- lines present
            score -= 0.4
        return clamp(score)

    # Pattern 2: move <obj> (possibly in inventory) to <target>
    m_move = re.match(r'(?i)^\s*move\s+(.+?)\s+to\s+(.+)$', act)
    if m_move:
        raw_obj = m_move.group(1).strip().lower()
        target = m_move.group(2).strip().lower()
        # Remove common qualifiers like 'in inventory' from object phrase
        raw_obj = re.sub(r'\b(in|from|the)\s+inventory\b', '', raw_obj).strip()
        # Build object token variants
        tok = raw_obj.split()
        obj_variants = {raw_obj}
        if len(tok) == 2 and tok[0] == 'egg':
            obj_variants.add(f"{tok[1]} egg")
        if len(tok) == 2 and tok[1] == 'egg':
            obj_variants.add(f"egg {tok[0]}")
        # Observation should state moving and mention object and target
        ok_obs = ('you move' in obs) and any(v in obs for v in obj_variants) and (target in obs)
        if ok_obs:
            score += 0.5
        else:
            # partial credit if mentions move+object OR move+target
            if ('you move' in obs) and (any(v in obs for v in obj_variants) or (target in obs)):
                score += 0.25
        # Inventory diff should remove the object (a '- ' line) and should not add it back (+)
        has_minus_obj = any(ln.startswith('-') and any(v in ln.lower() for v in obj_variants) for ln in inv_lines)
        has_plus_obj = any(ln.startswith('+') and any(v in ln.lower() for v in obj_variants) for ln in inv_lines)
        if has_minus_obj and not has_plus_obj:
            score += 0.5
        else:
            # partial credit if at least minus exists (even if plus present) or minus missing but no plus
            if has_minus_obj and has_plus_obj:
                score += 0.2
                score -= 0.2  # minor penalty for contradictory signs
            elif has_plus_obj and not has_minus_obj:
                score -= 0.5
            else:
                # no inventory lines mentioning object -> small penalty
                score -= 0.2
        return clamp(score)

    # If action not matched by this rule, do not apply
    return 0.0

# Rule 14
def rule_reward(state, action, choice):
    import re
    # Extract action if not provided
    if not action:
        m = re.search(r'(?mi)^\s*current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''
    # Parse predicted_observation, predicted_reward, predicted_inventory_diff
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)\n(?=predicted_reward:)', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)
    if not (obs_m and rew_m and diff_m is not None):
        return -0.5  # can't parse choice reliably
    obs = obs_m.group(1).strip().lower()
    inv = diff_m.group(1) or ''
    # Only apply when action is "focus on <obj>"
    fm = re.match(r'(?i)^\s*focus on\s+(.+)$', action.strip())
    if not fm:
        return 0.0
    obj = fm.group(1).strip().lower()
    # Build set of content words from the object (alphanumeric tokens)
    obj_words = set(re.findall(r'\w+', obj))
    # Condition 1: observation contains the phrase "you focus on" and mentions at least one obj word
    cond_obs_phrase = 'you focus on' in obs
    cond_obs_mentions_obj = any(w in obs for w in obj_words) if obj_words else False
    cond_obs = cond_obs_phrase and cond_obs_mentions_obj
    # Condition 2: inventory diff contains no '+' or '-' lines (ignores empty diff)
    inv_lines = [ln.strip() for ln in inv.splitlines() if ln.strip()]
    has_changes = any(ln.startswith('+') or ln.startswith('-') for ln in inv_lines)
    cond_inv = not has_changes
    # Scoring logic
    if cond_obs and cond_inv:
        return 1.0
    if cond_obs and not cond_inv:
        return -0.8
    if (not cond_obs) and cond_inv:
        return -0.6
    return -1.0

# Rule 15
def rule_reward(state, action, choice):
    import re
    # If action not provided, extract from state
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''
    # Parse the predicted fields from choice
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)(?:\npredicted_reward:|\Z)', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)
    if not (obs_m and rew_m and diff_m is not None):
        # malformed choice
        return -0.5

    obs = obs_m.group(1).strip().lower()
    rew = float(rew_m.group(1))
    inv_diff = diff_m.group(1).strip()

    # Only apply this rule to "move ... to ..." actions
    m = re.match(r'(?i)move\s+(.+?)\s+to\s+(.+)$', action.strip())
    if not m:
        return 0.0

    raw_obj = m.group(1).strip().lower()
    raw_target = m.group(2).strip().lower()

    # Clean object and target strings (remove common qualifiers)
    def clean(s):
        s = re.sub(r'\s+in inventory\s*$', '', s)
        s = re.sub(r'\s+containing\s.*$', '', s)
        s = re.sub(r'\s+and\s+.*$', '', s)
        s = re.sub(r'^\s*(the|a|an)\s+', '', s)
        return s.strip()
    obj = clean(raw_obj)
    target = clean(raw_target)

    score = 0.0

    # Observation check: must mention moving and include object and target substrings
    obs_ok = False
    if 'move' in obs:
        # require both object and target to appear in observation text
        if obj and obj in obs and target and target in obs:
            obs_ok = True
    if obs_ok:
        score += 0.6
    else:
        # small negative for clearly wrong observation
        score -= 0.3

    # Inventory diff check: require a removal line '- ' referencing the object
    inv_lines = [ln.strip() for ln in inv_diff.splitlines() if ln.strip()]
    has_minus_obj = any(ln.startswith('-') and obj in ln.lower() for ln in inv_lines)
    if has_minus_obj:
        score += 0.4
    else:
        score -= 0.4

    # Penalize any spurious additions (+ lines) that mention unrelated items
    plus_lines = [ln for ln in inv_lines if ln.startswith('+')]
    if plus_lines:
        # heavy penalty for inventing additions
        score -= 0.5

    # Clip to [-1, 1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return score

# Rule 16
# Task group: boil
def rule_reward(state, action, choice):
    """
    Returns a float in [-1, 1] according to a conservative merged rule for several thermometer/examine actions.
    - Applies only when the current step action matches one of the target actions.
    - Expects choice formatted with three sections:
        predicted_observation: <text>
        predicted_reward: <float>
        predicted_inventory_diff: <maybe-empty text>
    - If parsing fails, action doesn't match, or inventory diff is non-empty -> return 0.0.
    - Otherwise score according to observation/reward closeness.
    """
    import re, math

    # Helper: extract current action from state if action param empty
    act = action.strip() if action and isinstance(action, str) else ""
    if not act:
        m = re.search(r'current_step_action:\s*(.*)', state or "", flags=re.IGNORECASE)
        if m:
            act = m.group(1).strip()

    # Define targets: expected numeric temps and expected rewards
    targets = {
        "examine ice cream": {
            "type": "examine",
            "expected_obs": "liquid ice cream",
            "expected_reward": 0.24,  # center between reported 0.23 and 0.25 in examples
            "reward_tol": 0.02
        },
        "use thermometer in inventory on liquid soap": {
            "type": "thermometer",
            "expected_temps": [507.0],
            "expected_reward": 0.22,
            "temp_exact_tol": 0.5,
            "temp_close_tol": 5.0,
            "reward_tol": 0.05
        },
        "use thermometer in inventory on liquid rubber": {
            "type": "thermometer",
            "expected_temps": [235.0, 227.0],
            "expected_reward": 0.22,
            "temp_exact_tol": 0.5,
            "temp_close_tol": 6.0,
            "reward_tol": 0.06
        },
        "use thermometer in inventory on liquid lead": {
            "type": "thermometer",
            "expected_temps": [1713.0],
            "expected_reward": 0.22,
            "temp_exact_tol": 0.5,
            "temp_close_tol": 10.0,
            "reward_tol": 0.05
        }
    }

    if act not in targets:
        return 0.0

    # Parse the choice into the three labeled parts using a DOTALL-aware regex
    pattern = r'^\s*predicted_observation:\s*(.*?)\s*\n\s*predicted_reward:\s*([+-]?\d+(?:\.\d+)?)\s*\n\s*predicted_inventory_diff:\s*(.*)\s*$'
    m = re.match(pattern, choice or "", flags=re.DOTALL)
    if not m:
        return 0.0

    predicted_observation = m.group(1).strip()
    try:
        predicted_reward = float(m.group(2))
    except Exception:
        return 0.0
    predicted_inventory_diff = m.group(3).strip()

    # If inventory diff contains any non-empty content, do not apply rule (avoid penalizing unrelated inventory changes)
    if predicted_inventory_diff != "":
        return 0.0

    cfg = targets[act]

    # Helper for reward closeness mapping: returns True if within tol
    def reward_close(pred, expected, tol):
        return abs(pred - expected) <= tol

    # EXAMINE case
    if cfg["type"] == "examine":
        expected_obs = cfg["expected_obs"]
        expected_reward = cfg["expected_reward"]
        reward_tol = cfg["reward_tol"]
        if predicted_observation == expected_obs:
            # observation correct: score by reward closeness
            if reward_close(predicted_reward, expected_reward, reward_tol):
                return 1.0
            else:
                return 0.5
        else:
            # clear mismatch (conservative penalty)
            return -1.0

    # THERMOMETER cases
    if cfg["type"] == "thermometer":
        # If observation exactly equals the canonical full-text expected string for one of the temps, prefer that as exact match.
        for t in cfg["expected_temps"]:
            canon = f"the thermometer measures a temperature of {int(t)} degrees celsius"
            if predicted_observation == canon:
                # exact text match
                if reward_close(predicted_reward, cfg["expected_reward"], 1e-6):
                    return 1.0
                else:
                    return 0.5

        # Otherwise try to extract a numeric temperature from the observation text
        mtemp = re.search(r'thermometer measures a temperature of\s*([+-]?\d+(?:\.\d+)?)', predicted_observation, flags=re.IGNORECASE)
        if mtemp:
            try:
                temp = float(mtemp.group(1))
            except Exception:
                temp = None
            # If we got a numeric reading, evaluate closeness to any expected temp
            if temp is not None:
                # find smallest distance to expected temps
                dists = [abs(temp - et) for et in cfg["expected_temps"]]
                min_dist = min(dists) if dists else float('inf')
                # assess reward closeness
                if reward_close(predicted_reward, cfg["expected_reward"], cfg.get("reward_tol", 0.05)):
                    # both temp and reward close -> high score
                    if min_dist <= cfg.get("temp_exact_tol", 0.5):
                        return 1.0
                    if min_dist <= cfg.get("temp_close_tol", 6.0):
                        return 0.9
                    if min_dist <= max(20.0, cfg.get("temp_close_tol", 6.0) * 2):
                        return 0.6
                    return 0.3
                else:
                    # temp present and reasonably close but reward off -> partial credit
                    if min_dist <= cfg.get("temp_close_tol", 6.0):
                        return 0.5
                    return 0.2
            else:
                # temperature mention but couldn't parse numeric -> small penalty
                return -0.5

        # If no thermometer reading present in observation, penalize strongly (conservative)
        return -1.0

    # Fallback conservative
    return 0.0

# Rule 17
# Task group: change the
def rule_reward(state, action, choice):
    import re, math

    def clamp(x, a=-1.0, b=1.0):
        return max(a, min(b, x))

    # helper to extract current action from state if action missing/empty
    def extract_current_action(state_text):
        for line in reversed(state_text.splitlines()):
            line = line.strip()
            if line.startswith("current_step_action:"):
                return line.split("current_step_action:", 1)[1].strip()
        return ""

    # determine action to check
    act = ""
    if action is not None and str(action).strip() != "":
        act = str(action).strip()
    else:
        act = extract_current_action(state if state is not None else "")

    # Only apply for these three exact actions
    if act not in ("examine ice cream", "open cupboard", "use thermometer in inventory on lead"):
        return 0.0

    # Must parse the choice into the three fields. Be conservative: if parsing fails, return 0.0
    if choice is None:
        return 0.0
    lines = choice.splitlines()

    # find header indices (first occurrences)
    po_idx = pr_idx = pid_idx = None
    for i, ln in enumerate(lines):
        if po_idx is None and ln.startswith("predicted_observation:"):
            po_idx = i
        if pr_idx is None and ln.startswith("predicted_reward:"):
            pr_idx = i
        if pid_idx is None and ln.startswith("predicted_inventory_diff:"):
            pid_idx = i
    if po_idx is None or pr_idx is None or pid_idx is None:
        return 0.0

    # extract observation (allow multi-line observation up to the reward header)
    try:
        obs_prefix = "predicted_observation:"
        obs_first = lines[po_idx][len(obs_prefix):].strip()
        obs_lines = [obs_first] if obs_first != "" else []
        for j in range(po_idx + 1, pr_idx):
            obs_lines.append(lines[j])
        predicted_observation = "\n".join([l for l in obs_lines]).strip()
    except Exception:
        return 0.0

    # extract reward (single line after header)
    try:
        pr_prefix = "predicted_reward:"
        predicted_reward_str = lines[pr_idx][len(pr_prefix):].strip()
        predicted_reward = float(predicted_reward_str)
    except Exception:
        # If reward missing or unparsable, be conservative and do not apply the rule
        return 0.0

    # extract inventory diff: any non-header lines after the inventory header
    inv_lines = []
    for j in range(pid_idx + 1, len(lines)):
        ln = lines[j]
        # stop if we accidentally hit another top-level header (be conservative)
        if ln.startswith("predicted_observation:") or ln.startswith("predicted_reward:") or ln.startswith("predicted_inventory_diff:"):
            break
        if ln.strip() != "":
            inv_lines.append(ln.strip())
    inv_empty = (len(inv_lines) == 0)

    # Now apply per-action checks

    # 1) examine ice cream: expected reward 0.26; if reward equals expected -> +1, else -1
    if act == "examine ice cream":
        expected = 0.26
        if math.isclose(predicted_reward, expected, abs_tol=1e-6):
            return clamp(1.0)
        else:
            return clamp(-1.0)

    # 2) open cupboard: expected observation "The cupboard is now open." and empty inventory diff; expected reward 0.0
    if act == "open cupboard":
        expected_obs = "The cupboard is now open."
        # exact trimmed match for observation
        if predicted_observation.strip() != expected_obs:
            return 0.0
        # inventory diff must be empty to consider this the correct continuation
        if not inv_empty:
            return 0.0
        if math.isclose(predicted_reward, 0.0, abs_tol=1e-6):
            return clamp(1.0)
        else:
            return clamp(-1.0)

    # 3) use thermometer in inventory on lead: expect observation containing
    # "the thermometer measures a temperature of X degrees celsius"
    if act == "use thermometer in inventory on lead":
        obs_lower = predicted_observation.lower()
        mtemp = re.search(r'the thermometer measures a temperature of\s*([0-9]+(?:\.[0-9]+)?)\s*degrees celsius', obs_lower)
        if not mtemp:
            # observation not of expected form -> conservative 0.0
            return 0.0
        try:
            temp = float(mtemp.group(1))
        except Exception:
            return 0.0

        expected_temp = 275.0
        expected_reward = 0.24
        # Full credit only if temp exactly expected, reward close, and no inventory change
        if abs(temp - expected_temp) <= 1e-6 and math.isclose(predicted_reward, expected_reward, abs_tol=1e-2) and inv_empty:
            return clamp(1.0)
        # Otherwise return a conservative non-negative score proportional to temperature similarity,
        # scaled by the expected reward magnitude, halved if inventory diff present.
        similarity = max(0.0, 1.0 - abs(temp - expected_temp) / expected_temp)
        score = similarity * expected_reward
        if not inv_empty:
            score *= 0.5
        return clamp(score)

    # Fallback conservative value
    return 0.0

# Rule 18
# Task group: change the
def rule_reward(state, action, choice):
    """
    Returns a float in [-1, 1] according to the merged conservative rule:
    - Targets only the exact actions:
        "use thermometer in inventory on soap in metal pot"
        "use thermometer in inventory on soap"
    - Parses choice for labeled fields:
        predicted_observation:, predicted_reward:, predicted_inventory_diff:
      (predicted_inventory_diff is required only for the second action and must be empty)
    - If parsing fails or action doesn't match, returns 0.0.
    - If required fields all match expected values -> 1.0.
    - If parsing succeeded but required fields are present and inconsistent -> -0.5.
    """
    import re

    try:
        # Normalize / extract action
        act = action if action is not None and str(action).strip() != "" else None
        if not act:
            # try to extract from state text
            m = re.search(r"current_step_action:\s*(.*)", state or "", re.IGNORECASE)
            if m:
                act = m.group(1).strip()
        if not act:
            return 0.0
        act = str(act).strip()

        # Only apply to the two exact actions
        action_a = "use thermometer in inventory on soap in metal pot"
        action_b = "use thermometer in inventory on soap"
        if act not in (action_a, action_b):
            return 0.0

        # Parse choice: require labeled fields
        # predicted_observation: <text>
        # predicted_reward: <number>
        # predicted_inventory_diff: <text>   (optional for action_a, required for action_b)
        if choice is None:
            return 0.0
        choice_text = str(choice)

        obs_match = re.search(r"predicted_observation:\s*(.*?)\s*(?=\npredicted_reward:|\Z)", choice_text, re.S)
        reward_match = re.search(r"predicted_reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", choice_text)
        inv_match = re.search(r"predicted_inventory_diff:\s*(.*)", choice_text, re.S)

        if not obs_match or not reward_match:
            # required labels missing -> don't judge
            return 0.0

        predicted_observation = obs_match.group(1).strip()
        try:
            predicted_reward = float(reward_match.group(1))
        except Exception:
            return 0.0

        predicted_inventory_diff = inv_match.group(1).strip() if inv_match else None

        tol = 1e-3

        # Evaluate each action's expected values
        if act == action_a:
            expected_observation = "the thermometer measures a temperature of 138 degrees celsius"
            expected_reward = 0.26

            # If observation matches exactly
            if predicted_observation == expected_observation:
                # reward must be very close
                if abs(predicted_reward - expected_reward) <= tol:
                    return 1.0
                else:
                    # observation correct but reward wrong -> moderate penalty
                    return -0.5
            # observation did not match -> do not judge (avoid false positives)
            return 0.0

        else:  # act == action_b
            expected_observation = "the thermometer measures a temperature of 127 degrees celsius"
            expected_reward = 0.25

            # predicted_inventory_diff must be present and explicitly empty
            if predicted_inventory_diff is None:
                return 0.0

            obs_ok = (predicted_observation == expected_observation)
            reward_ok = (abs(predicted_reward - expected_reward) <= tol)
            inv_ok = (predicted_inventory_diff == "")

            if obs_ok and reward_ok and inv_ok:
                return 1.0
            else:
                # parsing succeeded but fields inconsistent -> moderate penalty
                return -0.5

    except Exception:
        # On unexpected errors, abstain from judging
        return 0.0

# Rule 19
# Task group: determine if
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    import math

    # Allowed target actions
    allowed_actions = {
        "move paper clip to red box": "red",
        "move paper clip to green box": "green",
        "move paper clip to blue box": "blue",
        "move paper clip to yellow box": "yellow"
    }

    # Helper: extract current_step_action from state if action missing/empty
    act = (action or "").strip()
    if not act:
        for line in (state or "").splitlines():
            line = line.strip()
            if line.startswith("current_step_action:"):
                parts = line.split("current_step_action:", 1)
                if len(parts) > 1:
                    act = parts[1].strip()
                break

    if act not in allowed_actions:
        return 0.0

    # Parse the choice into three required fields (must find headers)
    try:
        obs_marker = "predicted_observation:"
        rew_marker = "predicted_reward:"
        inv_marker = "predicted_inventory_diff:"

        # Find indices of the markers (require all three and in order)
        i_obs = choice.index(obs_marker)
        i_rew = choice.index(rew_marker)
        i_inv = choice.index(inv_marker)
        if not (i_obs < i_rew < i_inv):
            return 0.0

        predicted_observation = choice[i_obs + len(obs_marker):i_rew].strip()
        predicted_reward_text = choice[i_rew + len(rew_marker):i_inv].strip()
        predicted_inventory_text = choice[i_inv + len(inv_marker):].strip()
    except Exception:
        return 0.0

    # Parse reward as float
    try:
        predicted_reward = float(predicted_reward_text)
    except Exception:
        return 0.0

    # Normalize inventory diff lines: split into non-empty lines, allow lines starting with + or - or any non-empty trailing text
    inv_lines = []
    for ln in predicted_inventory_text.splitlines():
        s = ln.strip()
        if s != "":
            inv_lines.append(s)

    # Expected canonical observation and reward
    color = allowed_actions[act]
    expected_obs_full = "(disconnecting paper clip)You move the paper clip to the {} box.".format(color)
    # Allow optional leading parenthetical; build a simple canonical phrase to look for
    expected_phrase = "you move the paper clip to the {} box".format(color)
    expected_reward = 0.17
    tol = 1e-6

    # Quick exact-match check: exact observation string, reward nearly equal, and empty inventory diff -> full credit
    if predicted_observation == expected_obs_full and abs(predicted_reward - expected_reward) <= tol and len(inv_lines) == 0:
        return 1.0

    # Otherwise compute conservative component scores in [-1,1]
    # Observation component (weight 0.5)
    obs_text = predicted_observation.strip()
    # remove optional leading parenthetical e.g., "(disconnecting paper clip)"
    stripped = obs_text
    if stripped.startswith("("):
        # remove first parenthetical group if it closes
        endp = stripped.find(")")
        if endp != -1:
            stripped = stripped[endp+1:].strip()
    low = stripped.lower()

    if obs_text == expected_obs_full:
        obs_val = 1.0
    elif expected_phrase in low:
        # contains the exact expected phrase without the parenthetical
        obs_val = 0.6
    else:
        # partial credit if it mentions both 'paper clip' and '<color> box' and a movement verb
        if "paper clip" in low and (color + " box") in low and any(v in low for v in ("move", "moved", "put", "place", "placed", "putting")):
            obs_val = 0.2
        else:
            obs_val = -0.6  # fairly strong penalty for wrong observation

    # Reward component (weight 0.3)
    diff = abs(predicted_reward - expected_reward)
    if diff <= tol:
        rew_val = 1.0
    elif diff <= 0.05:
        # small deviation -> neutral (no credit, no strong penalty)
        # map linearly [tol, 0.05] -> [1.0, 0.0]
        # but for simplicity give neutral 0.0
        rew_val = 0.0
    else:
        # reward far from expected -> negative
        rew_val = -1.0

    # Inventory component (weight 0.2)
    if len(inv_lines) == 0:
        inv_val = 1.0
    else:
        # If inventory diff explicitly removes a paper clip, that's a contradiction for these moves -> strong penalty
        lowered = [ln.lower() for ln in inv_lines]
        removes_paper_clip = any(ln.startswith("-") and "paper clip" in ln.lower() for ln in inv_lines)
        adds_paper_clip = any(ln.startswith("+") and "paper clip" in ln.lower() for ln in inv_lines)
        if removes_paper_clip:
            inv_val = -1.0
        else:
            # other inventory changes may be unrelated or permissible; be conservative and give mild penalty
            # but do not strongly penalize additions of other objects
            if adds_paper_clip:
                # adding a paper clip might be redundant but not necessarily impossible; mild penalty
                inv_val = -0.3
            else:
                inv_val = -0.4

    # Weighted sum and clamp
    score = 0.5 * obs_val + 0.3 * rew_val + 0.2 * inv_val
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return float(score)

# Rule 20
# Task group: determine if
def rule_reward(state, action, choice):
    """
    Returns a float in [-1, 1] according to the merged conservative rule.
    Triggers only for exact actions:
      - "open door to workshop"
      - "open door to hallway"   (requires state contains "The door is now open.")
      - "open door to kitchen"   (requires state contains "The door is now open.")
    Parses choice into predicted_observation, predicted_reward (float), predicted_inventory_diff (zero or more lines).
    """
    try:
        # Helper: get current action (prefer explicit arg, else extract from state)
        act = ""
        if action is not None and isinstance(action, str):
            act = action.strip()
        if not act:
            for line in (state or "").splitlines():
                if line.strip().startswith("current_step_action:"):
                    act = line.split("current_step_action:", 1)[1].strip()
                    break

        # Only apply to the three exact actions
        if act not in ("open door to workshop", "open door to hallway", "open door to kitchen"):
            return 0.0

        # For hallway and kitchen, require the precondition that the door was already opened earlier.
        expects_already_open = (act in ("open door to hallway", "open door to kitchen"))
        if expects_already_open:
            if "The door is now open." not in (state or ""):
                return 0.0

        # Parse the choice into three fields.
        # We search for headers anywhere in the choice; predicted_inventory_diff consumes remainder.
        lines = (choice or "").splitlines()
        po = None
        pr = None
        pid_lines = None
        i = 0
        n = len(lines)
        while i < n:
            line = lines[i]
            stripped = line.lstrip()
            if stripped.startswith("predicted_observation:"):
                # capture everything after the first colon (preserve trailing punctuation and case)
                po = line.split("predicted_observation:", 1)[1].lstrip()
                i += 1
                continue
            if stripped.startswith("predicted_reward:"):
                pr_str = line.split("predicted_reward:", 1)[1].strip()
                try:
                    pr = float(pr_str)
                except Exception:
                    return 0.0
                i += 1
                continue
            if stripped.startswith("predicted_inventory_diff:"):
                # capture remainder of this line after header and all subsequent lines
                rest = []
                trailing = line.split("predicted_inventory_diff:", 1)[1]
                if trailing is not None and trailing.strip() != "":
                    rest.append(trailing.rstrip())
                i += 1
                while i < n:
                    # include all subsequent lines (including blank ones) as part of inventory diff
                    rest.append(lines[i].rstrip())
                    i += 1
                # Normalize by removing purely-empty lines
                pid_lines = [l for l in rest if l.strip() != ""]
                break
            i += 1

        # All three fields must be present (inventory diff may be empty list)
        if po is None or pr is None or pid_lines is None:
            return 0.0

        # Normalized flags
        inventory_empty = (len(pid_lines) == 0)

        # Define expected values per action
        if act == "open door to workshop":
            expected_obs = "The door is now open."
            expected_reward = 0.0
            # Strict: exact match -> 1.0, else parsed but mismatch -> -1.0
            if po == expected_obs and abs(pr - expected_reward) < 1e-9 and inventory_empty:
                return 1.0
            else:
                return -1.0

        # For hallway and kitchen expect "already open"
        expected_obs = "The door is already open."
        expected_reward = 0.0

        if act == "open door to hallway":
            # Exact match -> 1.0
            if po == expected_obs and abs(pr - expected_reward) < 1e-9 and inventory_empty:
                return 1.0
            # Observation correct but reward mismatch (inventory must be empty) -> partial credit
            if po == expected_obs:
                if not inventory_empty:
                    return 0.0
                return 0.5
            # Observation incorrect -> strong penalty
            return -1.0

        if act == "open door to kitchen":
            # Exact match -> 1.0
            if po == expected_obs and abs(pr - expected_reward) < 1e-9 and inventory_empty:
                return 1.0
            # Otherwise weighted scoring (conservative)
            score = 0.0
            if po == expected_obs:
                score += 0.6
            else:
                score -= 0.6
            if abs(pr - expected_reward) < 1e-9:
                score += 0.3
            else:
                score -= 0.3
            if inventory_empty:
                score += 0.1
            else:
                score -= 0.1
            # Clip to [-1, 1]
            if score > 1.0:
                score = 1.0
            if score < -1.0:
                score = -1.0
            return float(score)

        # Fallback (should not reach)
        return 0.0

    except Exception:
        return 0.0

# Rule 21
# Task group: determine whether
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    try:
        # Extract action from argument or from state line 'current_step_action:'
        act = action.strip() if action is not None and action.strip() != "" else None
        if not act:
            for line in (state or "").splitlines():
                if line.strip().startswith("current_step_action:"):
                    act = line.split("current_step_action:", 1)[1].strip()
                    break
        if act is None:
            return 0.0

        # Only apply to these exact actions
        valid_actions = {
            "focus on green box": "You focus on the green box.",
            "focus on blue box":  "You focus on the blue box.",
            "focus on red box":   "You focus on the red box."
        }
        if act not in valid_actions:
            return 0.0

        expected_obs = valid_actions[act]
        expected_reward_val = 0.57

        # Parse the choice block for the three fields (robust to order)
        po = None
        pr = None
        pid_index = None
        lines = (choice or "").splitlines()

        # find indices and values
        for i, ln in enumerate(lines):
            if ln.startswith("predicted_observation:"):
                po = ln.split("predicted_observation:", 1)[1].strip()
            elif ln.startswith("predicted_reward:"):
                pr_str = ln.split("predicted_reward:", 1)[1].strip()
                try:
                    pr = float(pr_str)
                except Exception:
                    return 0.0
            elif ln.startswith("predicted_inventory_diff:"):
                pid_index = i
                # inventory diff is everything after this header (including any text on the header line)
                # capture rest starting at this line
                break

        # require all three fields present
        if po is None or pr is None or pid_index is None:
            return 0.0

        # Collect inventory diff lines: text after header on the same line plus any following lines
        inv_text_parts = []
        header_line = lines[pid_index]
        after_header = header_line.split("predicted_inventory_diff:", 1)[1]
        if after_header is not None and after_header.strip() != "":
            inv_text_parts.append(after_header)
        for ln in lines[pid_index+1:]:
            inv_text_parts.append(ln)
        # Consider inventory non-empty if any non-whitespace line present
        inv_nonblank = [ln for ln in inv_text_parts if ln.strip() != ""]
        inv_empty = (len(inv_nonblank) == 0)

        # If observation and empty inventory then score by reward closeness
        if po == expected_obs and inv_empty:
            diff = abs(pr - expected_reward_val)
            if diff <= 1e-9:
                return 1.0
            if diff <= 0.01:
                return 0.9
            if diff <= 0.1:
                return 0.5
            # observation correct and inventory empty but reward far off -> no credit
            return 0.0

        # If inventory changed when no change expected -> strong negative penalty
        if not inv_empty:
            return max(-1.0, -0.8)

        # Observation wrong but inventory empty -> moderate negative penalty
        return max(-1.0, -0.5)

    except Exception:
        # conservative fallback: do not apply rule on unexpected errors
        return 0.0

# Rule 22
# Task group: determine whether
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    import math
    try:
        # If action not provided, try to extract from state
        if not action or action.strip() == "":
            for line in state.splitlines():
                line = line.strip()
                if line.startswith('current_step_action:'):
                    action = line.split('current_step_action:', 1)[1].strip()
                    break

        # Only apply for the two exact target actions
        target_actions = ("pour jug into flower pot 5", "pour jug into flower pot 6")
        if action not in target_actions:
            return 0.0

        # Find header positions
        idx_obs = choice.find('predicted_observation:')
        idx_rew = choice.find('predicted_reward:')
        idx_inv = choice.find('predicted_inventory_diff:')
        if idx_obs == -1 or idx_rew == -1 or idx_inv == -1:
            return 0.0

        # Extract fields by slicing between headers
        obs_text = choice[idx_obs + len('predicted_observation:'):idx_rew].strip()
        rew_text = choice[idx_rew + len('predicted_reward:'):idx_inv].strip()
        inv_text = choice[idx_inv + len('predicted_inventory_diff:'):].strip()

        # Parse reward
        try:
            predicted_reward = float(rew_text)
        except Exception:
            return 0.0

        expected_reward = 0.0

        # Additional strict checks for action 6
        if action == "pour jug into flower pot 6":
            expected_obs = "You pour the contents of the jug into the flower pot 6."
            # observation must match exactly
            if obs_text != expected_obs:
                return 0.0
            # inventory diff must be empty (no content after header)
            if inv_text != "":
                return 0.0

        # For action 5 we only require successful parsing (done above); do not enforce obs/inv
        # Score according to reward correctness
        if math.isclose(predicted_reward, expected_reward, rel_tol=0.0, abs_tol=1e-9):
            return 1.0
        else:
            return -1.0

    except Exception:
        return 0.0

# Rule 23
# Task group: determine whether
def rule_reward(state, action, choice):
    """
    Returns a float in [-1, 1].
    Applies only when the current step action is exactly "wait1".
    Expects choice to contain lines with:
      predicted_observation: <text>
      predicted_reward: <float>
      predicted_inventory_diff: <maybe-empty content>
    The rule triggers only if:
      - predicted_observation == "You decide to wait for 1 iterations."
      - predicted_inventory_diff is empty (no non-whitespace content after the header)
    If triggered, returns 1.0 when predicted_reward ~= 0.0, -1.0 otherwise.
    Returns 0.0 for non-matching actions, non-matching observation/inventory, or parse errors.
    """
    import math

    try:
        # Determine action: prefer explicit parameter, otherwise extract from state
        act = action.strip() if action is not None and isinstance(action, str) and action.strip() != "" else None
        if not act and isinstance(state, str):
            for line in state.splitlines():
                line = line.strip()
                if line.startswith("current_step_action:"):
                    act = line.split("current_step_action:", 1)[1].strip()
                    break

        if act != "wait1":
            return 0.0

        if not isinstance(choice, str):
            return 0.0

        lines = choice.splitlines()

        pref_obs = "predicted_observation:"
        pref_rew = "predicted_reward:"
        pref_inv = "predicted_inventory_diff:"

        idx_obs = idx_rew = idx_inv = None
        # locate the first occurrences of each prefix (allow leading whitespace)
        for i, ln in enumerate(lines):
            s = ln.lstrip()
            if idx_obs is None and s.startswith(pref_obs):
                idx_obs = i
            if idx_rew is None and s.startswith(pref_rew):
                idx_rew = i
            if idx_inv is None and s.startswith(pref_inv):
                idx_inv = i
            if idx_obs is not None and idx_rew is not None and idx_inv is not None:
                break

        if idx_obs is None or idx_rew is None or idx_inv is None:
            return 0.0

        # Extract observation (remainder of the obs line)
        obs_line = lines[idx_obs]
        obs_text = obs_line.split(pref_obs, 1)[1].strip() if pref_obs in obs_line else None
        if obs_text is None:
            return 0.0

        expected_obs = "You decide to wait for 1 iterations."
        if obs_text != expected_obs:
            return 0.0

        # Extract reward (remainder of the reward line)
        rew_line = lines[idx_rew]
        rew_text = rew_line.split(pref_rew, 1)[1].strip() if pref_rew in rew_line else None
        if rew_text is None:
            return 0.0
        try:
            pred_reward = float(rew_text)
        except Exception:
            return 0.0

        # Extract inventory diff content: after the inventory header line, include that line's remainder and any following lines
        inv_line = lines[idx_inv]
        inv_after = inv_line.split(pref_inv, 1)[1].strip() if pref_inv in inv_line else ""
        tail = ""
        if idx_inv + 1 < len(lines):
            tail = "\n".join(lines[idx_inv + 1 : ])
        inv_all = (inv_after + ("\n" + tail if tail else "")).strip()
        # Require inventory diff to be empty (no non-whitespace content)
        if inv_all != "":
            return 0.0

        # Reward must be exactly 0.0 within a small tolerance
        if math.isclose(pred_reward, 0.0, abs_tol=1e-6):
            return 1.0
        else:
            return -1.0

    except Exception:
        # On any unexpected error, do not apply the rule
        return 0.0

# Rule 24
# Task group: determine which
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    try:
        # Determine which action to check: prefer explicit action argument; otherwise extract from state
        check_action = ""
        if action is not None:
            check_action = action.strip()
        if check_action == "":
            # search for a line starting with "current_step_action:" (case-insensitive)
            for line in state.splitlines():
                if line.strip().lower().startswith("current_step_action:"):
                    # take text after the first ':' and strip whitespace (preserve case of the action text)
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        check_action = parts[1].strip()
                    break

        # Only apply this rule to the two exact action strings
        if check_action not in ("focus on inclined plane f", "focus on inclined plane d"):
            return 0.0

        # Parse the choice into the three expected labeled parts.
        # Be permissive about spacing but require the labels.
        lines = [ln.rstrip("\n") for ln in choice.splitlines()]

        obs_line = None
        rew_line = None
        inv_index = None
        inv_first = ""
        for i, ln in enumerate(lines):
            if ln.startswith("predicted_observation:"):
                obs_line = ln.split(":", 1)[1].lstrip()
            elif ln.startswith("predicted_reward:"):
                rew_line = ln.split(":", 1)[1].lstrip()
            elif ln.startswith("predicted_inventory_diff:"):
                inv_index = i
                inv_first = ln.split(":", 1)[1].lstrip()
                break  # inventory is last of the three labels; stop scanning further for labels

        # If any required field is missing, be conservative and do not apply the rule
        if obs_line is None or rew_line is None or inv_index is None:
            return 0.0

        # Parse reward as float
        try:
            predicted_reward = float(rew_line)
        except Exception:
            return 0.0

        # Parse inventory: allow content on the same line after the colon and any subsequent non-empty lines
        inventory_entries = []
        if inv_first:
            inventory_entries.append(inv_first)
        for ln in lines[inv_index+1:]:
            # treat non-empty subsequent lines as inventory entries (conservative but only applied for the two actions)
            if ln.strip() != "":
                inventory_entries.append(ln.strip())
            else:
                # skip blank lines
                continue

        # Determine expected observation based on the exact action
        if check_action == "focus on inclined plane f":
            expected_obs = "You focus on the inclined plane F."
        else:  # "focus on inclined plane d"
            expected_obs = "You focus on the inclined plane D."

        expected_reward = 0.5
        # If observation matches exactly and there are no inventory changes
        if obs_line == expected_obs and len(inventory_entries) == 0:
            # exact reward -> full score
            if abs(predicted_reward - expected_reward) < 1e-6:
                return 1.0
            else:
                # observation & inventory OK but reward differs -> partial credit
                return 0.5
        else:
            # observation incorrect or inventory non-empty -> penalize
            return -1.0

    except Exception:
        # On unexpected errors, be conservative and do not apply the rule
        return 0.0

# Rule 25
# Task group: find a
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    import math

    def extract_action_from_state(s):
        if not isinstance(s, str):
            return ""
        for line in s.splitlines():
            line = line.strip()
            if line.startswith("current_step_action:"):
                return line.split("current_step_action:", 1)[1].strip()
        return ""

    # Ensure we have an action string
    act = action.strip() if isinstance(action, str) and action.strip() != "" else extract_action_from_state(state or "")
    if not act:
        return 0.0

    # Only apply to the specified pattern and allowed colors
    prefix = "move egg parrot egg in inventory to "
    suffix = " box"
    if not (act.startswith(prefix) and act.endswith(suffix)):
        return 0.0
    color = act[len(prefix):-len(suffix)]
    allowed_colors = {"green", "blue", "purple", "orange"}
    if color not in allowed_colors:
        return 0.0

    # choice must be a string
    if not isinstance(choice, str):
        return 0.0
    lines = choice.splitlines()

    # Find header indices for the three required fields
    obs_idx = rew_idx = inv_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("predicted_observation:") and obs_idx is None:
            obs_idx = i
        elif ln.startswith("predicted_reward:") and rew_idx is None:
            rew_idx = i
        elif ln.startswith("predicted_inventory_diff:") and inv_idx is None:
            inv_idx = i

    # Require all three headers and correct ordering
    if obs_idx is None or rew_idx is None or inv_idx is None:
        return 0.0
    if not (obs_idx < rew_idx < inv_idx):
        return 0.0

    # Extract observation text (rest of the obs header line)
    try:
        predicted_observation = lines[obs_idx].split("predicted_observation:", 1)[1].strip()
    except Exception:
        return 0.0

    # Extract reward float
    try:
        rew_text = lines[rew_idx].split("predicted_reward:", 1)[1].strip()
        predicted_reward = float(rew_text)
    except Exception:
        return 0.0

    # Extract inventory diff lines: include any non-empty trailing text on header line, then subsequent non-empty lines
    inv_header_tail = lines[inv_idx].split("predicted_inventory_diff:", 1)[1].strip()
    inv_lines = []
    if inv_header_tail:
        inv_lines.append(inv_header_tail)
    for ln in lines[inv_idx + 1:]:
        s = ln.strip()
        if s != "":
            inv_lines.append(s)
    # Normalize inventory lines
    inv_lines = [ln.rstrip() for ln in inv_lines]

    # Checks
    # Observation expectation
    expected_exact_obs = f"You move the parrot to the {color} box."
    if color == "purple":
        low_obs = predicted_observation.lower()
        obs_ok = ("parrot" in low_obs) and ("purple box" in low_obs)
    else:
        obs_ok = (predicted_observation == expected_exact_obs)

    # Reward check (allow small tolerance)
    reward_ok = abs(predicted_reward - 0.17) <= 1e-3

    # Inventory checks
    inv_has_minus_parrot = any(ln == "- a parrot egg" for ln in inv_lines)
    inv_has_plus_parrot = any((ln.lstrip().startswith("+") or ln.lstrip().startswith("+=")) and ("parrot" in ln.lower()) for ln in inv_lines)

    # Strong penalty for explicit incorrect addition of the parrot
    if inv_has_plus_parrot:
        return -1.0

    # Full correct continuation
    if obs_ok and reward_ok and inv_has_minus_parrot:
        return 1.0

    # Observation and reward match but removal missing -> penalize moderately
    if obs_ok and reward_ok and not inv_has_minus_parrot:
        return -0.5

    # Observation and removal present but reward wrong -> give partial positive
    if obs_ok and inv_has_minus_parrot and not reward_ok:
        return 0.5

    # Observation only
    if obs_ok and (not reward_ok) and (not inv_has_minus_parrot):
        return 0.2

    # Parsed but inconsistent with expectations -> conservative negative
    return -0.5

# Rule 26
# Task group: find a
def rule_reward(state, action, choice):
    """
    Returns a float in [-1, 1] according to the merged conservative rule described above.
    """
    import re

    try:
        # Helper: extract action from parameter or from state current_step_action
        act = action.strip() if action is not None and str(action).strip() != "" else None
        if not act:
            for line in (state or "").splitlines():
                if line.strip().startswith("current_step_action:"):
                    act = line.split("current_step_action:", 1)[1].strip()
                    break
        if not act:
            return 0.0

        act_lower = act.lower()

        # Recognize supported action types
        move_match = re.search(r'^move flower pot (\d+).*in inventory to (?:the )?([a-z]+) box$', act_lower)
        pickup_match = re.search(r'^pick up flower pot (\d+)$', act_lower)
        open_kitchen = (act_lower == "open door to kitchen")
        go_kitchen = (act_lower == "go to kitchen")

        if not (move_match or pickup_match or open_kitchen or go_kitchen):
            # action not relevant for this rule
            return 0.0

        # Parse choice into three required fields
        lines = choice.splitlines() if choice is not None else []
        # Find the three headers (allow whitespace before header)
        def find_line_index(prefix):
            for i, ln in enumerate(lines):
                if ln.strip().startswith(prefix):
                    return i
            return None

        idx_obs = find_line_index("predicted_observation:")
        idx_reward = find_line_index("predicted_reward:")
        idx_inv = find_line_index("predicted_inventory_diff:")

        if idx_obs is None or idx_reward is None or idx_inv is None:
            return 0.0

        # Extract predicted_observation (text after the first colon on that line)
        pred_obs = lines[idx_obs].split("predicted_observation:", 1)[1].strip()

        # Extract predicted_reward
        rew_text = lines[idx_reward].split("predicted_reward:", 1)[1].strip()
        try:
            pred_reward = float(rew_text)
        except:
            return 0.0

        # Extract inventory diff lines: content on same line after colon plus subsequent lines that start with + or -
        inv_lines = []
        inv_after = lines[idx_inv].split("predicted_inventory_diff:", 1)[1].strip()
        if inv_after != "":
            # if the same-line content looks like one or more inventory lines, treat as one entry
            inv_lines.append(inv_after)
        # collect following lines that begin with + or -
        for j in range(idx_inv + 1, len(lines)):
            if lines[j].strip().startswith(('+', '-')):
                inv_lines.append(lines[j].strip())
            else:
                # stop at first non-inventory-looking line (conservative)
                break

        # Normalize lowercase for many checks
        pred_obs_l = pred_obs.lower()
        inv_l = [l.lower() for l in inv_lines]

        # Tolerances
        small_tol = 1e-3
        partial_tol = 0.05

        # --- MOVE TO BOX handling ---
        if move_match:
            pot_num = move_match.group(1)
            color = move_match.group(2)
            # Observation should mention moving that pot to that color box
            obs_good = ("flower pot " + pot_num) in pred_obs_l and color + " box" in pred_obs_l and ("move" in pred_obs_l or "moved" in pred_obs_l)
            # Inventory should contain a removal '-' of that pot
            has_minus_pot = any(l.startswith("-") and ("flower pot " + pot_num) in l for l in inv_l)
            # Should not add the same pot back (inconsistent)
            has_plus_same_pot = any(l.startswith("+") and ("flower pot " + pot_num) in l for l in inv_l)
            # Expected reward around 0.17
            reward_close = abs(pred_reward - 0.17) <= small_tol
            reward_near = abs(pred_reward - 0.17) <= partial_tol

            # Full credit: explicit observation, minus present, reward near exact, and not adding same item
            if obs_good and has_minus_pot and reward_close and not has_plus_same_pot:
                return 1.0
            # High partial if semantics correct and minus present, reward slightly off
            if obs_good and has_minus_pot and reward_near and not has_plus_same_pot:
                return 0.8
            # Partial if semantics correct but reward deviates more (still reasonable)
            if obs_good and has_minus_pot and not has_plus_same_pot:
                return 0.5
            # Strong negative if model added the same pot back or failed to remove it while claiming a move
            if has_plus_same_pot or (("move" in pred_obs_l or "moved" in pred_obs_l) and not has_minus_pot):
                return -1.0
            # Otherwise parsed but inconsistent
            return -0.5

        # --- PICK UP handling ---
        if pickup_match:
            pot_num = pickup_match.group(1)
            # Expect observation describing moving pot into inventory
            expected_obs_phrase = ("you move the flower pot " + pot_num + " to the inventory").lower()
            obs_exact = pred_obs_l.strip().startswith(expected_obs_phrase) or expected_obs_phrase in pred_obs_l
            # Inventory must include a '+' adding that pot
            has_plus_pot = any(l.startswith("+") and ("flower pot " + pot_num) in l for l in inv_l)
            # Expected reward ~0.08
            reward_exact = abs(pred_reward - 0.08) <= small_tol

            if has_plus_pot and obs_exact and reward_exact:
                return 1.0
            if has_plus_pot:
                # inventory correct but something else off -> partial positive depending on which field mismatched
                if obs_exact and not reward_exact:
                    return 0.6
                if reward_exact and not obs_exact:
                    return 0.5
                return 0.3
            # If inventory lines present but none match expected, penalize strongly
            if len(inv_l) > 0 and not has_plus_pot:
                return -0.8
            # parsed but no inventory info -> conservative 0.0
            return 0.0

        # --- OPEN DOOR TO KITCHEN handling ---
        if open_kitchen:
            # Expect reward ~0.0 and inventory diff contains both a '-' and a '+' referring to same pot number (replacement)
            if abs(pred_reward - 0.0) > partial_tol:
                # parsed but reward not near expected
                return -0.5
            # find any pot numbers present in minus and plus lines
            minus_pots = set()
            plus_pots = set()
            pot_re = re.compile(r'flower pot (\d+)')
            for l in inv_l:
                m = pot_re.search(l)
                if not m:
                    continue
                num = m.group(1)
                if l.startswith("-"):
                    minus_pots.add(num)
                if l.startswith("+"):
                    plus_pots.add(num)
            # find intersection (same pot replaced)
            common = minus_pots.intersection(plus_pots)
            if common and ("kitchen" in pred_obs_l or "open" in pred_obs_l):
                return 1.0
            # parsed but inventory transformation missing -> conservative penalty
            return -0.5

        # --- GO TO KITCHEN handling ---
        if go_kitchen:
            # Expect observation mentions moving to the kitchen and reward ~0.0
            if "move to the kitchen" in pred_obs_l or "you move to the kitchen" in pred_obs_l or pred_obs_l.strip().startswith("you move to the kitchen"):
                pass  # ok
            elif "you move to the kitchen" not in pred_obs_l and "move to the kitchen" not in pred_obs_l and "kitchen" not in pred_obs_l:
                # not the expected observation
                # but we will still check inventory; if inventory wrong we penalize
                pass
            if abs(pred_reward - 0.0) > partial_tol:
                return -0.5
            # For go to kitchen we require that inventory diff contains both a '+' and a '-' referring to the same pot (replacement)
            minus_pots = set()
            plus_pots = set()
            pot_re = re.compile(r'flower pot (\d+)')
            for l in inv_l:
                m = pot_re.search(l)
                if not m:
                    continue
                num = m.group(1)
                if l.startswith("-"):
                    minus_pots.add(num)
                if l.startswith("+"):
                    plus_pots.add(num)
            if len(minus_pots) == 0 or len(plus_pots) == 0:
                return -1.0
            if minus_pots.intersection(plus_pots):
                return 1.0
            # inventory diff missing matching update -> penalize
            return -1.0

        # Fallback conservative return
        return 0.0

    except Exception:
        return 0.0

# Rule 27
# Task group: find the
def rule_reward(state, action, choice):
    """
    Returns a float in [-1, 1] according to the rule described above.
    """
    try:
        # Determine action: prefer provided action, otherwise extract from state
        act = (action or "").strip()
        if not act:
            for line in state.splitlines()[::-1]:
                if line.strip().startswith("current_step_action:"):
                    act = line.split("current_step_action:", 1)[1].strip()
                    break
        if not act:
            return 0.0

        # Only apply to the two exact action strings we care about
        if act not in ("focus on baby baby elephant", "focus on baby baby hedgehog"):
            return 0.0

        # Expected mappings
        expected = {
            "focus on baby baby elephant": ("You focus on the baby elephant.", 0.50),
            "focus on baby baby hedgehog": ("You focus on the baby hedgehog.", 0.17),
        }
        expected_obs, expected_reward = expected[act]

        # Parse choice: find prefixes on their own lines
        po_prefix = "predicted_observation:"
        pr_prefix = "predicted_reward:"
        pid_prefix = "predicted_inventory_diff:"

        lines = choice.splitlines()
        i_po = i_pr = i_pid = None
        for i, ln in enumerate(lines):
            if i_po is None and ln.startswith(po_prefix):
                i_po = i
            if i_pr is None and ln.startswith(pr_prefix):
                i_pr = i
            if i_pid is None and ln.startswith(pid_prefix):
                i_pid = i
            # stop early if all found
            if i_po is not None and i_pr is not None and i_pid is not None:
                break

        # Require all three fields present
        if i_po is None or i_pr is None or i_pid is None:
            return 0.0

        # Extract observation string (rest of the po line)
        predicted_observation = lines[i_po][len(po_prefix):].strip()

        # Extract reward and parse float
        pr_text = lines[i_pr][len(pr_prefix):].strip()
        try:
            predicted_reward = float(pr_text)
        except Exception:
            return 0.0

        # Inventory diff: all lines after the pid line
        inv_lines = lines[i_pid+1:] if i_pid + 1 < len(lines) else []
        # Inventory considered empty if there are no non-blank lines after the header
        inventory_empty = all(ln.strip() == "" for ln in inv_lines)

        # Require exact observation match and empty inventory to apply scoring
        if predicted_observation == expected_obs and inventory_empty:
            # reward must equal expected (use small tolerance)
            if abs(predicted_reward - expected_reward) < 1e-6:
                return 1.0
            else:
                # observation & inventory correct but reward differs -> moderate penalty
                return -0.5
        else:
            # Do not penalize other mismatches to avoid false positives
            return 0.0

    except Exception:
        return 0.0

# Rule 28
# Task group: focus on
def rule_reward(state, action, choice):
    """
    Returns a float in [-1,1] per the merged conservative rule.
    Triggers only for the exact set of target actions (or when the state's
    current_step_action: line matches one of them). Parsing failures -> 0.0.
    """
    # target action specifications
    specs = {
        "focus on adult moth in outside": {
            "expected_obs": "You focus on the adult butterfly.",
            "expected_reward": 0.05,
            "style": "moth"  # tolerant small deviations
        },
        "wait1": {
            "special": "wait1"  # special formatting check for predicted_reward text == "0.00"
        },
        "focus on adult frog in outside": {
            "expected_obs": "You focus on the adult frog.",
            "expected_reward": 0.08,
            "style": "frog"  # small tolerance -> full credit, otherwise moderate penalty
        },
        "go to greenhouse": {
            "expected_obs": "You move to the greenhouse.",
            "expected_reward": 0.05,
            "strict_reward": True  # require essentially exact reward
        },
        "focus on adult blue jay in outside": {
            "expected_obs": "You focus on the adult blue jay.",
            "expected_reward": 0.05,
            "strict_reward": True
        },
        "focus on cherry tree in the reproducing stage in self watering flower pot 4": {
            "expected_obs": "You focus on the cherry tree.",
            "expected_reward": 0.09,
            "style": "cherry"  # linear mapping score = 1 - 4*diff
        },
        "focus on grapefruit tree in the reproducing stage in self watering flower pot 5": {
            "expected_obs": "You focus on the grapefruit tree.",
            "expected_reward": 0.09,
            "style": "grapefruit"  # moderate penalty for differing reward
        }
    }

    # Helper: extract current action if none provided or blank
    act = (action or "").strip()
    if not act:
        for line in (state or "").splitlines():
            if line.strip().startswith("current_step_action:"):
                parts = line.split(":", 1)
                if len(parts) > 1:
                    act = parts[1].strip()
                break

    if act not in specs:
        return 0.0

    spec = specs[act]

    # Special-case for wait1: require exact textual predicted_reward of "0.00"
    if spec.get("special") == "wait1":
        pred_reward_text = None
        for ln in (choice or "").splitlines():
            if ln.strip().startswith("predicted_reward:"):
                parts = ln.split(":", 1)
                pred_reward_text = parts[1].strip() if len(parts) > 1 else ""
                break
        if pred_reward_text is None:
            return 0.0
        return 1.0 if pred_reward_text == "0.00" else -1.0

    # Otherwise parse the three required fields
    lines = (choice or "").splitlines()
    obs_line = None
    rew_line = None
    inv_index = None
    for i, ln in enumerate(lines):
        s = ln.lstrip()
        if s.startswith("predicted_observation:") and obs_line is None:
            obs_line = ln
        elif s.startswith("predicted_reward:") and rew_line is None:
            rew_line = ln
        elif s.startswith("predicted_inventory_diff:") and inv_index is None:
            inv_index = i
            # do not break; ensure obs and reward found too

    if obs_line is None or rew_line is None or inv_index is None:
        return 0.0

    # Extract observation text
    try:
        pred_obs = obs_line.split("predicted_observation:", 1)[1].strip()
    except Exception:
        return 0.0

    # Extract reward text and float
    rew_text = rew_line.split("predicted_reward:", 1)[1].strip()
    try:
        pred_reward = float(rew_text)
    except Exception:
        return 0.0

    # Inventory diff lines are everything after the inventory header line
    inv_lines = [ln for ln in lines[inv_index + 1:]]
    inv_nonempty = any((ln.strip() != "") for ln in inv_lines)

    # Check observation correctness (exact match after stripping)
    expected_obs = spec.get("expected_obs")
    if pred_obs != expected_obs:
        return -1.0

    # If inventory changed when it should be empty => strong penalty
    if inv_nonempty:
        return -0.8

    expected_reward = spec.get("expected_reward")
    # tiny tolerance for exact comparisons
    tiny_tol = 1e-9

    # Strict reward actions: require essentially exact numeric match
    if spec.get("strict_reward"):
        return 1.0 if abs(pred_reward - expected_reward) <= tiny_tol else -1.0

    # Per-style scoring
    style = spec.get("style", "default")
    diff = abs(pred_reward - expected_reward)

    if style == "moth":
        # prefer exact, small deviations get partial credit, large deviations small positive
        if diff <= tiny_tol:
            return 1.0
        if diff < 0.05:
            score = 0.6 + 0.4 * (1.0 - (diff / 0.05))
            return max(-1.0, score)
        return 0.2

    if style == "frog":
        # exact within tiny tol -> full credit; otherwise moderate penalty
        tol = 1e-6
        if diff <= tol:
            return 1.0
        return -0.5

    if style == "cherry":
        # linear mapping: score = 1 - 4*diff, clamped to [-1,1]
        score = 1.0 - 4.0 * diff
        if score > 1.0:
            score = 1.0
        if score < -1.0:
            score = -1.0
        return score

    if style == "grapefruit":
        # accept exact, otherwise moderate penalty
        if diff <= tiny_tol:
            return 1.0
        return -0.5

    # default conservative behavior: exact -> 1.0, small deviation -> partial, large -> -0.5
    if diff <= tiny_tol:
        return 1.0
    if diff < 0.05:
        score = 0.6 + 0.4 * (1.0 - (diff / 0.05))
        return max(-1.0, score)
    return -0.5

# Rule 29
# Task group: freeze
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    import re
    def extract_action_from_state(s):
        for line in s.splitlines():
            if line.strip().startswith("current_step_action:"):
                return line.split("current_step_action:",1)[1].strip().splitlines()[0].strip()
        return ""

    # Normalize action preference: explicit arg wins, else extract from state
    act = ""
    if action is not None:
        act = action.strip()
    if not act:
        act = extract_action_from_state(state)

    # Only handle a small set of exact actions to avoid false positives
    if act not in {
        "wait",
        "wait1",
        "use thermometer in inventory on ice cream",
        "pick up metal pot",
        "go to outside"
    }:
        return 0.0

    # Minimal sanity: choice must exist
    if not choice:
        return 0.0

    # Parse predicted_... fields robustly
    lines = choice.splitlines()
    po = None
    pr = None
    # collect inventory diff lines that start with + or -
    pid = []
    mode = None
    for i, ln in enumerate(lines):
        ln_stripped = ln.rstrip("\n")
        if ln_stripped.startswith("predicted_observation:"):
            po = ln_stripped.split("predicted_observation:",1)[1].lstrip()
            mode = "obs"
            continue
        if ln_stripped.startswith("predicted_reward:"):
            pr_text = ln_stripped.split("predicted_reward:",1)[1].strip()
            try:
                pr = float(pr_text)
            except:
                # Can't parse reward -> give up
                return 0.0
            mode = "rew"
            continue
        if ln_stripped.startswith("predicted_inventory_diff:"):
            # the header line may be followed by +/- lines; also the same line may contain content
            after = ln_stripped.split("predicted_inventory_diff:",1)[1].strip()
            if after:
                # if remainder begins with + or - treat it as inventory line (otherwise ignore)
                if after.startswith("+") or after.startswith("-"):
                    pid.append(after)
            mode = "inv"
            continue
        # If in inventory mode, collect +/- lines
        if mode == "inv":
            s = ln_stripped.strip()
            if s.startswith("+") or s.startswith("-"):
                pid.append(s)

    # Ensure at least observation and reward parsed for decisions (inventory diff may be empty)
    if po is None or pr is None:
        return 0.0

    # Helper tolerances
    def close(a, b, tol=1e-6):
        return abs(a - b) <= tol

    # Action-specific checks and scoring
    # 1) wait (ice-cream freezing context) -> expect specific wait observation, reward ~0.18, no inventory changes
    if act == "wait":
        # avoid misapplying to unrelated tasks: require task mentions ice cream/freezer/freeze
        ctx = state.lower()
        if not ("ice cream" in ctx or "freeze" in ctx or "freezer" in ctx):
            return 0.0
        expected_obs = "You decide to wait for 10 iterations."
        expected_reward = 0.18

        # exact good continuation
        if po == expected_obs and close(pr, expected_reward) and len(pid) == 0:
            return 1.0
        # observation correct but inventory changed -> strong penalty
        if po == expected_obs and len(pid) > 0:
            return -0.8
        # observation correct but reward wrong -> strong penalty
        if po == expected_obs and not close(pr, expected_reward):
            return -1.0
        # reward correct but observation different -> modest partial credit
        if close(pr, expected_reward) and po != expected_obs:
            return 0.5
        # well-parsed but wrong continuation -> moderate penalty
        return -0.5

    # 2) wait1 -> expect "You decide to wait for 1 iterations.", reward 0.0, empty inventory
    if act == "wait1":
        expected_obs = "You decide to wait for 1 iterations."
        expected_reward = 0.0
        if po == expected_obs and close(pr, expected_reward) and len(pid) == 0:
            return 1.0
        if po == expected_obs and len(pid) > 0:
            return -1.0
        if po == expected_obs and not close(pr, expected_reward):
            return 0.5
        if close(pr, expected_reward) and po != expected_obs:
            return -0.5
        return 0.0

    # 3) use thermometer in inventory on ice cream
    if act == "use thermometer in inventory on ice cream":
        # require ice-cream context to avoid misfires
        ctx = state.lower()
        if not ("ice cream" in ctx or "freezer" in ctx or "freeze" in ctx):
            return 0.0
        expected_obs = "the thermometer measures a temperature of -7 degrees celsius"
        expected_reward = 0.0
        # inventory changes not expected -> penalize strongly if present
        if len(pid) > 0:
            return -1.0
        # exact match
        if po == expected_obs and close(pr, expected_reward):
            return 1.0
        # correct numeric behavior (reports a degrees celsius) with near-zero reward -> partial credit
        low = po.lower()
        deg_match = re.search(r"(-?\d+(\.\d+)?)\s*degrees\s*celsius", low)
        if ("thermometer" in low or "thermometer" in po.lower()) and deg_match and abs(pr - expected_reward) < 0.05:
            return 0.5
        # otherwise incorrect observation -> moderate penalty
        return -0.5

    # 4) pick up metal pot -> expect exact observation, reward 0.0, and exact inventory diff replacement of thermometer readings
    if act == "pick up metal pot":
        # require some pot mention in state for safety
        ctx = state.lower()
        if "metal pot" not in ctx and "pot" not in ctx:
            return 0.0
        expected_obs = "You move the metal pot to the inventory."
        expected_reward = 0.0
        expected_pid_set = {
            "+ a thermometer, currently reading a temperature of 46 degrees celsius",
            "- a thermometer, currently reading a temperature of 30 degrees celsius"
        }
        pid_set = set([p.strip() for p in pid])
        # exact match required for full credit
        if po == expected_obs and close(pr, expected_reward) and pid_set == expected_pid_set and len(pid_set) == len(expected_pid_set):
            return 1.0
        # if parsed but any element deviates, give strong penalty (this is a precise inventory-change step)
        return -1.0

    # 5) go to outside -> expect move observation mentioning outside, reward 0.0, and inventory diffs replacing a thermometer reading 207->244
    if act == "go to outside":
        # require "outside" in action already; make sure state mentions thermometer to avoid misfires
        ctx = state.lower()
        if ("thermometer" not in ctx) and ("thermometers" not in ctx):
            return 0.0
        expected_reward = 0.0
        # inventory diffs expected
        expected_pid_set = {
            "+ a thermometer, currently reading a temperature of 244 degrees celsius",
            "- a thermometer, currently reading a temperature of 207 degrees celsius"
        }
        pid_set = set([p.strip() for p in pid])
        # observation should mention outside/go/goes/move to be robust
        low_po = po.lower()
        if (("outside" in low_po or "go" in low_po or "move" in low_po) and close(pr, expected_reward) and pid_set == expected_pid_set):
            return 1.0
        # if inventory diffs or reward mismatch while otherwise parsed, penalize strongly
        if pid and (pid_set != expected_pid_set or not close(pr, expected_reward)):
            return -1.0
        # if observation doesn't mention outside but other parts are OK -> mild penalty
        if pid_set == expected_pid_set and close(pr, expected_reward):
            return 0.5
        return -0.5

    # Default fallback: no judgment
    return 0.0

# Rule 30
# Task group: grow a
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    import re
    def clamp(x, lo=-1.0, hi=1.0):
        return max(lo, min(hi, x))

    # Determine current action (allow extracting from state if action empty)
    act = (action or "").strip()
    if act == "":
        m = re.search(r"current_step_action:\s*(.+)", state or "")
        if not m:
            return 0.0
        act = m.group(1).strip()

    # Only apply for the specific action "0"
    if act != "0":
        return 0.0

    if choice is None:
        return 0.0

    # Split lines and locate labeled fields
    lines = choice.splitlines()
    try:
        i_obs = next(i for i, l in enumerate(lines) if l.startswith("predicted_observation:"))
        i_r = next(i for i, l in enumerate(lines) if l.startswith("predicted_reward:"))
        i_inv = next(i for i, l in enumerate(lines) if l.startswith("predicted_inventory_diff:"))
    except StopIteration:
        return 0.0

    # Labels must appear in order
    if not (i_obs < i_r < i_inv):
        return 0.0

    def after_colon(s):
        parts = s.split(":", 1)
        return parts[1].lstrip() if len(parts) > 1 else ""

    obs = after_colon(lines[i_obs])
    reward_str = after_colon(lines[i_r])

    # Inventory diff is everything after the inventory label line
    inv_lines = lines[i_inv+1:] if i_inv+1 <= len(lines) else []
    inv_text = "\n".join(inv_lines).strip()

    # Only apply when observation exactly matches expected and there are no inventory changes
    expected_obs = "You focus on the cherry."
    if obs != expected_obs:
        return 0.0
    if inv_text != "":
        return 0.0

    # Parse predicted_reward
    try:
        pred_reward = float(reward_str)
    except Exception:
        return 0.0

    # Conservative linear score: exact match -> 1.0, decreases with distance from 0.50
    score = 1.0 - 2.0 * abs(pred_reward - 0.50)
    return clamp(score)

# Rule 31
# Task group: grow a
def rule_reward(state, action, choice):
    import re
    def clamp(x, lo=-1.0, hi=1.0):
        return max(lo, min(hi, x))

    try:
        # Determine action: prefer explicit param, otherwise extract from state
        act = (action or "").strip()
        if not act:
            m = re.search(r"current_step_action:\s*(.+)", state)
            if m:
                act = m.group(1).strip()
        if not act:
            return 0.0

        # Only apply for these exact action strings
        if act not in ("pour jug into flower pot 1", "pour jug into flower pot 3"):
            return 0.0

        # Expect choice to contain three headers
        if not choice:
            return 0.0
        lines = choice.splitlines()

        obs_idx = None
        rew_idx = None
        inv_idx = None
        for i, ln in enumerate(lines):
            if ln.startswith("predicted_observation:"):
                obs_idx = i
            elif ln.startswith("predicted_reward:"):
                rew_idx = i
            elif ln.startswith("predicted_inventory_diff:"):
                inv_idx = i

        # Require all three headers to be present to avoid false positives
        if obs_idx is None or rew_idx is None or inv_idx is None:
            return 0.0

        # Extract predicted_observation (rest of that header line)
        predicted_observation = lines[obs_idx].split("predicted_observation:", 1)[1].strip()

        # Extract predicted_reward (rest of that header line) and parse float
        rew_text = lines[rew_idx].split("predicted_reward:", 1)[1].strip()
        try:
            predicted_reward = float(rew_text)
        except Exception:
            return 0.0

        # Collect inventory diff lines: any non-empty content on the header line and
        # subsequent lines that look like diffs (start with + or -) are considered inventory changes.
        inv_header_tail = lines[inv_idx].split("predicted_inventory_diff:", 1)[1].strip()
        inv_lines = []
        if inv_header_tail:
            inv_lines.append(inv_header_tail)
        for ln in lines[inv_idx + 1:]:
            # treat + or - prefixed lines as inventory entries; stop on first non-diff line to be conservative
            if ln.strip().startswith(("+", "-")):
                inv_lines.append(ln.strip())
            elif ln.strip() == "":
                continue
            else:
                break

        # Expected canonical observations and rewards per action
        expected_map = {
            "pour jug into flower pot 1": ("You pour the contents of the jug into the flower pot 1.", 0.12),
            "pour jug into flower pot 3": ("You pour the contents of the jug into the flower pot 3.", 0.14),
        }
        expected_obs, expected_reward = expected_map[act]

        # Tolerance for numeric comparison
        tol = 1e-6

        # Scoring logic (only after successful parsing)
        # If observation wrong -> strong penalty
        if predicted_observation != expected_obs:
            return clamp(-1.0)

        # Observation correct:
        if len(inv_lines) != 0:
            # Invented inventory changes are inconsistent with these transitions -> penalize
            return clamp(-0.8)

        # Observation correct and no inventory diffs: check reward
        if abs(predicted_reward - expected_reward) <= tol:
            return clamp(1.0)
        else:
            # Correct textual continuation but numeric reward differs -> conservative penalty
            return clamp(-0.5)

    except Exception:
        return 0.0

# Rule 32
# Task group: grow a
def rule_reward(state, action, choice):
    import math

    # Helper: extract current_step_action from state if action not provided or empty
    if not action or action.strip() == "":
        for line in reversed(state.splitlines()):
            line = line.strip()
            if line.startswith("current_step_action:"):
                action = line.split("current_step_action:", 1)[1].strip()
                break

    # Only apply for the explicit wait1 action
    if action != "wait1":
        return 0.0

    # Split choice and locate headers
    lines = choice.splitlines()
    try:
        obs_idx = next(i for i, l in enumerate(lines) if l.startswith("predicted_observation:"))
        rew_idx = next(i for i, l in enumerate(lines) if l.startswith("predicted_reward:"))
        inv_idx = next(i for i, l in enumerate(lines) if l.startswith("predicted_inventory_diff:"))
    except StopIteration:
        # Missing required fields -> do not apply
        return 0.0

    # Extract observation text
    predicted_observation = lines[obs_idx].split("predicted_observation:", 1)[1].lstrip()

    # Extract reward and parse as float
    reward_str = lines[rew_idx].split("predicted_reward:", 1)[1].strip()
    try:
        predicted_reward = float(reward_str)
    except Exception:
        return 0.0

    # Collect inventory-diff lines: remainder of inv header plus any following lines
    inv_remainder = lines[inv_idx].split("predicted_inventory_diff:", 1)[1].lstrip()
    inv_lines = []
    if inv_remainder != "":
        inv_lines.append(inv_remainder)
    for j in range(inv_idx + 1, len(lines)):
        inv_lines.append(lines[j].rstrip())
    # Normalize: keep only non-empty lines
    inv_nonempty = [l for l in inv_lines if l.strip() != ""]

    # Expected values
    expected_obs = "You decide to wait for 1 iterations."
    obs_match = (predicted_observation == expected_obs)
    reward_match = math.isclose(predicted_reward, 0.0, abs_tol=1e-6)
    inv_empty = (len(inv_nonempty) == 0)

    # Full correct continuation -> reward
    if obs_match and reward_match and inv_empty:
        return 1.0

    # If observation or reward explicitly contradicts the expected wait behavior -> penalize
    if (not obs_match) or (not reward_match):
        return -1.0

    # Otherwise (parsing succeeded, only inventory diff present or other harmless variation) -> neutral
    return 0.0

# Rule 33
# Task group: measure the
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    try:
        # If action not provided or empty, try to extract from state
        act = (action or "").strip()
        if not act:
            for line in (state or "").splitlines():
                if line.strip().startswith("current_step_action:"):
                    act = line.split("current_step_action:", 1)[1].strip()
                    break

        # Apply only for the exact action
        if act != "focus on orange box":
            return 0.0

        # Split lines and locate required headers
        lines = [ln.rstrip("\n") for ln in (choice or "").splitlines()]
        obs_idx = rew_idx = inv_idx = None
        for i, ln in enumerate(lines):
            if ln.startswith("predicted_observation:") and obs_idx is None:
                obs_idx = i
            elif ln.startswith("predicted_reward:") and rew_idx is None:
                rew_idx = i
            elif ln.startswith("predicted_inventory_diff:") and inv_idx is None:
                inv_idx = i

        # All headers must be present and in order
        if obs_idx is None or rew_idx is None or inv_idx is None or not (obs_idx < rew_idx < inv_idx):
            return 0.0

        # Extract predicted_observation text
        obs_text = lines[obs_idx].split("predicted_observation:", 1)[1].lstrip()

        # Extract reward text and parse float
        reward_text = lines[rew_idx].split("predicted_reward:", 1)[1].strip()
        try:
            predicted_reward = float(reward_text)
        except Exception:
            return 0.0

        # Collect inventory diff lines after the predicted_inventory_diff: header
        inv_lines = []
        for ln in lines[inv_idx+1:]:
            if ln.strip() == "":
                continue
            inv_lines.append(ln)

        # Expected values
        expected_obs = "You focus on the orange box."
        tol = 1e-9
        expected_reward = 0.19

        # Check observation and inventory diff
        if obs_text != expected_obs:
            return -1.0
        if len(inv_lines) != 0:
            # Inventory diff present when none expected -> mismatch
            return -1.0

        # Observation and inventory match; check reward
        if abs(predicted_reward - expected_reward) < tol:
            return 1.0
        else:
            # observation OK but reward inconsistent -> moderate penalty
            return -0.5

    except Exception:
        return 0.0

# Rule 34
# Task group: melt
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    try:
        # Extract action from argument or from state if not provided
        act = action.strip() if action is not None and str(action).strip() != "" else None
        if not act:
            for line in (state or "").splitlines()[::-1]:
                if line.strip().startswith("current_step_action:"):
                    act = line.split("current_step_action:", 1)[1].strip()
                    break
        if act != "open door to outside":
            return 0.0

        # Split lines and find the three labeled fields
        lines = [ln.rstrip("\n") for ln in (choice or "").splitlines()]
        obs_line = None
        reward_line = None
        inv_index = None
        for i, ln in enumerate(lines):
            s = ln.lstrip()
            if obs_line is None and s.startswith("predicted_observation:"):
                obs_line = ln.split("predicted_observation:", 1)[1].strip()
            elif reward_line is None and s.startswith("predicted_reward:"):
                reward_line = ln.split("predicted_reward:", 1)[1].strip()
            elif inv_index is None and s.startswith("predicted_inventory_diff:"):
                inv_index = i

        # Require all three labels present to consider this rule
        if obs_line is None or reward_line is None or inv_index is None:
            return 0.0

        # Parse numeric reward
        try:
            pred_reward = float(reward_line)
        except Exception:
            return 0.0

        # Collect inventory diff lines: content on the predicted_inventory_diff: line (if any)
        # plus subsequent lines that start with + or - (stop at first non-diff line).
        inv_lines = []
        first_inv_content = lines[inv_index].split("predicted_inventory_diff:", 1)[1].strip()
        if first_inv_content:
            # If the first content contains multiple entries separated by ';' or commas,
            # keep it as a single line; we only check presence of expected strings below.
            inv_lines.append(first_inv_content)
        for ln in lines[inv_index + 1:]:
            stripped = ln.strip()
            if stripped == "":
                continue
            if stripped.startswith("+") or stripped.startswith("-"):
                inv_lines.append(stripped)
            else:
                break

        # Define expected pieces
        expected_obs_open = "The door is now open."
        expected_obs_already = "The door is already open."
        expected_add = "+ a metal pot (containing liquid ice cream)"
        expected_remove = "- a metal pot (containing ice cream)"

        # Check first acceptable continuation: door becomes open, reward ~0.28, inventory contains expected add/remove
        obs_ok_open = (obs_line == expected_obs_open)
        reward_ok_open = abs(pred_reward - 0.28) <= 0.01
        inv_ok_open = (any(il == expected_add for il in inv_lines) or any(expected_add in il for il in inv_lines)) and \
                      (any(il == expected_remove for il in inv_lines) or any(expected_remove in il for il in inv_lines))

        # Check second acceptable continuation: already open, reward 0.0, and no inventory diffs
        obs_ok_already = (obs_line == expected_obs_already)
        reward_ok_already = abs(pred_reward - 0.0) <= 1e-9
        inv_ok_already = (len(inv_lines) == 0)

        if obs_ok_open and reward_ok_open and inv_ok_open:
            return 1.0
        if obs_ok_already and reward_ok_already and inv_ok_already:
            return 1.0

        # Parsed successfully for this action but did not match either acceptable continuation -> penalize modestly
        return -0.8

    except Exception:
        # On unexpected errors, abstain from scoring to avoid false positives
        return 0.0

# Rule 35
# Task group: turn on
def rule_reward(state, action, choice):
    """
    Returns a float in [-1, 1] according to the rule described above.
    Triggers only for the exact action "connect black wire terminal 2 to anode in blue light bulb".
    Expects choice to contain lines:
      predicted_observation: <text>
      predicted_reward: <number>
    """
    import re

    try:
        # Determine current action: prefer provided action, else extract from state
        act = action.strip() if isinstance(action, str) and action.strip() != "" else None
        if not act and isinstance(state, str):
            m = re.search(r"current_step_action:\s*(.+)", state)
            if m:
                act = m.group(1).strip()

        target_action = "connect black wire terminal 2 to anode in blue light bulb"
        if act != target_action:
            return 0.0

        # Parse predicted_observation and predicted_reward from choice
        if not isinstance(choice, str):
            return 0.0

        pred_obs = None
        pred_reward = None
        for line in choice.splitlines():
            line = line.strip()
            if line.startswith("predicted_observation:"):
                pred_obs = line.split("predicted_observation:", 1)[1].strip()
            elif line.startswith("predicted_reward:"):
                numstr = line.split("predicted_reward:", 1)[1].strip()
                try:
                    pred_reward = float(numstr)
                except Exception:
                    return 0.0

        if pred_obs is None or pred_reward is None:
            return 0.0

        # Expected observation (must match exactly)
        expected_obs = "terminal 2 on black wire is now connected to anode on blue light bulb"
        if pred_obs != expected_obs:
            return 0.0

        # Possible correct reward targets inferred from per-question rules
        targets = [0.23, 0.47]

        # Scoring thresholds (conservative)
        eps_exact = 1e-6
        close_thresh = 0.05
        medium_thresh = 0.10

        # Check closeness to any target
        diffs = [abs(pred_reward - t) for t in targets]
        best_diff = min(diffs)

        if best_diff <= eps_exact:
            return 1.0
        if best_diff <= close_thresh:
            return 0.5
        if best_diff <= medium_thresh:
            return 0.2
        # Observation correct but reward far from any expected -> moderate penalty
        return -0.5

    except Exception:
        return 0.0

# Rule 36
# Task group: turn on
def rule_reward(state, action, choice):
    """
    Returns a float in [-1, 1] according to the merged, conservative rule described above.
    Triggers only for the two exact actions for connecting terminal 2 (black or orange) to the anode.
    """
    import re

    # Helper: extract current_step_action from state if action is empty or None
    if not action or action.strip() == "":
        marker = "current_step_action:"
        # find last occurrence
        idx = state.rfind(marker)
        if idx != -1:
            rest = state[idx + len(marker):].strip()
            action = rest.splitlines()[0].strip()
        else:
            action = ""

    action = action.strip()

    # Target actions
    target_black = "connect black wire terminal 2 to anode in electric motor"
    target_orange = "connect orange wire terminal 2 to anode in electric motor"
    if action not in (target_black, target_orange):
        return 0.0

    # Determine color and expected observation / allowed rewards
    if action == target_black:
        color = "black"
        allowed_rewards = [0.23, 0.47]  # accept both observed variants conservatively
    else:
        color = "orange"
        allowed_rewards = [0.47]

    expected_obs = f"terminal 2 on {color} wire is now connected to anode on electric motor"

    # Parse the choice into the three required fields.
    try:
        lines = choice.splitlines()
        # find lines that start with the exact field prefixes
        pred_obs = None
        pred_reward = None
        # predicted_inventory_diff may be a header line followed by zero or more lines;
        # we will capture all lines after the inventory header as inventory content.
        inv_idx = None
        for i, ln in enumerate(lines):
            if ln.startswith("predicted_observation:"):
                # take remainder of the same line after the prefix
                pred_obs = ln.split("predicted_observation:", 1)[1].strip()
            elif ln.startswith("predicted_reward:"):
                rew_text = ln.split("predicted_reward:", 1)[1].strip()
                # allow reward like "0.47" possibly followed/preceded by spaces
                pred_reward = float(rew_text)
            elif ln.startswith("predicted_inventory_diff:"):
                inv_idx = i
        # require that all three headers were present (inventory header may be last)
        if pred_obs is None or pred_reward is None or inv_idx is None:
            return 0.0
        # inventory content: all lines after inv_idx; if none -> empty
        inv_lines = []
        for j in range(inv_idx + 1, len(lines)):
            if lines[j].strip() != "":
                inv_lines.append(lines[j])
        predicted_inventory_diff = "\n".join(inv_lines).strip()
    except Exception:
        return 0.0

    # normalize observation for motor-on detection but keep exact-match requirement for correctness
    po_lower = pred_obs.lower()

    # Strong penalty if the model asserts the motor is on (explicit incorrect continuation)
    if "motor is now on" in po_lower or "the electric motor is now on" in po_lower:
        return -0.8

    # Exact observation required; be strict (strip surrounding whitespace when comparing)
    if pred_obs.strip() != expected_obs:
        return -1.0

    # Observation matches; inventory should be empty for full credit
    if predicted_inventory_diff != "":
        # unexpected inventory change; moderate penalty
        return -0.5

    # Check reward against allowed values
    eps = 1e-6
    for allowed in allowed_rewards:
        if abs(pred_reward - allowed) < eps:
            return 1.0

    # Observation correct but reward not one of the allowed values -> moderate penalty
    return -0.5

# Rule 37
# Task group: turn on
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    import re, math

    # Helper to extract current action from state if action not provided or empty
    act = action.strip() if action is not None else ""
    if act == "":
        m = re.search(r'current_step_action:\s*(.+)', state or "")
        if m:
            act = m.group(1).strip()

    # Only apply for the specific action
    if act != "wait1":
        return 0.0

    # Parse the choice into the three required fields.
    # Expect headers: predicted_observation:, predicted_reward:, predicted_inventory_diff:
    try:
        # Work with whole text to allow multi-line inventory diff
        full = choice or ""
        # Find observation line (take the first occurrence)
        obs_m = re.search(r'predicted_observation:\s*(.*)', full)
        # Find reward number (first numeric token after predicted_reward:)
        reward_m = re.search(r'predicted_reward:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', full)
        # Find inventory diff header and capture the remainder (may be empty or multiline)
        inv_m = re.search(r'predicted_inventory_diff:\s*(.*)', full, flags=re.DOTALL)

        if not obs_m or not reward_m or inv_m is None:
            return 0.0

        predicted_observation = obs_m.group(1).strip()
        try:
            predicted_reward = float(reward_m.group(1))
        except Exception:
            return 0.0

        inventory_text = inv_m.group(1)
        # Inventory considered empty if there are no non-blank lines after the header
        inv_lines = [ln for ln in (inventory_text.splitlines() if inventory_text is not None else []) if ln.strip() != ""]
        inventory_empty = (len(inv_lines) == 0)
    except Exception:
        return 0.0

    # Expected observation and plausible expected rewards (conservative merge of per-question variants)
    expected_obs = "You decide to wait for 1 iterations."
    plausible_rewards = [0.17, 0.42]

    # Tolerances
    tiny_tol = 1e-6
    small_tol = 0.05  # small tolerance for "close" reward

    # If observation exactly matches, evaluate reward and inventory
    if predicted_observation == expected_obs:
        # Check closeness to any plausible expected reward
        close_exact = any(math.isclose(predicted_reward, r, rel_tol=0.0, abs_tol=tiny_tol) for r in plausible_rewards)
        close_small = any(abs(predicted_reward - r) <= small_tol for r in plausible_rewards)

        # Best case: exact (within tiny tol) match to a plausible reward and empty inventory
        if close_exact and inventory_empty:
            return 1.0
        # Very good: within small tolerance and empty inventory
        if close_small and inventory_empty:
            return 0.8
        # Observation correct and inventory empty but reward not near expected -> moderate positive
        if inventory_empty:
            return 0.6
        # Observation correct but inventory non-empty (unexpected inventory change) -> lower positive
        return 0.3

    # Observation does not match expected continuation -> penalize (but be conservative)
    # If observation incorrect but inventory empty and reward happens to be very close to plausible value, give small negative.
    close_small_any = any(abs(predicted_reward - r) <= small_tol for r in plausible_rewards)
    if inventory_empty and close_small_any:
        return -0.2
    # Otherwise, stronger penalty
    return -0.8

# Rule 38
# Task group: use chemistry
def rule_reward(state, action, choice):
    """
    Returns a float in [-1.0, 1.0].
    Triggers only when current action == "focus on paint in bowl".
    Parses 'predicted_observation:' and 'predicted_reward:' from choice.
    - If parsing fails (missing fields or reward not parseable) -> 0.0 (conservative).
    - If both parsed and match expected observation and reward -> 1.0.
    - If both parsed but either mismatches -> -1.0.
    """
    try:
        # Determine current action: prefer explicit action argument if non-empty,
        # otherwise extract from state.
        cur_action = action.strip() if action is not None else ""
        if not cur_action:
            for line in state.splitlines():
                if line.strip().startswith("current_step_action:"):
                    cur_action = line.split("current_step_action:", 1)[1].strip()
                    break

        if cur_action != "focus on paint in bowl":
            return 0.0

        # Parse choice for predicted_observation and predicted_reward (first occurrences).
        obs = None
        reward_val = None
        import re
        for ln in choice.splitlines():
            s = ln.strip()
            if s.startswith("predicted_observation:") and obs is None:
                obs = s.split("predicted_observation:", 1)[1].strip()
            elif s.startswith("predicted_reward:") and reward_val is None:
                raw = s.split("predicted_reward:", 1)[1].strip()
                # Extract a leading numeric token if present
                m = re.match(r'([+-]?\d+(\.\d*)?([eE][+-]?\d+)?)', raw)
                if m:
                    try:
                        reward_val = float(m.group(1))
                    except Exception:
                        reward_val = None
                else:
                    # fallback: try full-string float parse
                    try:
                        reward_val = float(raw)
                    except Exception:
                        reward_val = None

        # Conservative behavior: if required fields aren't parseable, do not apply rule.
        if obs is None or reward_val is None:
            return 0.0

        expected_obs = "You focus on the violet-red paint."
        expected_reward = 0.17
        if obs == expected_obs and abs(reward_val - expected_reward) <= 1e-6:
            return 1.0
        else:
            # Action matched and parsing succeeded but values differ -> strong penalty
            return -1.0

    except Exception:
        # On unexpected errors, do not apply the rule (conservative)
        return 0.0

# Rule 39
# Task group: use chemistry
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    import re

    TARGET_ACTION = "focus on orange paint"
    TARGET_OBS = "You focus on the orange paint."
    TARGET_REWARD = 0.50
    REWARD_TOL = 1e-6  # tiny tolerance for floating comparison
    PENALTY = -0.9

    # Determine current action: prefer provided action, otherwise extract from state
    act = (action or "").strip()
    if not act:
        m = re.search(r'current_step_action:\s*(.*)', state or "")
        if m:
            act = m.group(1).strip()
    if act != TARGET_ACTION:
        return 0.0

    # Normalize and split choice into lines
    if choice is None:
        return 0.0
    lines = [ln.rstrip("\n") for ln in choice.splitlines()]

    # Locate header lines (allow leading whitespace before header)
    idx_obs = idx_rew = idx_inv = None
    for i, ln in enumerate(lines):
        s = ln.lstrip()
        if s.startswith("predicted_observation:") and idx_obs is None:
            idx_obs = i
        elif s.startswith("predicted_reward:") and idx_rew is None:
            idx_rew = i
        elif s.startswith("predicted_inventory_diff:") and idx_inv is None:
            idx_inv = i

    # If any header missing -> treat as parsing failure (do not penalize)
    if idx_obs is None or idx_rew is None or idx_inv is None:
        return 0.0

    # Extract observation text (text after the first ':' on the header line)
    try:
        obs_line = lines[idx_obs]
        obs = obs_line.split("predicted_observation:", 1)[1].strip()
    except Exception:
        return 0.0

    # Extract reward as float from its header line
    try:
        rew_line = lines[idx_rew]
        rew_str = rew_line.split("predicted_reward:", 1)[1].strip()
        predicted_reward = float(rew_str)
    except Exception:
        return 0.0

    # Inventory diff: any non-empty lines after the inventory header count as changes
    inv_lines = []
    try:
        for ln in lines[idx_inv + 1:]:
            if ln.strip() != "":
                inv_lines.append(ln.strip())
    except Exception:
        return 0.0

    # Check expected values
    obs_match = (obs == TARGET_OBS)
    reward_match = (abs(predicted_reward - TARGET_REWARD) <= REWARD_TOL)
    inv_empty = (len(inv_lines) == 0)

    if obs_match and reward_match and inv_empty:
        return 1.0

    # Parsed successfully but one or more fields disagree -> conservative penalty
    return float(PENALTY)

# Rule 40
def rule_reward(state, action, choice):
    import re
    # If action not provided, extract from state
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''
    # Parse predicted_observation, predicted_reward, predicted_inventory_diff from choice
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)\n(?=predicted_reward:)', choice)
    rew_m = re.search(r'(?m)predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff:\s*(.*)$', choice)
    if not (obs_m and rew_m and diff_m is not None):
        return -0.5
    obs = obs_m.group(1).strip().lower()
    inv_block = diff_m.group(1)
    # Determine if this rule applies: action is 'look' or 'look around'
    if not re.match(r'(?i)^\s*look(?:\s+around)?\s*$', action):
        return 0.0
    score = 0.0
    # Check observation contains full-room description cue
    if 'this room is called' in obs:
        score += 0.5
    # Check inventory diff contains no +/- lines with content
    inv_lines = [ln.strip() for ln in inv_block.splitlines() if ln.strip()]
    has_plus_or_minus = any((ln.startswith('+') or ln.startswith('-')) for ln in inv_lines)
    if not has_plus_or_minus:
        score += 0.5
    # Clamp to [-1,1]
    if score == 0.0:
        # failed both checks -> give a negative feedback but bounded
        return -0.6
    return max(-1.0, min(1.0, score))

# Rule 41
def rule_reward(state, action, choice):
    """
    Rule for 'examine <object>' actions:
    - If action starts with 'examine <obj>' (or current_step_action is used), then:
      * predicted_observation should mention <obj> (case-insensitive substring match)
      * predicted_inventory_diff should be empty (no non-empty +/- lines)
      * predicted_reward should be > 0 (informative examine should have positive reward)
    Returns a score in [-1, 1].
    """
    import re

    # Extract action from state if not provided
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''

    # Only apply this rule to 'examine ...' actions
    m_act = re.match(r'(?i)\s*examine\s+(.+)$', action.strip())
    if not m_act:
        return 0.0

    obj = m_act.group(1).strip().lower()
    # normalize object (remove surrounding punctuation)
    obj_norm = re.sub(r'[^a-z0-9 ]+', '', obj)

    # Parse the choice fields
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)(?:\npredicted_reward:|\npredicted_inventory_diff:|$)', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)

    if obs_m is None or rew_m is None or diff_m is None:
        return -0.5

    obs = obs_m.group(1).strip().lower()
    try:
        rew = float(rew_m.group(1))
    except Exception:
        rew = 0.0
    inv = diff_m.group(1)

    # Evaluate whether observation mentions the object
    obs_ok = False
    if obj_norm and obj_norm in obs:
        obs_ok = True
    else:
        # fallback: check that at least one word from the object appears
        words = [w for w in obj_norm.split() if w]
        if any(w in obs for w in words):
            obs_ok = True

    # Evaluate inventory diff empty (no non-empty lines)
    inv_lines = [ln.strip() for ln in inv.splitlines() if ln.strip()]
    inv_ok = (len(inv_lines) == 0)

    # Reward preference: positive reward expected
    rew_ok = (rew > 0.0)

    # Scoring: start neutral, add/subtract components
    score = 0.0
    if obs_ok:
        score += 0.6
    else:
        score -= 0.6
    if inv_ok:
        score += 0.3
    else:
        score -= 0.3
    if rew_ok:
        score += 0.1
    else:
        score -= 0.1

    # Clamp to [-1, 1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return score

# Rule 42
def rule_reward(state, action, choice):
    import re
    # If action not provided, try to extract from state
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''
    # Parse predicted_observation, predicted_reward, predicted_inventory_diff from choice
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)(?:\r?\n)predicted_reward:', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)
    if not (obs_m and rew_m and diff_m is not None):
        return -0.5  # malformed choice
    obs = obs_m.group(1).strip().lower()
    try:
        rew = float(rew_m.group(1))
    except:
        return -0.5
    inv = diff_m.group(1)
    # Only apply rule when action matches 'focus on <obj>'
    m_act = re.match(r'(?i)\s*focus on\s+(.+)$', action.strip())
    if not m_act:
        return 0.0  # rule not applicable
    obj = m_act.group(1).strip().lower()
    # Normalize object string: collapse repeated words and split into tokens
    obj = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', obj)  # "baby baby dragonfly" -> "baby dragonfly"
    obj_tokens = [t for t in re.split(r'[^a-z0-9]+', obj) if len(t) > 2]
    # Check observation: must contain "you focus on" and mention at least one meaningful token from object
    ok_obs = False
    if 'you focus on' in obs:
        if not obj_tokens:
            ok_obs = True
        else:
            for tk in obj_tokens:
                if tk in obs:
                    ok_obs = True
                    break
    # Check inventory diff: should contain no non-empty lines (no +/- changes)
    inv_lines = [ln.strip() for ln in inv.splitlines() if ln.strip()]
    ok_inv = (len(inv_lines) == 0)
    # Check reward: should be positive (> 0)
    ok_rew = (rew > 0.0)
    # Score aggregation
    score = 0.0
    if ok_obs:
        score += 0.5
    if ok_rew:
        score += 0.3
    if ok_inv:
        score += 0.2
    if score == 1.0:
        return 1.0
    # If any required condition fails, penalize
    return -0.3

# Rule 43
def rule_reward(state, action, choice):
    import re
    # Extract action from state if not provided
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''
    # Parse choice fields robustly
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)(?=\r?\npredicted_reward:)', choice)
    rew_m = re.search(r'(?m)^\s*predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)
    if not (obs_m and rew_m and diff_m is not None):
        return -0.5
    obs = obs_m.group(1).strip().lower()
    try:
        rew = float(rew_m.group(1))
    except:
        return -0.5
    inv = diff_m.group(1)
    inv_lines = [ln.strip() for ln in inv.splitlines() if ln.strip()]
    no_inv_change = len(inv_lines) == 0

    a = action.strip().lower()

    # Helper checks for observation matching
    def obs_has_movement(o):
        return ('you move to' in o) or ('you go to' in o) or ('you move' in o)

    def obs_has_focus(o):
        return ('you focus on' in o) or (o.startswith('you focus')) or ('you focus' in o)

    def obs_has_open(o, obj=None):
        if 'is now open' in o or 'already open' in o:
            return True
        # sometimes phrasing: "the door is now open." check generic
        return False

    # Match action patterns
    m_go = re.match(r'^\s*go to\s+(.+)$', a)
    m_focus = re.match(r'^\s*focus(?:\s+on)?\s+(.+)$', a)
    m_open = re.match(r'^\s*open\s+(.+)$', a)

    # If none match, rule does not apply
    if not (m_go or m_focus or m_open):
        return 0.0

    # For go to actions
    if m_go:
        # Expect movement phrase, empty inventory diff, and reward >= 0.4
        if not obs_has_movement(obs):
            return -0.8
        if not no_inv_change:
            return -0.5
        return 1.0 if rew >= 0.4 else -0.5

    # For focus actions
    if m_focus:
        if not obs_has_focus(obs):
            return -0.8
        if not no_inv_change:
            return -0.5
        return 1.0 if rew >= 0.4 else -0.5

    # For open actions
    if m_open:
        if not obs_has_open(obs, m_open.group(1)):
            return -0.8
        if not no_inv_change:
            return -0.5
        return 1.0 if rew >= 0.05 else -0.5

    return 0.0

# Rule 44
def rule_reward(state, action, choice):
    import re
    # Extract action if not provided
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''
    # Parse predicted_observation, predicted_reward, predicted_inventory_diff
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)\n\s*predicted_reward', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)
    if not (obs_m and rew_m and diff_m is not None):
        return -0.5
    obs = obs_m.group(1).strip().lower()
    try:
        rew = float(rew_m.group(1))
    except:
        rew = 0.0
    inv = diff_m.group(1).strip()
    # Apply only to connect actions
    if not re.match(r'(?i)\s*connect\s+', action):
        return 0.0
    # Try to split left and right endpoints from action: "connect X to Y"
    m = re.match(r'(?i)\s*connect\s+(.+?)\s+to\s+(.+)$', action.strip())
    if not m:
        # Could not parse endpoints: check minimal expectation (observation indicates connection)
        if 'now connected' in obs or 'is now connected' in obs:
            # still expect no inventory change
            inv_lines = [ln for ln in inv.splitlines() if ln.strip()]
            return 1.0 if not inv_lines and rew >= 0.2 else (-0.7 if not inv_lines else -0.3)
        return -0.6
    left = m.group(1).lower()
    right = m.group(2).lower()
    # Basic observation requirements: mentions connection phrase
    if not ('now connected' in obs or 'is now connected' in obs):
        return -0.6
    # If action referenced 'terminal', require 'terminal' in observation; similarly for anode/cathode
    action_lc = action.lower()
    if 'terminal' in action_lc and 'terminal' not in obs:
        return -0.6
    if ('anode' in action_lc or 'cathode' in action_lc) and not ('anode' in obs or 'cathode' in obs):
        return -0.6
    # Expect at least some mention of both endpoints (wire/device names or terminal tokens)
    # We'll check whether significant tokens from left and right appear in the observation
    def significant_tokens(s):
        # break into tokens but keep multiword items like "red wire" if present
        toks = []
        # prefer phrases like "<color> wire", "<device> anode", "terminal N"
        # simple heuristics:
        s = s.replace('  ', ' ')
        if 'terminal' in s:
            mterm = re.search(r'terminal\s*\d+', s)
            if mterm:
                toks.append(mterm.group(0))
        for keyword in ['wire', 'battery', 'solar panel', 'generator', 'motor', 'light bulb', 'bulb', 'anode', 'cathode', 'switch']:
            if keyword in s:
                toks.append(keyword)
        # also add the first word (often color) to help matching (e.g., 'red')
        first = s.split()[0]
        toks.append(first)
        return list(dict.fromkeys(toks))
    left_toks = significant_tokens(left)
    right_toks = significant_tokens(right)
    def tokens_present(toks):
        return any(tok in obs for tok in toks if tok)
    if not (tokens_present(left_toks) and tokens_present(right_toks)):
        # allow a weaker pass if at least one endpoint clearly present and terminal/anode/cathode matched earlier
        if not (tokens_present(left_toks) or tokens_present(right_toks)):
            return -0.6
    # Inventory diff should be empty (no + or - lines)
    inv_lines = [ln for ln in inv.splitlines() if ln.strip()]
    if inv_lines:
        return -0.3
    # Reward should be reasonably positive for a successful connection
    if rew >= 0.2:
        return 1.0
    else:
        # observation is good but reward is suspiciously low
        return -0.7

# Rule 45
def rule_reward(state, action, choice):
    import re
    # Extract action if not provided
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''
    # Parse predicted fields from the choice
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)\npredicted_reward:', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff:\s*(.*)$', choice)
    if not (obs_m and rew_m and diff_m is not None):
        # malformed choice
        return -0.5
    obs = obs_m.group(1).strip().lower()
    inv_text = diff_m.group(1) or ""
    inv_lines = [ln.strip() for ln in inv_text.splitlines() if ln.strip()]
    # Only apply rule to "move ... to ..." actions
    m = re.match(r'(?i)\s*move\s+(.+?)\s+(?:in\s+inventory\s+)?to\s+(.+)$', action.strip())
    if not m:
        return 0.0
    # Detect whether action explicitly included 'in inventory'
    in_inv_flag = bool(re.search(r'(?i)\bin\s+inventory\b', action))
    obj_raw = m.group(1).strip().lower()
    container_raw = m.group(2).strip().lower()
    # Tokenize object and container for fuzzy matching
    obj_tokens = [t for t in re.findall(r'\w+', obj_raw) if t not in ('a','an','the','in','inventory')]
    cont_tokens = [t for t in re.findall(r'\w+', container_raw) if t not in ('a','an','the')]
    # Observation check: must mention a move and the container and at least one object token
    obs_ok = False
    if 'you move' in obs:
        cont_present = any(tok in obs for tok in cont_tokens) if cont_tokens else True
        obj_present = any(tok in obs for tok in obj_tokens) if obj_tokens else True
        if cont_present and obj_present:
            obs_ok = True
    # Inventory diff check:
    # If action says 'in inventory' we require a '- ' line that references the object.
    # If action does NOT say 'in inventory' we require that no '- ' line references the object.
    obj_removed = any(ln.startswith('-') and any(tok in ln.lower() for tok in obj_tokens) for ln in inv_lines)
    if in_inv_flag:
        inv_ok = obj_removed
    else:
        # ok if no removal of the object occurs
        inv_ok = not obj_removed
    # Score composition
    score = 0.0
    if obs_ok:
        score += 0.5
    if inv_ok:
        score += 0.5
    # If both checks fail, penalize strongly
    if not obs_ok and not inv_ok:
        score = -0.8
    # Clamp to [-1,1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return float(score)

# Rule 46
def rule_reward(state, action, choice):
    import re
    def clamp(x, lo=-1.0, hi=1.0):
        return max(lo, min(hi, x))

    # If action not provided, extract from state
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''

    # Only apply to thermometer uses
    if not re.search(r'(?i)use\s+.*thermometer', action):
        return 0.0

    # Parse predicted fields from choice
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)(?:\npredicted_reward:)', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)

    if not (obs_m and rew_m and diff_m is not None):
        return -0.5

    obs = obs_m.group(1).strip().lower()
    inv = diff_m.group(1)
    # Look for temperature number in predicted observation
    temp_m = re.search(r'the thermometer measures a temperature of\s*([-+]?\d*\.?\d+)\s*degrees', obs)
    if not temp_m:
        # Missing the expected thermometer measurement phrase
        return -0.6

    try:
        temp = float(temp_m.group(1))
    except:
        return -0.6

    # Inventory diff should be empty (no +/- lines)
    inv_lines = [ln.strip() for ln in inv.splitlines() if ln.strip()]
    inv_ok = len(inv_lines) == 0

    score = 0.0
    if inv_ok:
        score += 0.5
    else:
        score -= 0.5

    # Gather previous thermometer measurements from the state history (if any)
    prev_temps = [float(n) for n in re.findall(
        r'the thermometer measures a temperature of\s*([-+]?\d*\.?\d+)\s*degrees', state, flags=re.I)]
    last_temp = prev_temps[-1] if prev_temps else None

    # Detect if a heater is currently on in the state
    heater_on = bool(re.search(r'(?i)\b(stove|blast furnace)\b[^.!\n]*turned on', state))

    # Apply monotonicity / plausibility checks
    if last_temp is not None:
        if heater_on:
            # If heating, temperature should not decrease
            if temp + 1e-9 >= last_temp:
                score += 0.4
            else:
                score -= 0.4
        else:
            # If not heating, reading should be approximately stable (within ~3 degrees)
            if abs(temp - last_temp) <= 3.0:
                score += 0.4
            else:
                score -= 0.25
    else:
        # No prior measurement: accept plausible physical range
        if -100.0 <= temp <= 2000.0:
            score += 0.2
        else:
            score -= 0.4

    return clamp(score)

# Rule 47
def rule_reward(state, action, choice):
    import re
    # If action not provided, extract current_step_action from state
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''

    # Parse choice fields
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)\n(?=predicted_reward:)', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)

    # If required parts are missing, return a moderate negative signal
    if not (obs_m and rew_m and diff_m is not None):
        return -0.5

    obs = obs_m.group(1).strip().lower()
    inv = diff_m.group(1)
    inv_lines = [ln.strip() for ln in inv.splitlines() if ln.strip()]

    # Match action patterns: 'move ... to ...' or 'drop ...'
    m_move = re.match(r'(?i)^\s*move\s+(?:the\s+)?(.+?)\s+to\s+(.+)$', action.strip())
    m_drop = re.match(r'(?i)^\s*drop\s+(?:the\s+)?(.+)$', action.strip())

    if not (m_move or m_drop):
        # Rule not applicable
        return 0.0

    # Extract object name (normalize)
    if m_move:
        obj = m_move.group(1).strip().lower()
        container = m_move.group(2).strip().lower()
    else:
        obj = m_drop.group(1).strip().lower()
        container = None

    # Simplify object match token: choose a short representative token (first noun-like token)
    # e.g., "unknown substance b" -> "unknown" or "substance b"; prefer last token if it is a single letter/word
    obj_tokens = re.findall(r'\w+', obj)
    obj_token = obj_tokens[-1] if obj_tokens else obj

    score = 0.0

    # Check predicted_observation mentions the move/drop to a location
    ok_obs = False
    # Accept several reasonable phrasings
    if re.search(r'\bmove the\b.*\b' + re.escape(obj_token) + r'\b', obs) and 'to' in obs:
        ok_obs = True
    if re.search(r'\byou move\b.*\b' + re.escape(obj_token) + r'\b', obs):
        ok_obs = True
    if re.search(r'\byou drop\b.*\b' + re.escape(obj_token) + r'\b', obs):
        ok_obs = True
    # Also accept "You move the <obj> to the <container>"
    if container:
        if re.search(r'\byou move\b.*' + re.escape(obj_token) + r'.*to.*' + re.escape(container), obs):
            ok_obs = True
        if re.search(r'\bmove the\b.*' + re.escape(obj_token) + r'.*to.*' + re.escape(container), obs):
            ok_obs = True

    if ok_obs:
        score += 0.6

    # Check predicted_inventory_diff contains a '- ' line mentioning the object
    has_minus = False
    for ln in inv_lines:
        if ln.startswith('- '):
            if obj_token in ln.lower() or any(tok in ln.lower() for tok in obj_tokens):
                has_minus = True
                break
    if has_minus:
        score += 0.4

    # Final score in [-1,1]
    if score == 0.0:
        # If action matched but neither condition met, penalize moderately
        return -0.5
    return max(-1.0, min(1.0, score))

# Rule 48
def rule_reward(state, action, choice):
    import re
    # Extract action from state if not provided
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''

    # Parse choice into fields
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)(?:\npredicted_reward:)', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)

    # If required fields missing, low confidence negative
    if not (obs_m and rew_m and diff_m is not None):
        return -0.6

    obs = obs_m.group(1).strip().lower()
    inv_diff = diff_m.group(1)

    # Match move action pattern: move <obj> to <container>
    m = re.match(r'(?i)\s*move\s+(.+?)\s+to\s+(.+)$', action.strip())
    if not m:
        # Rule does not apply
        return 0.0

    obj = m.group(1).strip().lower()
    container = m.group(2).strip().lower()

    # Determine if the object appears connected in the state/history
    # Look for phrases indicating connections involving the object
    conn_patterns = [
        rf'{re.escape(obj)}.*connected',
        rf'connected.*{re.escape(obj)}',
        rf'{re.escape(obj)}.*terminal',
        rf'{re.escape(obj)}.*anode',
        rf'{re.escape(obj)}.*cathode',
    ]
    connection_found = False
    for p in conn_patterns:
        if re.search(p, state, re.I | re.S):
            connection_found = True
            break

    score = 0.0

    # Check that observation describes the move
    move_ok = False
    # Accept variants like "you move the <obj> to the <container>" or "you move <obj> to <container>"
    if (f'move the {obj} to' in obs) or (f'move {obj} to' in obs) or (f'you move the {obj} to' in obs) or (f'you move {obj} to' in obs):
        move_ok = True
        score += 0.5

    # If object was connected, require a disconnection mention
    if connection_found:
        if any(w in obs for w in ('disconnect', 'disconnecting', 'disconnected', 'disconnects')):
            score += 0.5
        else:
            # Strong penalty if we moved a connected object without mentioning disconnection
            return -1.0

    # Ensure no inventory +/- lines for an intra-environment move
    inv_lines = [ln for ln in inv_diff.splitlines() if ln.strip()]
    has_inventory_changes = any(ln.strip().startswith(('+', '-')) for ln in inv_lines)
    if has_inventory_changes:
        # Penalize small amount: inventory should not change when moving within room/container
        score -= 0.5

    # Normalize final score to [-1,1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return float(score)

