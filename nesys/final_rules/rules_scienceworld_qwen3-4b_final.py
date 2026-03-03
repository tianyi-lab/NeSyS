# WMQA Improved Rules
# Improved from (2 files):
#   - transition_mcq/rules_scienceworld_qwen3-4b.py
#   - transition_mcq/scienceworld_task_combined_rules_qwen3-4b.py
# Dev unit-weight improvement vs original: +4.23%
# Dev unit-weight accuracy (improved rules): 82.90%
# Dev weighted accuracy (learned on dev): 84.10%
# Test baseline accuracy: 68.48%
# Test weighted accuracy: 70.62%
# Test weighted improvement: +2.14%

# Rule 1
def rule_reward(state, action, choice):
    import re
    # Extract current_step_action from state if action not provided
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''

    # Parse predicted_observation, predicted_reward, predicted_inventory_diff from choice
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)(?:\npredicted_reward:|\n$)', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff:\s*(.*)$', choice)

    if not obs_m or diff_m is None:
        # malformed choice: penalize moderately
        return -0.6

    obs = obs_m.group(1).strip().lower()
    inv_diff = diff_m.group(1)

    # Only apply this rule for thermometer-use actions with an 'on' target
    if not re.search(r'(?i)\buse\b.*\bthermometer\b.*\bon\b', action):
        return 0.0

    # Condition 1: observation contains the thermometer measurement phrase
    has_measurement = 'the thermometer measures a temperature of' in obs

    # Condition 2: inventory diff should be empty of +/- lines
    inv_lines = [ln.strip() for ln in inv_diff.splitlines() if ln.strip()]
    has_inventory_change = any(ln.startswith('+') or ln.startswith('-') for ln in inv_lines)

    if has_measurement and not has_inventory_change:
        return 1.0
    if not has_measurement:
        return -0.6
    # measurement present but inventory changed unexpectedly
    return -0.4

# Rule 2
def rule_reward(state, action, choice):
    import re
    # Extract action from state if not provided
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''
    # Parse predicted_observation, predicted_reward, predicted_inventory_diff
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)\s*[\r\n]+predicted_reward:', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)
    # If required fields missing, penalize
    if not (obs_m and rew_m and diff_m is not None):
        return -0.5
    obs = obs_m.group(1).strip().lower()
    inv = diff_m.group(1)
    # Only apply rule for "go to <location>"
    act_m = re.match(r'(?i)^\s*go to\s+(.+)$', action.strip())
    if not act_m:
        return 0.0
    location = act_m.group(1).strip().lower().rstrip('.')
    # Check observation contains the expected move message
    expected_move = f'you move to the {location}'
    obs_ok = expected_move in obs
    # Check predicted_inventory_diff: empty or only thermometer +/- lines
    inv_lines = [ln.strip() for ln in inv.splitlines() if ln.strip()]
    inv_ok = True
    for ln in inv_lines:
        low = ln.lower()
        if not (low.startswith('+ a thermometer') or low.startswith('- a thermometer')):
            inv_ok = False
            break
    # Final scoring
    if obs_ok and inv_ok:
        return 1.0
    else:
        return -0.5

# Rule 3
def rule_reward(state, action, choice):
    import re

    # Safe strings
    state_text = (state or "")
    action_text = (action or "")
    choice_text = (choice or "")

    # If action not provided, try to extract from state
    if not action_text:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state_text)
        action_text = m.group(1).strip() if m else ""

    # Extract predicted_observation and predicted_inventory_diff from choice (robust)
    obs_m = re.search(r'(?si)predicted_observation:\s*(.*?)\s*predicted_reward:', choice_text)
    diff_m = re.search(r'(?si)predicted_inventory_diff:\s*(.*)$', choice_text)

    # If we can't find the observation or the inventory-diff field, be conservative: no judgement.
    if not obs_m or diff_m is None:
        return 0.0

    obs = obs_m.group(1).strip().lower()
    inv = diff_m.group(1).strip().lower()
    state_l = state_text.lower()
    action_l = action_text.strip().lower()

    # Parse action: expect "move <object> [in inventory] to <dest>" (be forgiving)
    act_m = re.match(r'(?i)\s*move\s+(.+?)\s+(?:in inventory\s+)?to\s+(.+)$', action_l)
    if not act_m:
        # Not a move-to-container action we should judge
        return 0.0

    obj_raw = act_m.group(1).strip()
    dest_raw = act_m.group(2).strip()

    # Tokenize significant words from object (prefer tokens length >2)
    obj_tokens = [t for t in re.findall(r"[a-z0-9]+", obj_raw) if len(t) > 2]
    if not obj_tokens:
        obj_tokens = [t for t in re.findall(r"[a-z0-9]+", obj_raw)]

    # Prepare regexes
    dest_re = re.compile(r'\b' + re.escape(dest_raw) + r'\b')
    obj_token_res = [re.compile(r'\b' + re.escape(t) + r'\b') for t in obj_tokens] if obj_tokens else []

    # Observation must explicitly indicate a direct action (conservative)
    obs_ok = bool(re.search(r'\byou\s+(?:move|put|place|drop)\b', obs))

    # Destination must be mentioned in observation (word boundaries)
    dest_ok = bool(dest_re.search(obs))

    # Object mention in observation: require at least one significant token to match
    obj_in_obs = any(r.search(obs) for r in obj_token_res) if obj_token_res else False

    # Determine inventory removal mentioned in the choice's predicted_inventory_diff
    inv_lines = [ln.strip() for ln in inv.splitlines() if ln.strip()]
    inv_has_removal = False
    for ln in inv_lines:
        # look for lines starting with '-' (allow '- ' or just '-')
        if ln.startswith('-') or ln.startswith('- '):
            # ensure the line mentions one of the object tokens
            if any(r.search(ln) for r in obj_token_res):
                inv_has_removal = True
                break

    # Compute whether the object was in inventory at the time of the current action.
    # Conservative approach:
    # 1) Use the last explicit "In your inventory, you see:" block (if any) as the initial snapshot.
    # 2) Then apply all inventory_diff blocks found in the state in chronological order to update presence.
    def compute_obj_currently_in_inventory(state_text_local, tokens_res):
        present = None  # None means unknown; True/False known
        try:
            # 1) last explicit inventory snapshot
            inv_blocks = re.findall(r'(?si)in your inventory, you see:(.*?)(?:\n\s*\n|$)', state_text_local)
            if inv_blocks:
                last_blk = inv_blocks[-1].lower()
                # If any token appears in that last block, set present True, else False (we have a snapshot)
                if any(r.search(last_blk) for r in tokens_res):
                    present = True
                else:
                    present = False
            # 2) apply inventory_diff blocks in chronological order
            # Find inventory_diff: ... blocks - iterate in text order
            diffs = []
            for m in re.finditer(r'(?si)inventory_diff:\s*(.*?)(?=(?:\n\s*action:|\n\s*$)|\Z)', state_text_local):
                diffs.append(m.group(1))
            # Apply diffs sequentially
            for d in diffs:
                lines = [ln.strip() for ln in d.splitlines() if ln.strip()]
                for ln in lines:
                    # a line like '+ unknown substance o' or '- unknown substance o'
                    if ln.startswith('+') or ln.startswith('+ '):
                        if any(r.search(ln.lower()) for r in tokens_res):
                            present = True
                    elif ln.startswith('-') or ln.startswith('- '):
                        if any(r.search(ln.lower()) for r in tokens_res):
                            present = False
            return present  # may be True/False/None
        except Exception:
            return None

    obj_was_in_inventory = compute_obj_currently_in_inventory(state_l, obj_token_res)

    # Scoring logic (conservative and more precise):
    # Only apply confident judgments when observation explicitly says the move and destination appears.
    if obs_ok and dest_ok and obj_in_obs:
        # If we know the object was in inventory, the predicted inventory diff should include a removal.
        if obj_was_in_inventory is True:
            if inv_has_removal:
                # observation, dest, object mention, and removal present -> strong positive
                return 1.0
            else:
                # observation and object mention indicate a drop/move from inventory but predicted diff lacks removal:
                # Penalize moderately, but only when presence was confidently determined.
                return -0.4
        elif obj_was_in_inventory is False:
            # Object was not in inventory (we have a snapshot/updates indicating that).
            # If predicted diff shows removal of object from inventory despite that, that's inconsistent -> small penalty.
            if inv_has_removal:
                return -0.2
            else:
                # plausible: moving object in environment to container, inventory diff omission is fine -> modest credit
                return 0.3
        else:
            # obj_was_in_inventory is None (uncertain). Be conservative:
            # If predicted_inventory_diff explicitly shows removal, give small positive; else no strong judgement.
            if inv_has_removal:
                return 0.2
            else:
                return 0.0

    # If observation indicates move & destination but object isn't mentioned in observation:
    if obs_ok and dest_ok and not obj_in_obs:
        # If inventory shows removal, give partial credit; else be neutral (avoid penalizing omission)
        if inv_has_removal:
            return 0.2
        else:
            return 0.0

    # Otherwise be conservative and abstain
    return 0.0

# Rule 4
def rule_reward(state, action, choice):
    import re
    # If action not provided, extract from state
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''
    # Only apply to 'look around' action
    if not re.match(r'(?i)^\s*look around\s*$', action.strip()):
        return 0.0

    # Parse predicted_observation, predicted_reward, predicted_inventory_diff from choice
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)(?:\n(?=predicted_reward:)|\n$)', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff:\s*(.*)$', choice)

    if not (obs_m and rew_m and diff_m is not None):
        return -0.5

    obs = obs_m.group(1).strip().lower()
    inv = diff_m.group(1) or ''
    inv_lines = [ln.strip() for ln in inv.splitlines() if ln.strip()]

    # Check for good room-description phrases
    has_room_phrase = ('this room is called' in obs) or ('in it, you see' in obs) or ('you see:' in obs)
    # Check for movement/result phrases which shouldn't appear for look around
    movement_phrases = ['you enter', 'you move', 'you go', 'you walk', 'you pick up', 'you move to', 'you put', 'you pick']
    has_movement = any(p in obs for p in movement_phrases)

    # Check inventory diff lines indicating changes
    has_inv_change = any(ln.startswith('+') or ln.startswith('-') for ln in inv_lines)

    # Scoring logic
    if has_room_phrase and not has_movement and not has_inv_change:
        return 1.0
    if has_movement or has_inv_change:
        return -0.8
    # Observation present but not clearly a room description (no room phrase) and no inventory changes
    return -0.4

# Rule 5
def rule_reward(state, action, choice):
    import re
    # If action not provided, extract from state
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''
    # Parse predicted_observation, predicted_reward (optional), predicted_inventory_diff
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)\n(?=predicted_reward:)', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)
    if not (obs_m and diff_m is not None):
        # Can't parse choice properly -> moderate penalty
        return -0.5
    obs = obs_m.group(1).strip().lower()
    inv = diff_m.group(1)
    # Check whether action is a "focus on" action
    m = re.match(r'(?i)^\s*focus on\s+(.+?)(?:\s+in\s+inventory\s*)?$', action.strip())
    if not m:
        # Rule not applicable
        return 0.0
    obj = m.group(1).strip().lower()
    # Simplify object token for matching (take main noun words)
    # e.g., "the thermometer" or "glass cup" -> check that at least one significant token appears in obs
    obj_tokens = [t for t in re.split(r'[\s,]+', obj) if t and t not in ('the','a','an','your','my','in','inventory')]
    obj_tokens = obj_tokens or [obj]
    # Condition 1: observation mentions focusing and mentions object token
    ok_obs_focus = 'focus' in obs or 'you focus' in obs
    ok_obs_obj = any(tok in obs for tok in obj_tokens)
    ok_obs = ok_obs_focus and ok_obs_obj
    # Condition 2: inventory diff must be empty (no '+ ' or '- ' lines)
    inv_lines = [ln.strip() for ln in inv.splitlines() if ln.strip()]
    has_inventory_change = any(ln.startswith('+ ') or ln.startswith('- ') for ln in inv_lines)
    # Scoring logic
    if ok_obs and not has_inventory_change:
        return 1.0
    if ok_obs and has_inventory_change:
        # Correct observation but incorrectly mutates inventory -> strong penalty
        return -0.8
    if (not ok_obs) and not has_inventory_change:
        # Observation wrong but no inventory corruption -> mild penalty
        return -0.3
    # Both wrong: strong penalty
    return -1.0

# Rule 6
def rule_reward(state, action, choice):
    import re
    # Helper to extract current_step_action from state if action not provided
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', (state or ""))
        action = m.group(1).strip() if m else ''
    action_l = (action or '').strip().lower()

    # Only apply this rule to the specified non-inventory-changing verbs
    verbs = ['open ', 'close ', 'activate ', 'deactivate ', 'focus on ', 'go to ']
    applies = False
    if any(action_l.startswith(v) for v in verbs) or re.match(r'(?i)^\s*wait', action_l):
        applies = True
    if not applies:
        return 0.0

    # Extract predicted fields from choice text
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)(?:\npredicted_reward:|$)', choice or "")
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice or "")
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice or "")

    # If any field is missing, be conservative: give a small negative instead of a large one
    if not (obs_m and rew_m and diff_m is not None):
        # smaller penalty to avoid false negatives when the model omits a field
        return -0.2

    obs = (obs_m.group(1) or "").strip().lower()
    try:
        rew = float(rew_m.group(1))
    except:
        rew = 0.0
    inv_diff_text = diff_m.group(1) or ""
    inv_lines = [ln.strip() for ln in inv_diff_text.splitlines() if ln.strip()]

    score = 0.0

    # Helper: normalize an inventory line to a base-name for matching property updates.
    def normalize_base_name(line):
        # remove leading + or - and whitespace
        s = line.lstrip('+-').strip().lower()
        # cut off at common property separators (comma, "currently", "containing", "(")
        s = re.split(r',|\bcurrently\b|\bcontaining\b|\(|\bwhich is\b', s, 1)[0].strip()
        # remove leading articles
        s = re.sub(r'^(a |an |the )', '', s).strip()
        # collapse spaces
        s = re.sub(r'\s+', ' ', s)
        return s

    # Check inventory diff: allow paired property updates (e.g., thermometer reading changed)
    plus_lines = [ln for ln in inv_lines if ln.startswith('+')]
    minus_lines = [ln for ln in inv_lines if ln.startswith('-')]
    other_lines = [ln for ln in inv_lines if not (ln.startswith('+') or ln.startswith('-'))]

    # If there are other unexpected lines, treat conservatively (mild penalty)
    if other_lines:
        score -= 0.25

    # If there are no +/- changes at all, that's fine for these actions
    if not plus_lines and not minus_lines:
        score += 0.5
    else:
        # Try to match each plus line to a minus line by normalized base name.
        plus_bases = [normalize_base_name(ln) for ln in plus_lines]
        minus_bases = [normalize_base_name(ln) for ln in minus_lines]

        # Count matches
        unmatched_plus = list(plus_bases)
        unmatched_minus = list(minus_bases)
        # Attempt greedy matching
        for p in plus_bases:
            if p in unmatched_minus:
                unmatched_plus.remove(p)
                unmatched_minus.remove(p)

        # If all plus/minus are matched pairwise -> property updates only -> allow
        if not unmatched_plus and not unmatched_minus:
            # Accept property updates (e.g., thermometer reading changed) as not real inventory change
            score += 0.5
        else:
            # There are unmatched adds/removes -> likely real inventory change; penalize strongly
            # But be conservative: if unmatched items are only one and refer to same base with minor textual diffs, allow
            # (this is to avoid tiny formatting mismatches)
            def liberal_same(a, b):
                return a == b or a in b or b in a

            if len(unmatched_plus) == 1 and len(unmatched_minus) == 1 and liberal_same(unmatched_plus[0], unmatched_minus[0]):
                score += 0.3
            else:
                # Strong negative when actual add/remove of items is indicated
                return -0.8

    # Check predicted_observation contains an expected phrase for the verb (conservative matching)
    ok_obs = False
    if action_l.startswith('open '):
        obj = action_l[len('open '):].strip()
        ok_obs = ('is now open' in obs) or ('already open' in obs) or (obj and obj in obs and ('open' in obs))
    elif action_l.startswith('close '):
        obj = action_l[len('close '):].strip()
        ok_obs = ('is now closed' in obs) or ('already closed' in obs) or (obj and obj in obs and ('closed' in obs))
    elif action_l.startswith('activate '):
        ok_obs = ('now activated' in obs) or ('is now activated' in obs) or ('now on' in obs) or ('turned on' in obs)
    elif action_l.startswith('deactivate '):
        ok_obs = ('now deactivated' in obs) or ('is now deactivated' in obs) or ('now off' in obs) or ('turned off' in obs)
    elif action_l.startswith('focus on '):
        obj = action_l[len('focus on '):].strip()
        ok_obs = ('you focus on the ' in obs) or (obj and ('you focus on ' + obj in obs)) or ('you focus on' in obs)
    elif action_l.startswith('go to '):
        ok_obs = ('you move to' in obs) or ('you go to' in obs) or ('you move' in obs)
    elif re.match(r'(?i)^\s*wait', action_l):
        ok_obs = ('you decide to wait' in obs) or ('you wait' in obs) or ('you do nothing' in obs)

    if ok_obs:
        score += 0.5
    else:
        # Missing or non-matching confirmation phrase: smaller penalty than before
        score -= 0.5

    # Reward sanity: non-inventory actions typically small; penalize extreme rewards conservatively
    if rew < -0.1 or rew > 0.6:
        # if everything else was correct, give partial credit; else penalize a bit
        if score >= 1.0:
            score -= 0.4
        else:
            score -= 0.2

    # Clamp to [-1, 1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return score

# Rule 7
def rule_reward(state, action, choice):
    import re
    # If action not given, extract from state
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
    inv = diff_m.group(1)
    # Only apply rule to "look around" actions
    if not re.match(r'(?i)^\s*look around\s*$', action.strip()):
        return 0.0
    # Check observation contains room description phrase
    obs_ok = ('this room is called' in obs) or ('in it, you see' in obs)
    # Check for any inventory change lines starting with + or -
    inv_lines = [ln for ln in inv.splitlines() if ln.strip()]
    inv_has_changes = any(ln.strip().startswith(('+', '-')) for ln in inv_lines)
    # Scoring per specification
    if obs_ok and not inv_has_changes:
        return 1.0
    if obs_ok and inv_has_changes:
        return 0.2
    if (not obs_ok) and (not inv_has_changes):
        return -0.5
    # not obs_ok and inv_has_changes
    return -1.0

# Rule 8
def rule_reward(state, action, choice):
    import re
    # Extract action from state if not provided
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''
    # Parse choice fields
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)(?:\npredicted_reward:|\Z)', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)
    if not (obs_m and rew_m and diff_m is not None):
        # malformed choice
        return -0.5
    obs = obs_m.group(1).strip().lower()
    inv = diff_m.group(1).strip()
    # Only apply this rule for 'open' actions
    if not re.match(r'(?i)^\s*open\b', action.strip()):
        return 0.0
    # Check observation: should say something like 'is now open' or 'already open'
    obs_ok = bool(re.search(r'(?i)\b(is now open|already open)\b', obs))
    # Check inventory diff: should be empty (no '+' or '-' lines and no other non-empty text)
    inv_lines = [ln.strip() for ln in inv.splitlines() if ln.strip()]
    inv_changed = False
    for ln in inv_lines:
        if ln.startswith('+') or ln.startswith('-'):
            inv_changed = True
            break
        # if any non-empty text present, consider it a change
        inv_changed = True
    # Scoring policy
    if obs_ok and not inv_changed:
        return 1.0
    if obs_ok and inv_changed:
        return 0.0
    if (not obs_ok) and (not inv_changed):
        return -0.2
    # not obs_ok and inv_changed
    return -0.6

# Rule 9
def rule_reward(state, action, choice):
    import re
    # Extract action from state if not provided
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''
    # Parse choice into fields
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)(?:\n(?=predicted_reward:)|\Z)', choice)
    rew_m = re.search(r'(?m)predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)
    # If parsing failed, return a mild penalty
    if not (obs_m and rew_m and diff_m is not None):
        return -0.5
    obs = obs_m.group(1).strip().lower()
    inv_text = diff_m.group(1)
    # Only apply this rule to "open ..." actions
    if not re.match(r'(?i)^\s*open\b', action):
        return 0.0
    # Condition A: observation should indicate the object is now open or already open
    ok_obs = ('is now open' in obs) or ('already open' in obs)
    # Condition B: inventory diff should be empty (no non-empty + or - lines)
    inv_lines = [ln.strip() for ln in inv_text.splitlines() if ln.strip()]
    has_changes = any(ln.startswith('+') or ln.startswith('-') for ln in inv_lines)
    ok_inv = not has_changes and len(inv_lines) == 0
    # Scoring: full credit if both conditions met, negative if action matched but failed
    if ok_obs and ok_inv:
        return 1.0
    else:
        return -0.5

# Rule 10
# Task group: boil
def rule_reward(state, action, choice):
    """
    Returns a float in [-1, 1].

    Applies only for a small set of exact actions. Parses `choice` for the
    labeled fields predicted_observation:, predicted_reward:, predicted_inventory_diff:
    The observation may span multiple lines between the obs and reward labels;
    inventory diff is everything after predicted_inventory_diff:. If parsing fails
    or action doesn't match one of the targeted actions, return 0.0.
    """
    import math, re

    def extract_action(act, state_text):
        if act and act.strip():
            return act.strip()
        # prefer the last occurrence of 'current_step_action:' if present
        key = 'current_step_action:'
        idx = state_text.rfind(key)
        if idx == -1:
            return ''
        # take remainder of that line
        rest = state_text[idx + len(key):].splitlines()[0].strip()
        return rest

    def parse_choice(text):
        # returns (obs (str), reward (float), inv_text (str)) or (None, None, None) on parse fail
        if not isinstance(text, str):
            return (None, None, None)
        lines = text.splitlines()
        # find indices
        obs_idx = rew_idx = inv_idx = None
        for i, l in enumerate(lines):
            if l.startswith('predicted_observation:') and obs_idx is None:
                obs_idx = i
            if l.startswith('predicted_reward:') and rew_idx is None:
                rew_idx = i
            if l.startswith('predicted_inventory_diff:') and inv_idx is None:
                inv_idx = i
        if obs_idx is None or rew_idx is None or inv_idx is None:
            return (None, None, None)
        if not (obs_idx < rew_idx < inv_idx):
            return (None, None, None)
        # extract observation: content from obs_idx line after the label plus any intermediate lines up to rew_idx
        obs_first = lines[obs_idx][len('predicted_observation:'):].lstrip()
        if obs_first == '':
            obs_lines = lines[obs_idx+1:rew_idx]
        else:
            obs_lines = [obs_first] + (lines[obs_idx+1:rew_idx] if obs_idx+1 < rew_idx else [])
        obs = '\n'.join([ln.rstrip() for ln in obs_lines]).strip()

        # extract reward: expect it on the rew_idx line
        rew_text = lines[rew_idx][len('predicted_reward:'):].strip()
        try:
            rew_val = float(rew_text)
        except Exception:
            return (None, None, None)

        # extract inventory diff: content after the inv_idx label (rest of that line plus following lines)
        inv_first = lines[inv_idx][len('predicted_inventory_diff:'):].lstrip()
        inv_tail = []
        if inv_first == '':
            # take remaining lines as inventory diff
            if inv_idx+1 <= len(lines)-1:
                inv_tail = [ln.rstrip() for ln in lines[inv_idx+1:]]
            inv_text = '\n'.join(inv_tail).strip()
        else:
            extra = lines[inv_idx+1:] if inv_idx+1 < len(lines) else []
            inv_text = '\n'.join([inv_first] + [ln.rstrip() for ln in extra]).strip()

        return (obs, rew_val, inv_text)

    # determine action
    act = extract_action(action, state or "")

    # Only handle the specific actions below
    targets = {
        "open door to hallway",
        "open door to kitchen",
        "examine soap in metal pot",
        "activate stove",
        "use thermometer in inventory on liquid soap",
        "use thermometer in inventory on liquid rubber"
    }
    if act not in targets:
        return 0.0

    obs, rew, inv_text = parse_choice(choice)
    if obs is None or rew is None or inv_text is None:
        # parsing failed; be conservative
        return 0.0

    # helper: inventory empty check (no non-blank lines)
    inv_nonblank = any((line.strip() != "") for line in inv_text.splitlines()) if inv_text is not None else False
    inv_empty = not inv_nonblank

    # Common tolerances
    ZERO_TOL = 1e-6
    THERM_REWARD = 0.22
    THERM_TOL = 0.05

    # Action-specific checks and scoring
    if act in ("open door to hallway", "open door to kitchen"):
        # Expect exact observation "The door is now open.", zero reward, and empty inventory diff
        if obs == "The door is now open." and math.isclose(rew, 0.0, abs_tol=ZERO_TOL) and inv_empty:
            return 1.0
        # parsed but inconsistent for this exact deterministic action -> penalize
        return -1.0

    if act == "examine soap in metal pot":
        # Only enforce reward of 0.0 for this examine action
        if math.isclose(rew, 0.0, abs_tol=ZERO_TOL):
            return 1.0
        else:
            return -1.0

    if act == "activate stove":
        # Expected exact broken-stove observation and zero reward
        correct_obs = "The stove appears broken, and can't be activated or deactivated."
        lower_obs = obs.lower()
        if obs == correct_obs and math.isclose(rew, 0.0, abs_tol=ZERO_TOL):
            return 1.0
        # strongly penalize continuations that claim the stove became activated
        if ("activated" in lower_obs) or ("turned on" in lower_obs) or ("is now on" in lower_obs) or ("is now activated" in lower_obs):
            return -1.0
        # otherwise be conservative and do not influence
        return 0.0

    if act == "use thermometer in inventory on liquid soap":
        expected_obs = "the thermometer measures a temperature of 507 degrees celsius"
        if obs == expected_obs:
            # best score when reward close and no inventory changes
            if abs(rew - THERM_REWARD) <= THERM_TOL and inv_empty:
                return 1.0
            # observation correct but reward/inventory slightly off -> reduced positive
            score = 0.8 if abs(rew - THERM_REWARD) <= 0.2 else 0.5
            if not inv_empty:
                score -= 0.2
            return max(-1.0, min(1.0, score))
        # if observation contains a numeric temperature, give graded scores based on closeness
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*degrees\s*celsius", obs, re.IGNORECASE)
        if m:
            try:
                temp = float(m.group(1))
                diff = abs(temp - 507.0)
                if diff <= 5.0:
                    score = 0.6 if abs(rew - THERM_REWARD) <= THERM_TOL else 0.3
                elif diff <= 20.0:
                    score = 0.2 if abs(rew - THERM_REWARD) <= THERM_TOL else -0.2
                else:
                    score = -0.5
                if not inv_empty:
                    score -= 0.2
                return max(-1.0, min(1.0, score))
            except Exception:
                return -0.5
        # observation not matching expected form -> moderate penalty
        return -0.5

    if act == "use thermometer in inventory on liquid rubber":
        expected_obs = "the thermometer measures a temperature of 235 degrees celsius"
        if obs == expected_obs and inv_empty and math.isclose(rew, THERM_REWARD, abs_tol=ZERO_TOL):
            return 1.0
        if obs == expected_obs and inv_empty:
            # observation correct but reward off
            return 0.5
        if math.isclose(rew, THERM_REWARD, abs_tol=ZERO_TOL) and obs != expected_obs:
            # reward matches but observation wrong -> penalize
            return -0.4
        # otherwise parsed but inconsistent -> penalize conservatively
        return -1.0

    # fallback conservative
    return 0.0

# Rule 11
# Task group: change the
def rule_reward(state, action, choice):
    """
    Returns a float in [-1.0, 1.0].
    Applies only for a small set of exact actions. Parses choice for:
      predicted_observation:, predicted_reward:, predicted_inventory_diff:
    Conservative behavior:
      - If parsing fails or action doesn't match, return 0.0 (abstain).
      - If observation matches expected and inventory diffs/reward also match -> +1.0.
      - If observation matches but inventory diffs missing/incorrect when they must be present -> -1.0.
      - If observation matches but reward deviates -> conservative penalty (-0.5) except a few cases with graded scores.
      - For lead thermometer, allow graded score by numeric temp deviation with additional penalties if inventory diffs present or reward far.
    """
    import re
    import math

    try:
        # Determine action: prefer provided action, otherwise extract from state line 'current_step_action:'
        act = action.strip() if (action is not None and action.strip() != "") else ""
        if not act:
            m = re.search(r'current_step_action:\s*(.*)', state)
            if m:
                act = m.group(1).strip()

        # Only handle these exact actions
        allowed_actions = {
            "examine ice cream",
            "go to kitchen",
            "open door to outside",
            "use thermometer in inventory on soap in metal pot",
            "use thermometer in inventory on soap",
            "open cupboard",
            "use thermometer in inventory on rubber",
            "activate blast furnace",
            "use thermometer in inventory on lead",
        }
        if act not in allowed_actions:
            return 0.0

        # Parse the choice into the three labeled parts. Be tolerant to ordering but require all three headers.
        if not isinstance(choice, str):
            return 0.0
        lines = choice.splitlines()

        idx_obs = idx_rew = idx_inv = None
        for i, L in enumerate(lines):
            if L.startswith("predicted_observation:"):
                if idx_obs is None:
                    idx_obs = i
            elif L.startswith("predicted_reward:"):
                if idx_rew is None:
                    idx_rew = i
            elif L.startswith("predicted_inventory_diff:"):
                if idx_inv is None:
                    idx_inv = i
        if idx_obs is None or idx_rew is None or idx_inv is None:
            return 0.0

        # Extract observation (text after first colon on that line)
        pred_obs = lines[idx_obs].split("predicted_observation:", 1)[1].strip()

        # Extract reward (expect a numeric token after header)
        rew_text = lines[idx_rew].split("predicted_reward:", 1)[1].strip()
        try:
            pred_reward = float(rew_text)
        except Exception:
            return 0.0

        # Extract inventory-diff: content may be on the header line after the colon and/or subsequent lines until end
        inv_after = lines[idx_inv].split("predicted_inventory_diff:", 1)[1]
        inv_lines = []
        if inv_after is not None and inv_after.strip() != "":
            inv_lines.append(inv_after.strip())
        for L in lines[idx_inv + 1:]:
            # stop if another top-level header unexpectedly appears
            if L.startswith("predicted_observation:") or L.startswith("predicted_reward:") or L.startswith("predicted_inventory_diff:"):
                break
            if L.strip() != "":
                inv_lines.append(L.strip())
        inv_nonempty = any((ln.strip() != "") for ln in inv_lines)
        inv_set = set(inv_lines)

        # small tolerances
        tiny_tol = 1e-6

        # Helper: compare float with tiny tolerance
        def is_close(a, b, tol=1e-6):
            return abs(a - b) <= tol

        # Now per-action checks (conservative)
        # 1) examine ice cream
        if act == "examine ice cream":
            expected_obs = "ice cream"
            expected_reward = 0.26
            if pred_obs != expected_obs:
                return 0.0
            # observation matches; expect no inventory changes
            if inv_nonempty:
                return -1.0
            if is_close(pred_reward, expected_reward, tiny_tol):
                return 1.0
            else:
                return -0.5

        # 2) go to kitchen
        if act == "go to kitchen":
            expected_obs = "You move to the kitchen."
            expected_reward = 0.03
            if pred_obs != expected_obs:
                return 0.0
            # observation matches; reward must be expected
            if is_close(pred_reward, expected_reward, tiny_tol):
                return 1.0
            else:
                return -0.5

        # 3) open door to outside
        if act == "open door to outside":
            expected_obs = "The door is now open."
            expected_inv_set = {
                "+ a metal pot (containing liquid ice cream)",
                "- a metal pot (containing ice cream)",
                "+ a thermometer, currently reading a temperature of 9 degrees celsius",
                "- a thermometer, currently reading a temperature of 10 degrees celsius",
            }
            # If any of the required inventory diffs are missing -> strong penalty
            if not expected_inv_set.issubset(inv_set):
                return -1.0
            # required inventory diffs are present
            if pred_obs != expected_obs:
                # inventory matched but observation not exactly expected => mild penalty
                return -0.5
            # obs and inventory match; evaluate reward with graded tolerance
            expected_reward = 0.28
            diff = abs(pred_reward - expected_reward)
            if diff <= 0.05:
                return 1.0
            if diff <= 0.20:
                return 0.5
            return 0.0

        # 4) use thermometer in inventory on soap in metal pot
        if act == "use thermometer in inventory on soap in metal pot":
            expected_obs = "the thermometer measures a temperature of 138 degrees celsius"
            expected_reward = 0.26
            if pred_obs != expected_obs:
                return 0.0
            if inv_nonempty:
                return -1.0
            if is_close(pred_reward, expected_reward, tiny_tol):
                return 1.0
            else:
                return -1.0

        # 5) use thermometer in inventory on soap
        if act == "use thermometer in inventory on soap":
            expected_obs = "the thermometer measures a temperature of 127 degrees celsius"
            expected_reward = 0.25
            if pred_obs != expected_obs:
                return 0.0
            if inv_nonempty:
                return -1.0
            if is_close(pred_reward, expected_reward, tiny_tol):
                return 1.0
            else:
                return -1.0

        # 6) open cupboard
        if act == "open cupboard":
            expected_obs = "The cupboard is now open."
            expected_reward = 0.0
            if pred_obs != expected_obs:
                return 0.0
            if inv_nonempty:
                return -1.0
            if is_close(pred_reward, expected_reward, tiny_tol):
                return 1.0
            else:
                return -1.0

        # 7) use thermometer in inventory on rubber
        if act == "use thermometer in inventory on rubber":
            expected_obs = "the thermometer measures a temperature of 179 degrees celsius"
            expected_reward = 0.26
            if pred_obs != expected_obs:
                return 0.0
            if inv_nonempty:
                return -1.0
            if is_close(pred_reward, expected_reward, tiny_tol):
                return 1.0
            else:
                return -1.0

        # 8) activate blast furnace
        if act == "activate blast furnace":
            expected_reward = 0.03
            # This case primarily checks reward; abstain only if parsing failed (already checked)
            if is_close(pred_reward, expected_reward, tiny_tol):
                return 1.0
            else:
                return -1.0

        # 9) use thermometer in inventory on lead
        if act == "use thermometer in inventory on lead":
            # Expect observation to report a numeric temperature, ideally 275 degrees celsius
            expected_temp = 275.0
            expected_reward = 0.24
            # Try to extract numeric temperature from observed text
            m = re.search(r'(-?\d+(?:\.\d+)?)\s*degrees\s*celsius', pred_obs)
            if not m:
                # If observation doesn't report a numeric temperature matching pattern, abstain
                return 0.0
            try:
                reported_temp = float(m.group(1))
            except Exception:
                return 0.0
            temp_err = abs(reported_temp - expected_temp)
            # If inventory diffs are present when none are expected -> penalize
            if inv_nonempty:
                # if inventory changes claimed, conservative penalty
                return -0.5
            # If reported temp is exactly expected and reward matches -> full credit
            if temp_err <= 1e-6 and is_close(pred_reward, expected_reward, tiny_tol):
                return 1.0
            # Otherwise produce a conservative graded score:
            # small temp error -> small positive; moderate -> small positive; large -> negative
            # also require reward not too far from expected for positive scores
            reward_err = abs(pred_reward - expected_reward)
            if temp_err <= 5.0 and reward_err <= 0.05:
                return 0.5
            if temp_err <= 20.0 and reward_err <= 0.1:
                return 0.2
            # large deviation or reward far -> negative
            return -0.5

        # Default abstain
        return 0.0

    except Exception:
        return 0.0

# Rule 12
# Task group: determine if
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    try:
        # Helper: extract current action from state if action not provided or blank
        def action_from_state(s):
            if not s:
                return None
            for line in s.splitlines():
                if line.strip().startswith("current_step_action:"):
                    return line.split("current_step_action:", 1)[1].strip()
            return None

        act = action.strip() if isinstance(action, str) and action.strip() != "" else action_from_state(state)
        if act != "drop unknown substance":
            return 0.0

        if choice is None:
            return 0.0
        lines = choice.splitlines()

        # Find required prefixed lines and their indices
        idx_obs = idx_rew = idx_inv = None
        for i, ln in enumerate(lines):
            if idx_obs is None and ln.startswith("predicted_observation:"):
                idx_obs = i
            if idx_rew is None and ln.startswith("predicted_reward:"):
                idx_rew = i
            if idx_inv is None and ln.startswith("predicted_inventory_diff:"):
                idx_inv = i
        # Require all present and in order
        if idx_obs is None or idx_rew is None or idx_inv is None:
            return 0.0
        if not (idx_obs < idx_rew < idx_inv):
            return 0.0

        # Extract observation text (after the first colon)
        obs_text = lines[idx_obs].partition(":")[2].lstrip()

        # Extract and parse reward
        rew_text = lines[idx_rew].partition(":")[2].strip()
        try:
            pred_reward = float(rew_text)
        except Exception:
            return 0.0

        # Inventory diff: all subsequent non-empty lines after idx_inv
        inv_lines = []
        for ln in lines[idx_inv + 1:]:
            s = ln.strip()
            if s != "":
                inv_lines.append(s)

        # Expected exact values
        expected_obs = "You move the unknown substance O to the workshop."
        expected_inv = ["- unknown substance O"]

        if obs_text == expected_obs and inv_lines == expected_inv:
            # require reward exactly 0.0 (tolerance for float parsing)
            if abs(pred_reward - 0.0) < 1e-9:
                return 1.0
            else:
                return -1.0

        return 0.0

    except Exception:
        return 0.0

# Rule 13
# Task group: determine if
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    target_action = "pick up paper clip"

    # If action missing or blank, try to extract from state
    if (action is None or str(action).strip() == "") and state:
        for line in str(state).splitlines():
            if line.strip().startswith("current_step_action:"):
                action = line.split("current_step_action:", 1)[1].strip()
                break

    if action != target_action:
        return 0.0

    # Must parse the three required headers in order
    if choice is None:
        return 0.0
    lines = str(choice).splitlines()

    # Find indices of headers
    obs_idx = None
    rew_idx = None
    inv_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("predicted_observation:"):
            obs_idx = i
            break
    if obs_idx is None:
        return 0.0
    for i in range(obs_idx + 1, len(lines)):
        if lines[i].startswith("predicted_reward:"):
            rew_idx = i
            break
    if rew_idx is None:
        return 0.0
    for i in range(rew_idx + 1, len(lines)):
        if lines[i].startswith("predicted_inventory_diff:"):
            inv_idx = i
            break
    if inv_idx is None:
        return 0.0

    # Extract predicted_observation (text after colon on same line)
    try:
        predicted_observation = lines[obs_idx].split("predicted_observation:", 1)[1].strip()
    except Exception:
        return 0.0

    # Extract predicted_reward as float
    try:
        rew_text = lines[rew_idx].split("predicted_reward:", 1)[1].strip()
        predicted_reward = float(rew_text)
    except Exception:
        return 0.0

    # Collect inventory diff lines: remainder of header line plus subsequent lines
    inv_lines = []
    try:
        inv_rest = lines[inv_idx].split("predicted_inventory_diff:", 1)[1].strip()
        if inv_rest != "":
            inv_lines.append(inv_rest)
        for j in range(inv_idx + 1, len(lines)):
            # include non-empty and empty lines (strip whitespace)
            inv_lines.append(lines[j].strip())
    except Exception:
        return 0.0

    # Check for exact inventory addition line
    expected_inv_line = "+ A paper clip"
    has_add = any(line == expected_inv_line for line in inv_lines)

    if not has_add:
        # Inventory diff must include the exact addition; otherwise strongly penalize
        return -1.0

    # Inventory includes the expected addition; now check observation and reward exactly
    expected_obs = "You move the paper clip to the inventory."
    expected_reward = 0.0
    if predicted_observation == expected_obs and abs(predicted_reward - expected_reward) < 1e-8:
        return 1.0

    # Inventory correct but observation or reward differ -> conservative penalty
    return -0.8

# Rule 14
# Task group: determine if
def rule_reward(state, action, choice):
    """
    Returns:
      1.0  if action is one of the targeted open-door actions, the state shows that door already open,
           and choice parses as predicted_observation: "The door is already open.", predicted_reward: 0.0,
           and an empty predicted_inventory_diff.
     -1.0  if the rule applies and the choice parses but any of those fields differ.
      0.0  if the rule does not apply or parsing fails.
    """
    try:
        # Determine action: prefer explicit argument, else look for current_step_action in state
        act = None
        if action is not None and action.strip() != "":
            act = action.strip()
        else:
            for line in state.splitlines():
                if line.strip().startswith("current_step_action:"):
                    act = line.split("current_step_action:", 1)[1].strip()
                    break

        if act not in ("open door to kitchen", "open door to hallway"):
            return 0.0

        # Require state to indicate that the corresponding door is already open to avoid false positives.
        # Expect a phrase like "door to the kitchen (that is open)" or "door to the hallway (that is open)"
        if "kitchen" in act:
            required_phrase = "door to the kitchen (that is open)"
        else:
            required_phrase = "door to the hallway (that is open)"
        if required_phrase not in state:
            return 0.0

        # Parse the choice into the three expected headers.
        lines = choice.splitlines()

        # Find header line indices (allow leading whitespace)
        def find_header_index(header):
            for i, l in enumerate(lines):
                if l.lstrip().startswith(header):
                    return i
            return None

        i_obs = find_header_index("predicted_observation:")
        i_reward = find_header_index("predicted_reward:")
        i_inv = find_header_index("predicted_inventory_diff:")

        if i_obs is None or i_reward is None or i_inv is None:
            return 0.0
        if not (i_obs < i_reward < i_inv):
            return 0.0

        # Extract observation text: remainder of obs header line plus any intervening lines up to reward header
        obs_first = lines[i_obs].split("predicted_observation:", 1)[1].rstrip()
        obs_lines = [obs_first] if obs_first != "" else []
        for j in range(i_obs + 1, i_reward):
            obs_lines.append(lines[j].rstrip())
        obs_text = "\n".join([l for l in obs_lines]).strip()

        # Extract reward text: remainder of reward header line; if empty, include intervening lines up to inventory header
        reward_rest = lines[i_reward].split("predicted_reward:", 1)[1].strip()
        if reward_rest != "":
            reward_text = reward_rest
        else:
            extra = []
            for j in range(i_reward + 1, i_inv):
                if lines[j].strip() != "":
                    extra.append(lines[j].strip())
            reward_text = " ".join(extra).strip()

        # Extract inventory diff: remainder of inventory header line plus following lines
        inv_first = lines[i_inv].split("predicted_inventory_diff:", 1)[1].rstrip()
        inv_lines = []
        if inv_first != "":
            inv_lines.append(inv_first)
        for j in range(i_inv + 1, len(lines)):
            inv_lines.append(lines[j].rstrip())
        # Normalize inventory lines by removing blank lines
        inv_nonempty = [l for l in inv_lines if l.strip() != ""]

        # Parse reward as float
        try:
            reward_val = float(reward_text)
        except Exception:
            return 0.0

        # Expected exact values
        expected_obs = "The door is already open."
        expected_reward = 0.0

        if obs_text == expected_obs and abs(reward_val - expected_reward) < 1e-6 and len(inv_nonempty) == 0:
            return 1.0
        else:
            # Rule applies (action + state matched) and choice parsed, but fields disagree -> penalize
            return -1.0

    except Exception:
        # On any unexpected error, do not apply the rule
        return 0.0

# Rule 15
# Task group: determine whether
def rule_reward(state, action, choice):
    """
    Returns a float in [-1, 1] according to the merged conservative rule.
    - Only triggers for exact actions: "focus on blue box", "focus on red box", "focus on green box".
    - Parses predicted_observation, predicted_reward, predicted_inventory_diff from 'choice'.
    - On parse failure or non-matching action -> 0.0.
    - If observation != expected string -> -1.0.
    - If inventory diff is non-empty -> 0.0 (avoid penalizing unrelated inventory changes).
    - If observation matches and inventory empty -> +1.0 for exact reward (within tol), otherwise
      return -min(1.0, abs_diff / 0.4) where abs_diff is distance to the nearest expected reward.
    """
    try:
        # Helpers
        def extract_action_from_state(st):
            for line in st.splitlines():
                if line.strip().startswith("current_step_action:"):
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        return parts[1].strip()
            return ""

        # Determine action (use provided action if non-empty, otherwise fallback to state)
        act = (action or "").strip()
        if act == "":
            act = extract_action_from_state(state or "")

        # Only apply to these exact actions
        valid_actions = {"focus on blue box", "focus on red box", "focus on green box"}
        if act not in valid_actions:
            return 0.0

        # Parse the choice string for the three fields in a robust way.
        # We look for lines starting with the headers; inventory diff may span multiple lines.
        lines = (choice or "").splitlines()
        po_idx = None
        pr_idx = None
        pid_idx = None
        for i, ln in enumerate(lines):
            if ln.startswith("predicted_observation:") and po_idx is None:
                po_idx = i
            elif ln.startswith("predicted_reward:") and pr_idx is None:
                pr_idx = i
            elif ln.startswith("predicted_inventory_diff:") and pid_idx is None:
                pid_idx = i

        if po_idx is None or pr_idx is None or pid_idx is None:
            return 0.0

        # Enforce header order (observation before reward before inventory header)
        if not (po_idx < pr_idx < pid_idx):
            return 0.0

        # Extract observation (text after the first colon)
        predicted_observation = lines[po_idx].split("predicted_observation:", 1)[1].lstrip().rstrip()

        # Extract reward as float
        reward_str = lines[pr_idx].split("predicted_reward:", 1)[1].strip()
        try:
            predicted_reward = float(reward_str)
        except:
            return 0.0

        # Extract inventory diff: content after the inventory header plus any following lines
        inv_tail = lines[pid_idx].split("predicted_inventory_diff:", 1)[1]
        inv_lines = []
        if inv_tail is not None and inv_tail.strip() != "":
            inv_lines.append(inv_tail.rstrip())
        # include subsequent lines as part of inventory diff
        for j in range(pid_idx + 1, len(lines)):
            inv_lines.append(lines[j].rstrip())

        # Normalize inventory diff: consider it non-empty if any non-blank line exists
        inv_nonblank = any((ln.strip() != "" for ln in inv_lines))

        # Expected observation string for the matched action
        color = None
        if act == "focus on blue box":
            color = "blue"
            expected_rewards = [0.57]
        elif act == "focus on green box":
            color = "green"
            expected_rewards = [0.57]
        else:  # focus on red box
            color = "red"
            # Accept either 0.57 or 0.60 as plausible expected values (choose nearest for scoring)
            expected_rewards = [0.57, 0.60]

        expected_obs = f"You focus on the {color} box."

        # If observation does not exactly match expected, strong negative score
        if predicted_observation != expected_obs:
            return -1.0

        # If inventory diff is non-empty, avoid penalizing here (could be unrelated) => no judgement
        if inv_nonblank:
            return 0.0

        # Observation matches and inventory empty: score reward closeness
        tol = 1e-6
        # distance to nearest expected reward
        abs_diff = min(abs(predicted_reward - r) for r in expected_rewards)

        if abs_diff <= tol:
            return 1.0

        # Conservative linear penalty: diff 0.4 -> -1.0, scaled and clamped.
        score = -min(1.0, abs_diff / 0.4)
        return float(score)

    except Exception:
        # On any unexpected parsing/runtime failure, don't judge
        return 0.0

# Rule 16
# Task group: determine whether
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    try:
        # Determine current action: prefer explicit action argument, otherwise extract from state
        act = action.strip() if action is not None and isinstance(action, str) and action.strip() != "" else None
        if not act and isinstance(state, str):
            for line in state.splitlines():
                if line.strip().startswith("current_step_action:"):
                    act = line.split("current_step_action:", 1)[1].strip()
                    break

        # Only apply this rule for the exact action "wait1"
        if act != "wait1":
            return 0.0

        # Parse the choice for headers and values
        lines = choice.splitlines() if isinstance(choice, str) else []
        obs_line_idx = rew_line_idx = inv_line_idx = None
        for i, ln in enumerate(lines):
            if ln.startswith("predicted_observation:") and obs_line_idx is None:
                obs_line_idx = i
            elif ln.startswith("predicted_reward:") and rew_line_idx is None:
                rew_line_idx = i
            elif ln.startswith("predicted_inventory_diff:") and inv_line_idx is None:
                inv_line_idx = i

        # All three headers must be present to apply this rule
        if obs_line_idx is None or rew_line_idx is None or inv_line_idx is None:
            return 0.0

        # Extract observation text (rest of the line after header)
        predicted_observation = lines[obs_line_idx].split("predicted_observation:", 1)[1].strip()

        # Extract reward text and parse to float
        rew_text = lines[rew_line_idx].split("predicted_reward:", 1)[1].strip()
        if rew_text == "":
            return 0.0
        try:
            predicted_reward = float(rew_text)
        except Exception:
            return 0.0

        # Inventory diff: content on the header line after colon plus any subsequent lines until end
        inv_header_content = lines[inv_line_idx].split("predicted_inventory_diff:", 1)[1].strip()
        inv_lines_after = lines[inv_line_idx+1:]
        # Consider inventory non-empty if header content has non-whitespace or any subsequent line has non-whitespace
        inv_nonempty = bool(inv_header_content and inv_header_content.strip()) or any((ln.strip() != "") for ln in inv_lines_after)

        # Expected observation and reward
        expected_obs = "You decide to wait for 1 iterations."
        expected_reward = 0.00

        # Only judge when observation matches exactly and inventory is empty (no changes)
        if predicted_observation == expected_obs and not inv_nonempty:
            # reward must equal 0.00 (allow tiny tolerance)
            if abs(predicted_reward - expected_reward) < 1e-6:
                return 1.0
            else:
                return -1.0

        # If observation/inventory don't match exact expected continuation, do not apply this rule
        return 0.0

    except Exception:
        return 0.0

# Rule 17
# Task group: determine whether
def rule_reward(state, action, choice):
    """
    Returns a float in [-1.0, 1.0] according to the merged rule:
    - Trigger only when current step action is exactly
      "pour jug into flower pot 6" or "pour jug into flower pot 9".
    - For action 6: require predicted_observation equals the canonical
      string and a parsed numeric predicted_reward == 0.00 -> +1.0.
      If observation matches but reward parses and != 0.00 -> -1.0.
      If headers missing or parsing fails -> 0.0.
    - For action 9: require parsed numeric predicted_reward == 0.00 ->
      0.0 (no reward). If parsed and != 0.00 -> -1.0. If missing or parse
      fails -> 0.0.
    """
    try:
        import re
        eps = 1e-6

        # Extract action: prefer provided action argument, else try to find in state
        act = (action or "").strip()
        if not act:
            m = re.search(r'current_step_action:\s*(.*)', state or "")
            if m:
                act = m.group(1).strip()

        if act not in ("pour jug into flower pot 6", "pour jug into flower pot 9"):
            return 0.0

        # Split choice into lines and find lines by header (allow leading whitespace)
        lines = (choice or "").splitlines()

        def find_header_line(prefix):
            for ln in lines:
                s = ln.lstrip()
                if s.startswith(prefix):
                    return s  # return trimmed-leading version so content comes after prefix
            return None

        # Common: look for predicted_reward line
        pr_line = find_header_line("predicted_reward:")
        # For action 6 we also need predicted_observation
        if act == "pour jug into flower pot 6":
            po_line = find_header_line("predicted_observation:")
            if po_line is None or pr_line is None:
                return 0.0
            # Extract observation content (text after the header)
            obs = po_line[len("predicted_observation:"):].strip()
            expected_obs = "You pour the contents of the jug into the flower pot 6."
            if obs != expected_obs:
                return 0.0
            # Parse predicted_reward numeric value
            m = re.match(r'predicted_reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$', pr_line)
            if not m:
                return 0.0
            try:
                reward_val = float(m.group(1))
            except Exception:
                return 0.0
            if abs(reward_val - 0.0) <= eps:
                return 1.0
            else:
                return -1.0

        # For action 9: only enforce reward == 0.00 (no observation requirement)
        if act == "pour jug into flower pot 9":
            if pr_line is None:
                return 0.0
            m = re.match(r'predicted_reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$', pr_line)
            if not m:
                return 0.0
            try:
                reward_val = float(m.group(1))
            except Exception:
                return 0.0
            correct = 0.0
            if abs(reward_val - correct) <= eps:
                # reward correct: rule does not award positive score for this case
                return 0.0
            else:
                return -1.0

        return 0.0

    except Exception:
        return 0.0

# Rule 18
# Task group: determine which
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    import math, re
    try:
        # Helper: extract current action from state if action arg empty
        act = action.strip() if action is not None else ""
        if not act:
            m = re.search(r"current_step_action:\s*(.+)", state)
            if m:
                act = m.group(1).strip()
            else:
                act = ""

        # We only apply to these exact actions
        targets = {
            "pick up stopwatch",
            "activate stopwatch in inventory",
            "look at inclined plane with a unknown material b surface"
        }
        if act not in targets:
            return 0.0

        # Parse the choice into three sections: predicted_observation:, predicted_reward:, predicted_inventory_diff:
        lines = [ln.rstrip("\n") for ln in choice.splitlines()]

        # find indices of headers (must exist and in order)
        idx_obs = idx_rwd = idx_inv = None
        for i, ln in enumerate(lines):
            if ln.startswith("predicted_observation:") and idx_obs is None:
                idx_obs = i
            elif ln.startswith("predicted_reward:") and idx_rwd is None:
                idx_rwd = i
            elif ln.startswith("predicted_inventory_diff:") and idx_inv is None:
                idx_inv = i

        if idx_obs is None or idx_rwd is None or idx_inv is None:
            return 0.0
        if not (idx_obs < idx_rwd < idx_inv):
            return 0.0

        # Extract observation (rest of the obs line)
        obs = lines[idx_obs].split("predicted_observation:", 1)[1].strip()

        # Extract reward (rest of reward line -> float)
        rew_text = lines[idx_rwd].split("predicted_reward:", 1)[1].strip()
        try:
            predicted_reward = float(rew_text)
        except:
            return 0.0

        # Extract inventory diff lines: content after header line plus following lines that start with + or -
        inv_first = lines[idx_inv].split("predicted_inventory_diff:", 1)[1].strip()
        inv_lines = []
        if inv_first:
            # if the header line contains an immediate diff entry, include it
            # accept it only if it looks like a diff line (+ or -) or treat as raw line
            inv_lines.append(inv_first)
        for ln in lines[idx_inv + 1:]:
            # Collect continuation lines that look like inventory diff entries (+/-) or non-empty lines (tolerate some formats)
            if ln.strip() == "":
                continue
            # allow lines that start with + or - or full sentences (be permissive but content checks below are strict)
            inv_lines.append(ln.strip())

        # Normalize inventory text for checks (lowercase for content checks when appropriate)
        inv_text_lower = "\n".join(inv_lines).lower()

        # Now check per-action expected continuations
        if act == "pick up stopwatch":
            expected_obs = "You move the stopwatch to the inventory."
            # observation exact
            obs_ok = (obs == expected_obs)
            # reward approximately 0.0
            rew_ok = math.isclose(predicted_reward, 0.0, abs_tol=1e-6)
            # inventory diff must include a '+' line mentioning stopwatch, deactivated, and 0 ticks (allow "0 tick" or "0 ticks")
            inv_ok = False
            for ln in inv_lines:
                ln_l = ln.lower()
                if ln_l.lstrip().startswith("+") and "stopwatch" in ln_l and "deactivated" in ln_l and ("0 ticks" in ln_l or "0 tick" in ln_l):
                    inv_ok = True
                    break
            if obs_ok and rew_ok and inv_ok:
                return 1.0
            else:
                # parsed but inconsistent -> penalize
                return -1.0

        elif act == "activate stopwatch in inventory":
            expected_obs = "The stopwatch is now activated."
            obs_ok = (obs == expected_obs)
            rew_ok = math.isclose(predicted_reward, 0.05, abs_tol=1e-6)
            # require a '+' line that mentions activated and "The time reads 1 ticks." (exact phrase expected for the plus)
            plus_ok = any(ln.strip() == "+ a stopwatch, which is activated. The time reads 1 ticks." for ln in inv_lines)
            # require a '-' line that mentions deactivated and a "The time reads <N> ticks." phrase (allow different N values)
            minus_ok = False
            for ln in inv_lines:
                ln_s = ln.strip()
                if ln_s.startswith("-") and "deactivated" in ln_s and "the time reads" in ln_s.lower() and "ticks" in ln_s.lower():
                    minus_ok = True
                    break
            if obs_ok and rew_ok and plus_ok and minus_ok:
                return 1.0
            else:
                return -1.0

        else:  # "look at inclined plane with a unknown material b surface"
            expected_obs = ("an inclined plane with a unknown material B surface, with: "
                            "a steel block approximately 62% down the plane")
            # observation exact (case-sensitive as in examples)
            if obs != expected_obs:
                return 0.0
            # reward must be ~0.0
            if not math.isclose(predicted_reward, 0.0, abs_tol=1e-6):
                return 0.0
            # Expect exactly the two tick-update lines (both present, order-insensitive)
            expected_plus = "+ a stopwatch, which is activated. The time reads 29 ticks."
            expected_minus = "- a stopwatch, which is activated. The time reads 28 ticks."
            has_plus = any(ln.strip() == expected_plus for ln in inv_lines)
            has_minus = any(ln.strip() == expected_minus for ln in inv_lines)
            if has_plus and has_minus:
                return 1.0
            else:
                # obs and reward correct but inventory diff missing/incomplete -> milder penalty
                return -0.5

    except Exception:
        # conservative fallback: do not penalize on unexpected errors/parsing exceptions
        return 0.0

# Rule 19
# Task group: determine which
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    try:
        # 1) Determine current action: prefer explicit argument if non-empty, else extract from state
        act = None
        if isinstance(action, str) and action.strip() != "":
            act = action.strip()
        else:
            for line in (state or "").splitlines():
                # allow possible leading whitespace
                if line.strip().startswith("current_step_action:"):
                    # take text after first colon
                    _, _, tail = line.partition(":")
                    act = tail.strip()
                    break
        if act is None:
            return 0.0

        # 2) Only apply for the two exact target actions
        target_f = "focus on inclined plane with a unknown material f surface"
        target_j = "focus on inclined plane with a unknown material j surface"
        if act != target_f and act != target_j:
            return 0.0

        # 3) Parse the choice text into the three labeled fields.
        # We are conservative: require all three labels to be present and parseable.
        lines = (choice or "").splitlines()
        obs = None
        rew = None
        inv_text = None

        # We'll scan lines, allowing leading whitespace before labels.
        i = 0
        while i < len(lines):
            ln = lines[i]
            ln_strip = ln.lstrip()
            if ln_strip.startswith("predicted_observation:"):
                # take remainder of same line
                obs = ln_strip.split("predicted_observation:", 1)[1].lstrip()
                i += 1
            elif ln_strip.startswith("predicted_reward:"):
                rew_text = ln_strip.split("predicted_reward:", 1)[1].strip()
                try:
                    rew = float(rew_text)
                except Exception:
                    return 0.0
                i += 1
            elif ln_strip.startswith("predicted_inventory_diff:"):
                # capture remainder of this line and any following lines as inventory diff
                inv_rest = ln_strip.split("predicted_inventory_diff:", 1)[1]
                # include following lines as part of inventory diff
                following = []
                j = i + 1
                while j < len(lines):
                    following.append(lines[j])
                    j += 1
                if following:
                    inv_text = (inv_rest + ("\n" + "\n".join(following))).rstrip()
                else:
                    inv_text = inv_rest.rstrip()
                break  # inventory diff is the last section per format; stop parsing
            else:
                i += 1

        # Require all three fields present
        if obs is None or rew is None or inv_text is None:
            return 0.0

        # 4) Expected observation depends on action variant (uppercase material letter)
        expected_obs = ("You focus on the inclined plane with a unknown material F surface."
                        if act == target_f
                        else "You focus on the inclined plane with a unknown material J surface.")

        # Conservative behavior: do not apply the rule (return 0.0) unless observation matches exactly
        if obs != expected_obs:
            return 0.0

        # Inventory diff must be empty (no non-whitespace content). If not empty, do not apply.
        if inv_text.strip() != "":
            return 0.0

        # 5) Score predicted_reward by conservative linear penalty around correct value 0.50.
        correct = 0.50
        diff = abs(rew - correct)
        # Map diff in [0, 0.5] linearly to score in [1.0, 0.0]; diffs >= 0.5 -> 0.0.
        if diff >= 0.5:
            score = 0.0
        else:
            score = 1.0 - (diff / 0.5)

        # Ensure numeric bounds
        if score < -1.0:
            score = -1.0
        if score > 1.0:
            score = 1.0
        return float(score)

    except Exception:
        return 0.0

# Rule 20
# Task group: find a
def rule_reward(state, action, choice):
    """
    Returns a float in [-1, 1] according to the merged conservative rule.
    Triggers only when the current_step_action equals one of the three specific actions.
    """
    import re, math

    # Helper: extract current_step_action from state if action is empty/falsey
    if not action:
        m = re.search(r'current_step_action:\s*(.*)', state or "")
        if not m:
            return 0.0
        action = m.group(1).strip()

    # Only apply to these exact action strings
    targets = {
        "move baby baby beaver in inventory to red box": "red",
        "move baby baby beaver in inventory to yellow box": "yellow",
        "move baby baby beaver in inventory to blue box": "blue",
    }
    action = (action or "").strip()
    if action not in targets:
        return 0.0

    color = targets[action]
    expected_observation = f"You move the beaver to the {color} box."
    expected_inv_line = "- a baby beaver"
    # Parse choice: require labels predicted_observation:, predicted_reward:, predicted_inventory_diff:
    if not choice:
        return 0.0
    lines = [ln.rstrip("\n") for ln in choice.splitlines()]

    po_text = None
    pr_text = None
    pid_index = None
    # Find label lines (first occurrences)
    for i, ln in enumerate(lines):
        if ln.startswith("predicted_observation:") and po_text is None:
            po_text = ln.split("predicted_observation:", 1)[1].strip()
        elif ln.startswith("predicted_reward:") and pr_text is None:
            pr_text = ln.split("predicted_reward:", 1)[1].strip()
        elif ln.startswith("predicted_inventory_diff:") and pid_index is None:
            # capture any inline text after the colon as first inventory line
            rest = ln.split("predicted_inventory_diff:", 1)[1].strip()
            pid_index = i
            if rest != "":
                # keep the inline content as a pseudo following line
                # we'll prepend it when collecting inventory lines
                lines[i] = "predicted_inventory_diff:"  # normalize current line
                lines.insert(i+1, rest)

    # All three required labels must be present
    if po_text is None or pr_text is None or pid_index is None:
        return 0.0

    # Extract predicted_observation (po_text already)
    predicted_observation = po_text

    # Extract predicted_reward
    try:
        predicted_reward = float(pr_text)
    except Exception:
        return 0.0

    # Collect inventory diff lines: lines after pid_index that look like +/- entries
    inv_lines = []
    for ln in lines[pid_index+1:]:
        if ln.strip() == "":
            continue
        stripped = ln.lstrip()
        if stripped.startswith(("+", "-")):
            inv_lines.append(stripped)
        else:
            # stop collecting when encountering a non +/- line to be conservative
            break

    # Check inventory contains the exact removal line
    inv_ok = any(ln.strip() == expected_inv_line for ln in inv_lines)

    # Reward checks: must be reasonably close to 0.17 (reject if > 0.1 away)
    err = abs(predicted_reward - 0.17)
    if not math.isfinite(predicted_reward):
        return 0.0

    if not inv_ok:
        # Strong requirement: if inventory change doesn't show the exact removal, penalize
        return -1.0

    if err > 0.1:
        # reward far from expected: penalize
        return -1.0

    # Observation keyword check (case-insensitive)
    obs_lower = (predicted_observation or "").lower()
    keywords_ok = ("move" in obs_lower) and ("beaver" in obs_lower) and (f"{color} box" in obs_lower)

    # Scoring logic:
    # - Full credit if observation exactly matches expected and reward is very close (tiny float error allowed)
    # - Partial credit if inventory & reward pass and observation contains the keywords
    # - Moderate penalty if inventory & reward pass but observation does not meaningfully match
    # - Slightly reduce score when reward is close but not exact (within 0.1)
    score = 0.0
    exact_reward = err <= 1e-6

    if predicted_observation.strip() == expected_observation and exact_reward:
        score = 1.0
    elif predicted_observation.strip() == expected_observation and err <= 0.1:
        # exact wording but small numeric reward deviation
        score = 0.8
    elif keywords_ok and exact_reward:
        score = 0.5
    elif keywords_ok and err <= 0.1:
        score = 0.4
    else:
        # inventory and reward passed but observation doesn't match keywords -> moderate penalty
        score = -0.5

    # Clamp to [-1, 1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return float(score)

# Rule 21
# Task group: find a
def rule_reward(state, action, choice):
    """
    Returns a float in [-1, 1].
    Triggers only for actions:
      - "move egg giant tortoise egg in inventory to red box"
      - "move egg giant tortoise egg in inventory to orange box"
      - "move egg giant tortoise egg in inventory to yellow box"
    """
    try:
        # normalize/obtain action string
        act = ""
        if action is not None and str(action).strip() != "":
            act = str(action).strip()
        else:
            # try to extract from state
            for line in (state or "").splitlines():
                if line.strip().startswith("current_step_action:"):
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        act = parts[1].strip()
                    break

        valid_actions = {
            "move egg giant tortoise egg in inventory to red box",
            "move egg giant tortoise egg in inventory to orange box",
            "move egg giant tortoise egg in inventory to yellow box",
        }
        if act not in valid_actions:
            return 0.0

        # Parse predicted fields from choice text
        lines = (choice or "").splitlines()
        obs_text = None
        rew_val = None
        inv_idx = None
        for i, ln in enumerate(lines):
            if ln.startswith("predicted_observation:"):
                obs_text = ln.split("predicted_observation:", 1)[1].strip()
            elif ln.startswith("predicted_reward:"):
                val = ln.split("predicted_reward:", 1)[1].strip()
                try:
                    rew_val = float(val)
                except Exception:
                    return 0.0
            elif ln.startswith("predicted_inventory_diff:"):
                inv_idx = i
                break
        if obs_text is None or rew_val is None or inv_idx is None:
            return 0.0

        # collect inventory diff lines after the header until next predicted_ label or end
        inv_lines = []
        for ln in lines[inv_idx+1:]:
            if ln.startswith("predicted_observation:") or ln.startswith("predicted_reward:") or ln.startswith("predicted_inventory_diff:"):
                break
            if ln.strip() != "":
                inv_lines.append(ln.strip())

        # expected observation per action
        expected_obs_map = {
            "move egg giant tortoise egg in inventory to red box":    "You move the giant tortoise to the red box.",
            "move egg giant tortoise egg in inventory to orange box": "You move the giant tortoise to the orange box.",
            "move egg giant tortoise egg in inventory to yellow box": "You move the giant tortoise to the yellow box.",
        }
        expected_obs = expected_obs_map[act]
        removal_line = "- a giant tortoise egg"
        has_removal = any(ln.strip() == removal_line for ln in inv_lines)

        # If observation does not exactly match expected, abstain (conservative)
        if obs_text != expected_obs:
            return 0.0

        # Now observation matches; handle per-action logic
        expected_reward = 0.17
        EPS = 1e-9

        if act.endswith("red box"):
            # Require inventory removal; if missing, moderate penalty
            if not has_removal:
                return -0.5
            # graded score by closeness to 0.17 (linear scaling factor 10)
            diff = abs(rew_val - expected_reward)
            score = 1.0 - min(1.0, diff * 10.0)  # yields in [0.0, 1.0]
            # ensure bounds
            if score < -1.0:
                score = -1.0
            if score > 1.0:
                score = 1.0
            return score

        elif act.endswith("orange box"):
            # observation matches; reward must equal expected for full credit
            if abs(rew_val - expected_reward) > EPS:
                return -1.0
            # reward matches; require removal line for full credit, else moderate penalty
            if has_removal:
                return 1.0
            else:
                return -0.5

        elif act.endswith("yellow box"):
            # observation matches; require removal line
            if not has_removal:
                return -0.5
            # if removal present, reward must be exact to get full credit; else strong penalty
            if abs(rew_val - expected_reward) <= EPS:
                return 1.0
            else:
                return -1.0

        # default safe fallback
        return 0.0

    except Exception:
        return 0.0

# Rule 22
# Task group: find a
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    import math

    target_action = "focus on egg parrot"
    # Determine actual action: prefer provided action, else extract from state
    act = None
    if action is not None and str(action).strip() != "":
        act = str(action).strip()
    else:
        # look for a line like "current_step_action: <action>"
        for line in state.splitlines():
            if line.strip().startswith("current_step_action:"):
                parts = line.split(":", 1)
                if len(parts) >= 2:
                    act = parts[1].strip()
                break

    if act != target_action:
        # Not the action this rule applies to
        return 0.0

    # Parse the choice into the three headers
    try:
        lines = choice.splitlines()
    except Exception:
        return 0.0

    obs_idx = None
    rew_idx = None
    inv_idx = None
    for i, ln in enumerate(lines):
        s = ln.lstrip()
        if s.startswith("predicted_observation:"):
            obs_idx = i
        elif s.startswith("predicted_reward:"):
            rew_idx = i
        elif s.startswith("predicted_inventory_diff:"):
            inv_idx = i

    # All three headers must be present to evaluate; otherwise abstain
    if obs_idx is None or rew_idx is None or inv_idx is None:
        return 0.0

    try:
        # Extract observation text (after first colon on that line)
        obs_line = lines[obs_idx].split(":", 1)
        predicted_observation = obs_line[1].strip() if len(obs_line) > 1 else ""

        # Extract reward string and parse float
        rew_line = lines[rew_idx].split(":", 1)
        predicted_reward_str = rew_line[1].strip() if len(rew_line) > 1 else ""
        predicted_reward = float(predicted_reward_str)

        # Inventory diff is the remainder of the choice starting after the inventory header line.
        # Include any text on the same line after the colon and any following lines.
        inv_first = lines[inv_idx].split(":", 1)
        inv_rest = inv_first[1] if len(inv_first) > 1 else ""
        remaining = [inv_rest] + lines[inv_idx+1:]
        inv_text = "\n".join(remaining).strip()
    except Exception:
        # Parsing error -> abstain
        return 0.0

    # Expected values
    expected_observation = "You focus on the parrot egg."
    expected_reward = 0.50
    reward_tol = 1e-6

    obs_ok = (predicted_observation == expected_observation)
    rew_ok = (abs(predicted_reward - expected_reward) <= reward_tol)
    inv_ok = (inv_text == "")

    if obs_ok and rew_ok and inv_ok:
        return 1.0
    else:
        # Parsing succeeded and action matched but fields differ -> penalize
        return -1.0

# Rule 23
# Task group: find a
def rule_reward(state, action, choice):
    """
    Returns a float in [-1,1].
    Applies only to a small set of exact current_step_action strings (or when that action is found in state).
    """
    try:
        # Helper: extract current_step_action from state if needed
        act = action.strip() if action and action.strip() != "" else None
        if not act:
            for line in (state or "").splitlines():
                if line.strip().startswith("current_step_action:"):
                    act = line.split("current_step_action:", 1)[1].strip()
                    break
        if act is None:
            return 0.0

        # Only handle these exact actions
        handled_moves = {
            "move flower pot 4 containing pea plant and soil in inventory to blue box": {
                "expected_obs": "You move the flower pot 4 to the blue box.",
                "expected_remove": "- a flower pot 4 (containing a pea plant in the reproducing stage with a tall height, soil)",
                "expected_reward": 0.17
            },
            "move flower pot 8 containing apple tree and soil in inventory to blue box": {
                "expected_obs": "You move the flower pot 8 to the blue box.",
                "expected_remove": "- a flower pot 8 (containing a apple tree in the reproducing stage, soil)",
                "expected_reward": 0.17
            }
        }

        # If the action is one of the strict move-to-blue-box cases
        if act in handled_moves:
            spec = handled_moves[act]
            # Parse predicted fields
            lines = [ln.rstrip("\n") for ln in (choice or "").splitlines()]
            if not lines:
                return 0.0
            try:
                i_obs = next(i for i,l in enumerate(lines) if l.startswith("predicted_observation:"))
                i_rew = next(i for i,l in enumerate(lines) if l.startswith("predicted_reward:"))
                i_inv = next(i for i,l in enumerate(lines) if l.startswith("predicted_inventory_diff:"))
            except StopIteration:
                return 0.0
            if not (i_obs < i_rew < i_inv):
                return 0.0
            obs_text = lines[i_obs].split("predicted_observation:",1)[1].strip()
            rew_text = lines[i_rew].split("predicted_reward:",1)[1].strip()
            try:
                pred_reward = float(rew_text)
            except:
                return 0.0
            inv_lines = [l.strip() for l in lines[i_inv+1:] if l.strip() != ""]

            # Require exact observation and presence of expected removal line
            if obs_text != spec["expected_obs"]:
                return 0.0
            if spec["expected_remove"] not in inv_lines:
                return 0.0
            # If obs and inventory match, enforce reward exactly
            if abs(pred_reward - spec["expected_reward"]) <= 1e-6:
                return 1.0
            else:
                return -1.0

        # Handle "pick up flower pot 5"
        if act == "pick up flower pot 5":
            lines = (choice or "").splitlines()
            if not lines:
                return 0.0
            obs_idx = reward_idx = inv_idx = None
            for i, ln in enumerate(lines):
                if ln.startswith("predicted_observation:"):
                    obs_idx = i
                elif ln.startswith("predicted_reward:"):
                    reward_idx = i
                elif ln.startswith("predicted_inventory_diff:"):
                    inv_idx = i
            if obs_idx is None or reward_idx is None or inv_idx is None:
                return 0.0
            # Extract observation text (may span lines until reward_idx)
            obs_first = lines[obs_idx].split("predicted_observation:",1)[1].lstrip()
            if reward_idx > obs_idx + 1:
                obs_middle = "\n".join(lines[obs_idx+1:reward_idx])
                obs_text = (obs_first + ("\n" + obs_middle if obs_middle else "")).strip()
            else:
                obs_text = obs_first.strip()
            # Reward
            reward_text = lines[reward_idx].split("predicted_reward:",1)[1].strip()
            try:
                pred_reward = float(reward_text)
            except:
                return 0.0
            # Inventory diffs (all lines after inv_idx)
            inv_lines = []
            first_inv_part = lines[inv_idx].split("predicted_inventory_diff:",1)[1].strip()
            if first_inv_part != "":
                inv_lines.append(first_inv_part)
            for ln in lines[inv_idx+1:]:
                if ln.strip():
                    inv_lines.append(ln.strip())

            # Checks
            obs_ok = "You move the flower pot 5 to the inventory" in obs_text
            inv_ok = False
            for il in inv_lines:
                low = il.lower()
                if "flower pot 5" in low and "cherry" in low and "reproducing" in low:
                    inv_ok = True
                    break
            # Strong negative if inventory shows clearly wrong object
            for il in inv_lines:
                if "shovel" in il.lower():
                    return -0.8

            reward_ok = abs(pred_reward - 0.08) <= 1e-6

            if obs_ok and inv_ok and reward_ok:
                return 1.0
            if obs_ok and inv_ok and not reward_ok:
                return 0.5
            if (obs_ok and reward_ok and not inv_ok) or (inv_ok and reward_ok and not obs_ok):
                return 0.2
            # If parsed but none of the key pieces match, small negative
            return -0.5

        # Handle "go to kitchen" (conservative: only when state mentions relevant pot)
        if act == "go to kitchen":
            # Only apply when state mentions a relevant flower pot (7 or 9)
            has_pot9 = ("flower pot 9" in (state or ""))
            has_pot7 = ("flower pot 7" in (state or ""))
            if not (has_pot9 or has_pot7):
                return 0.0

            lines = (choice or "").splitlines()
            if not lines:
                return 0.0
            try:
                i_obs = next(i for i,l in enumerate(lines) if l.startswith("predicted_observation:"))
                i_rew = next(i for i,l in enumerate(lines) if l.startswith("predicted_reward:"))
                i_inv = next(i for i,l in enumerate(lines) if l.startswith("predicted_inventory_diff:"))
            except StopIteration:
                return 0.0
            if not (i_obs < i_rew < i_inv):
                return 0.0
            obs_text = lines[i_obs].split("predicted_observation:",1)[1].strip()
            rew_text = lines[i_rew].split("predicted_reward:",1)[1].strip()
            try:
                pred_reward = float(rew_text)
            except:
                return 0.0
            inv_lines = [l.strip() for l in lines[i_inv+1:] if l.strip() != ""]

            # Observation must match exactly
            obs_ok = (obs_text == "You move to the kitchen.")
            # Two kinds of expected inventory updates depending on pot present in state
            if has_pot9:
                reward_ok = abs(pred_reward - 0.08) <= 1e-6
                plus_ok = any(l.lstrip().startswith('+') and "a flower pot 9" in l and "purple flower" in l for l in inv_lines)
                minus_ok = any(l.lstrip().startswith('-') and "a flower pot 9" in l for l in inv_lines)
                inv_ok = plus_ok and minus_ok
                if obs_ok and reward_ok and inv_ok:
                    return 1.0
                if obs_ok and reward_ok and not inv_ok:
                    return -0.6
                if (obs_ok and not reward_ok) or (not obs_ok and reward_ok):
                    return -0.9
                if not obs_ok and not reward_ok:
                    return -1.0
                return 0.0

            if has_pot7:
                # For pot 7 we expect the reward 0.00 and both + and - lines mentioning pot 7 and a flower/apple
                reward_ok = abs(pred_reward - 0.00) <= 1e-6
                plus_ok = any(l.lstrip().startswith('+') and "a flower pot 7" in l and ("flower" in l.lower() or "apple" in l.lower()) for l in inv_lines)
                minus_ok = any(l.lstrip().startswith('-') and "a flower pot 7" in l for l in inv_lines)
                inv_ok = plus_ok and minus_ok
                if not obs_ok:
                    return -1.0
                if not reward_ok:
                    return -1.0
                if inv_ok:
                    return 1.0
                else:
                    return -1.0

        # If action is not one of the targeted ones, do not apply the rule
        return 0.0

    except Exception:
        # On any unexpected error, do not apply the rule
        return 0.0

# Rule 24
# Task group: find the
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    try:
        # Determine action: prefer provided parameter, otherwise look in state
        act = (action or "").strip()
        if not act:
            marker = "current_step_action:"
            for line in (state or "").splitlines():
                if line.strip().startswith(marker):
                    act = line.split(marker, 1)[1].strip()
                    break

        # This rule applies only for the exact action
        if act != "open door to outside":
            return 0.0

        # Helper: extract single-line field value after "key:"
        def extract_single_line_field(text, key):
            if not text:
                return None
            marker = key + ":"
            idx = text.find(marker)
            if idx == -1:
                return None
            rest = text[idx + len(marker):]
            # take up to the next newline (or whole rest if none)
            if '\n' in rest:
                return rest.split('\n', 1)[0].strip()
            return rest.strip()

        # Helper: extract trailing field (everything after "key:")
        def extract_trailing_field(text, key):
            if not text:
                return None
            marker = key + ":"
            idx = text.find(marker)
            if idx == -1:
                return None
            return text[idx + len(marker):].lstrip()

        obs = extract_single_line_field(choice, "predicted_observation")
        rew_s = extract_single_line_field(choice, "predicted_reward")
        inv = extract_trailing_field(choice, "predicted_inventory_diff")

        # If any required field is missing, abstain (don't penalize)
        if obs is None or rew_s is None or inv is None:
            return 0.0

        # Parse reward to float; if parse fails, abstain
        try:
            rew = float(rew_s)
        except Exception:
            return 0.0

        # Expectations:
        cond_obs = (obs == "The door is now open.")
        cond_rew = abs(rew - 0.0) < 1e-9
        cond_inv = (inv.strip() == "")

        # Positive only if all expectations met
        if cond_obs and cond_rew and cond_inv:
            return 1.0

        # If a numeric reward was provided but it's not 0.0, strongly penalize
        if not cond_rew:
            return -1.0

        # Fields parsed but some expectation(s) failed -> penalize (conservative)
        return -1.0

    except Exception:
        return 0.0

# Rule 25
# Task group: find the
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    import math

    def extract_current_action_from_state(s):
        for line in s.splitlines():
            if line.strip().startswith("current_step_action:"):
                return line.split("current_step_action:", 1)[1].strip()
        return ""

    # Use provided action (string) if non-empty, else fall back to state's current_step_action
    act = ""
    if isinstance(action, str) and action.strip() != "":
        act = action.strip()
    else:
        act = extract_current_action_from_state(state or "")

    # Only apply to the two specific actions
    target_map = {
        "focus on baby baby elephant": ("You focus on the baby elephant.", 0.50),
        "focus on baby baby hedgehog": ("You focus on the baby hedgehog.", 0.17),
    }
    if act not in target_map:
        return 0.0

    expected_observation, expected_reward = target_map[act]

    # Basic validation
    if choice is None or not isinstance(choice, str):
        return 0.0

    # Split lines and find header lines (allow whitespace)
    lines = [ln.rstrip("\n") for ln in choice.splitlines()]

    obs_idx = None
    rew_idx = None
    inv_idx = None
    for i, ln in enumerate(lines):
        stripped = ln.lstrip()
        if stripped.startswith("predicted_observation:") and obs_idx is None:
            obs_idx = i
        elif stripped.startswith("predicted_reward:") and rew_idx is None:
            rew_idx = i
        elif stripped.startswith("predicted_inventory_diff:") and inv_idx is None:
            inv_idx = i

    # Must find all three headers in that order to proceed; otherwise be conservative
    if obs_idx is None or rew_idx is None or inv_idx is None:
        return 0.0
    if not (obs_idx < rew_idx < inv_idx):
        return 0.0

    # Extract observation text (rest of the obs line after the colon)
    try:
        obs_line = lines[obs_idx]
        obs_text = obs_line.split("predicted_observation:", 1)[1].strip()
    except Exception:
        return 0.0

    # Extract reward and parse float
    try:
        rew_line = lines[rew_idx]
        rew_text = rew_line.split("predicted_reward:", 1)[1].strip()
        predicted_reward = float(rew_text)
    except Exception:
        return 0.0

    # Extract inventory diff content: rest of the inv header line after colon plus any subsequent lines
    try:
        inv_header_part = lines[inv_idx].split("predicted_inventory_diff:", 1)[1]
    except Exception:
        return 0.0
    tail_lines = lines[inv_idx + 1 :] if inv_idx + 1 < len(lines) else []
    inv_content = (inv_header_part + ("\n" + "\n".join(tail_lines) if tail_lines else "")).strip()

    # Normalize inv_content: consider empty only if it contains no non-whitespace characters
    inv_has_nonempty = any((ln.strip() != "" for ln in inv_content.splitlines()))

    # Conservative judgment: only score when observation matches expected exactly
    if obs_text != expected_observation:
        return 0.0

    # If observation matches but inventory diff has content -> small/strong penalty (conservative)
    if inv_has_nonempty:
        return -0.8

    # Observation matches and inventory empty: reward must match expected (tiny tolerance)
    if math.isclose(predicted_reward, expected_reward, rel_tol=1e-9, abs_tol=1e-9):
        return 1.0
    else:
        return -1.0

# Rule 26
# Task group: find the
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    import re

    # If action empty, try to extract from state
    if not action or action.strip() == "":
        m = re.search(r'current_step_action:\s*(.*)', state)
        if not m:
            return 0.0
        action = m.group(1).strip()
    else:
        action = action.strip()

    # Only handle the two exact actions
    if action not in ("go to greenhouse", "go to hallway"):
        return 0.0

    # Parse the choice into the three required fields using DOTALL
    m = re.search(
        r'predicted_observation:\s*(.*)\n\s*predicted_reward:\s*([^\n]+)\n\s*predicted_inventory_diff:\s*(.*)\Z',
        choice,
        re.DOTALL
    )
    if not m:
        return 0.0

    predicted_observation = m.group(1).strip()
    predicted_reward_str = m.group(2).strip()
    predicted_inventory_diff = m.group(3).strip()

    # Parse reward as float
    try:
        predicted_reward = float(predicted_reward_str)
    except Exception:
        return 0.0

    # Expected values per action
    if action == "go to greenhouse":
        expected_observation = "You move to the greenhouse."
        expected_reward = 0.0
        expected_inventory_diff = ""  # must be empty
    else:  # action == "go to hallway"
        expected_observation = "You move to the hallway."
        expected_reward = 0.25
        expected_inventory_diff = ""  # accept empty

    # Apply rule only when observation and inventory diff exactly match expected
    if predicted_observation == expected_observation and predicted_inventory_diff == expected_inventory_diff:
        # correct reward => full score
        if abs(predicted_reward - expected_reward) < 1e-8:
            return 1.0
        # observation & inventory correct but reward wrong => strong penalty
        else:
            return -1.0

    # Otherwise do not apply this rule
    return 0.0

# Rule 27
# Task group: focus on
def rule_reward(state, action, choice):
    """
    Returns a float in [-1, 1] according to the consolidated focus-on rules.
    Triggers only for exact actions listed in expected_map.
    """
    import math

    # Map expected values for each exact action
    expected_map = {
        "focus on adult moth in outside": {
            "expected_obs": "You focus on the adult butterfly.",
            "expected_reward": 0.05,
            "require_inv_key": False,
            "inv_must_be_empty": False,
            "special_forbidden_substring": "moth"  # forbid explicit 'moth' mention
        },
        "focus on adult frog in outside": {
            "expected_obs": "You focus on the adult frog.",
            "expected_reward": 0.08,
            "require_inv_key": False,
            "inv_must_be_empty": False,
            "special_forbidden_substring": None
        },
        "focus on cherry tree in the reproducing stage in self watering flower pot 4": {
            "expected_obs": "You focus on the cherry tree.",
            "expected_reward": 0.09,
            "require_inv_key": True,
            "inv_must_be_empty": True,
            "special_forbidden_substring": None
        },
        "focus on grapefruit tree in the reproducing stage in self watering flower pot 5": {
            "expected_obs": "You focus on the grapefruit tree.",
            "expected_reward": 0.09,
            "require_inv_key": True,
            "inv_must_be_empty": True,
            "special_forbidden_substring": None
        },
    }

    # Helper to extract current action if not provided
    def extract_action_from_state(state_text):
        if not state_text:
            return None
        for line in state_text.splitlines():
            line_strip = line.strip()
            if line_strip.startswith("current_step_action:"):
                # take content after first colon
                parts = line_strip.split("current_step_action:", 1)
                if len(parts) > 1:
                    return parts[1].strip()
        return None

    # Determine action string to use
    act = (action or "").strip()
    if not act:
        act = extract_action_from_state(state)
    if not act or act not in expected_map:
        return 0.0

    params = expected_map[act]
    obs_key = "predicted_observation:"
    rew_key = "predicted_reward:"
    inv_key = "predicted_inventory_diff:"

    # Basic presence checks
    i_obs = choice.find(obs_key)
    i_rew = choice.find(rew_key)
    i_inv = choice.find(inv_key) if inv_key in choice else -1

    # For all cases we require predicted_observation and predicted_reward present
    if i_obs == -1 or i_rew == -1:
        return 0.0

    # Determine substrings robustly: observation is between obs_key and rew_key
    obs_start = i_obs + len(obs_key)
    obs_end = i_rew
    predicted_observation = choice[obs_start:obs_end].strip()

    # reward is between rew_key and inv_key if inv present, else to end
    rew_start = i_rew + len(rew_key)
    rew_end = i_inv if i_inv != -1 else len(choice)
    predicted_reward_str = choice[rew_start:rew_end].strip()
    # inventory string is after inv_key if present
    predicted_inventory = ""
    if i_inv != -1:
        inv_start = i_inv + len(inv_key)
        predicted_inventory = choice[inv_start:].strip()

    # If the rule requires inventory key to be present, enforce its presence
    if params["require_inv_key"] and i_inv == -1:
        return 0.0

    # Parse reward number
    try:
        predicted_reward = float(predicted_reward_str)
    except Exception:
        return 0.0

    # Check special forbidden substring (e.g., explicit 'moth' when butterfly expected)
    forbidden = params.get("special_forbidden_substring")
    if forbidden:
        if forbidden.lower() in predicted_observation.lower():
            return -1.0

    # If inventory must be empty, any non-whitespace content counts as non-empty
    if params["inv_must_be_empty"]:
        if predicted_inventory.strip() != "":
            return -1.0

    # Compare observation exact match and reward closeness
    expected_obs = params["expected_obs"]
    expected_reward = params["expected_reward"]
    # Use a small absolute tolerance for numeric comparison
    tol = 1e-6

    if predicted_observation == expected_obs and math.isclose(predicted_reward, expected_reward, abs_tol=tol):
        return 1.0
    else:
        # parsed successfully but required field(s) differ -> penalize
        return -1.0

# Rule 28
# Task group: freeze
def rule_reward(state, action, choice):
    """
    Returns a float in [-1, 1].
    - Triggers only when current step action is exactly "wait".
    - Expects choice to contain lines with headers:
        predicted_observation: ...
        predicted_reward: <float>
        predicted_inventory_diff: ...
      (headers must appear in that order; inventory diff may span following lines)
    - Conservative behavior: only scores when observation exactly matches the expected
      continuation. Does not penalize unrelated actions or continuations.
    """
    import math

    # Helper to extract action from state if action arg is empty/None
    def extract_action_from_state(s):
        for line in s.splitlines():
            if line.strip().startswith("current_step_action:"):
                return line.split("current_step_action:", 1)[1].strip()
        return None

    try:
        act = action.strip() if (action is not None and action.strip() != "") else None
    except Exception:
        act = None
    if not act:
        act = extract_action_from_state(state or "")

    # Only apply for exact action "wait"
    if act != "wait":
        return 0.0

    # Parse the choice into lines and find the three headers in order
    lines = (choice or "").splitlines()
    try:
        i_obs = next(i for i, L in enumerate(lines) if L.startswith("predicted_observation:"))
        i_rew = next(i for i, L in enumerate(lines) if L.startswith("predicted_reward:"))
        i_inv = next(i for i, L in enumerate(lines) if L.startswith("predicted_inventory_diff:"))
        if not (i_obs < i_rew < i_inv):
            return 0.0
        obs = lines[i_obs].split("predicted_observation:", 1)[1].strip()
        rew_str = lines[i_rew].split("predicted_reward:", 1)[1].strip()
        # Inventory diff: take remainder of its line plus any subsequent lines
        inv_first = lines[i_inv].split("predicted_inventory_diff:", 1)[1]
        inv_lines = [inv_first.rstrip("\n")]
        for j in range(i_inv + 1, len(lines)):
            inv_lines.append(lines[j].rstrip("\n"))
        inv_text = "\n".join(inv_lines).strip()
    except StopIteration:
        return 0.0
    except Exception:
        return 0.0

    # Parse reward as float
    try:
        predicted_reward = float(rew_str)
    except Exception:
        return 0.0

    expected_obs = "You decide to wait for 10 iterations."
    expected_reward = 0.18
    # tolerance for floating point comparison
    if math.isclose(predicted_reward, expected_reward, rel_tol=1e-6, abs_tol=1e-6):
        reward_matches = True
    else:
        reward_matches = False

    # Conservative scoring:
    # - Only score when observation exactly matches expected_obs.
    # - If obs matches AND reward matches -> full positive score (do not penalize inventory diffs here).
    # - If obs matches but reward differs -> moderate negative penalty.
    if obs == expected_obs:
        if reward_matches:
            return 1.0
        else:
            return -0.5

    # Do not apply this rule to other observations (avoid false positives)
    return 0.0

# Rule 29
def rule_reward(state, action, choice):
    """
    Returns -1.0, 0.0, or 1.0 according to the refined rule described above.
    - Only applies for actions "go to hallway" and "go to kitchen".
    - Parses predicted_reward (must be float) and predicted_observation (if present) from choice.
    - Infers expected reward (and optionally observation) from the most recent prior
      occurrence of the same action in state that has a following reward line.
    - If no such prior example with a parseable reward exists, abstain (0.0).
    - If expected reward exists:
        - For hallway: if expected observation exists, require predicted_observation to match it;
          otherwise proceed to reward comparison. Return +1.0 for matching reward, -1.0 for mismatch.
        - For kitchen: compare rewards; +1.0 match, -1.0 mismatch.
    - If predicted_reward cannot be parsed, abstain (0.0).
    """
    # helper to extract current_step_action from state if action not provided
    def extract_current_action_from_state(s):
        for line in s.splitlines():
            ln = line.strip()
            if ln.startswith("current_step_action:"):
                return ln.split("current_step_action:", 1)[1].strip()
        return ""

    # Determine action to check: prefer explicit action if non-empty, else extract from state
    act = ""
    if action is not None and isinstance(action, str):
        act = action.strip()
    if not act:
        act = extract_current_action_from_state(state or "")

    # Only apply rule for these two exact actions
    if act not in ("go to hallway", "go to kitchen"):
        return 0.0

    # Parse choice lines for predicted_observation and predicted_reward
    pred_obs = None
    pred_reward = None
    for line in (choice or "").splitlines():
        ln = line.strip()
        if ln.startswith("predicted_observation:"):
            pred_obs = ln.split("predicted_observation:", 1)[1].strip()
        elif ln.startswith("predicted_reward:"):
            val = ln.split("predicted_reward:", 1)[1].strip()
            try:
                pred_reward = float(val)
            except Exception:
                # cannot parse reward -> abstain
                return 0.0

    # Need a parseable predicted_reward to make any judgment
    if pred_reward is None:
        return 0.0

    # Helper to find the most recent prior executed occurrence of the same action
    # and extract a following reward and observation if present.
    def find_most_recent_prior_reward_and_obs(state_text, target_action, lookahead_lines=6):
        lines = state_text.splitlines()
        candidates = []
        for idx, raw in enumerate(lines):
            ln = raw.strip()
            if ln.startswith("action:"):
                # get action text after "action:"
                a = ln.split("action:", 1)[1].strip()
                if a == target_action:
                    # look forward a few lines for reward and observation
                    found_reward = None
                    found_obs = None
                    for j in range(idx + 1, min(len(lines), idx + 1 + lookahead_lines)):
                        s = lines[j].strip()
                        if s.startswith("reward:") and found_reward is None:
                            try:
                                found_reward = float(s.split("reward:", 1)[1].strip())
                            except Exception:
                                # skip unparseable reward for this candidate
                                found_reward = None
                                # do not break; continue searching next lines for a parseable reward
                        elif s.startswith("observation:") and found_obs is None:
                            found_obs = s.split("observation:", 1)[1].strip()
                    # Only consider candidates that have a parseable reward
                    if found_reward is not None:
                        candidates.append((idx, found_reward, found_obs))
        if not candidates:
            return None, None
        # take the most recent one (largest index)
        _, reward_val, obs_val = max(candidates, key=lambda t: t[0])
        return reward_val, obs_val

    expected_reward, expected_observation = find_most_recent_prior_reward_and_obs(state or "", act)

    # If we cannot infer an expected reward from the state, abstain (be conservative)
    if expected_reward is None:
        return 0.0

    # Tolerance for float comparisons
    tol = 1e-6

    # For hallway, if an expected observation exists in the prior example, require it to match
    if act == "go to hallway":
        if expected_observation is not None:
            # predicted_observation must be present and match the expected one
            if pred_obs is None:
                return 0.0
            if pred_obs != expected_observation:
                return -1.0
            # observation matches, now check reward
            return 1.0 if abs(pred_reward - expected_reward) < tol else -1.0
        else:
            # No expected observation available; just compare rewards
            return 1.0 if abs(pred_reward - expected_reward) < tol else -1.0

    else:  # act == "go to kitchen"
        # For kitchen, compare predicted reward to expected reward inferred from history
        return 1.0 if abs(pred_reward - expected_reward) < tol else -1.0

# Rule 30
# Task group: grow a
def rule_reward(state, action, choice):
    """
    Returns a float in [-1, 1] according to the merged conservative rule.
    - Triggers only for actions "pour jug into flower pot 1" or "pour jug into flower pot 3".
    - Expects choice to contain labeled fields:
        predicted_observation: <text>
        predicted_reward: <number>
        predicted_inventory_diff: <maybe lines of +/- or empty>
      Inventory is considered empty only if there are no non-blank lines after the inventory header.
    - Requires predicted_observation to exactly match the expected text for the pot.
    - If observation matches and inventory is empty:
        pot 1: reward ≈ 0.12 -> +1.0
        pot 3: reward ≈ 0.14 -> +1.0
        pot 3: reward ≈ 0.12 -> -1.0  (known conflicting value)
      Otherwise -> 0.0
    """
    import re
    import math

    # small tolerance for float equality
    TOL = 1e-8

    # helper: extract current_step_action from state if action not provided or blank
    def extract_action_from_state(s):
        if not s:
            return ""
        # prefer the last occurrence (more likely current)
        act = ""
        for line in s.splitlines():
            if line.strip().startswith("current_step_action:"):
                act = line.split("current_step_action:", 1)[1].strip()
        return act

    act = (action or "").strip()
    if not act:
        act = extract_action_from_state(state)

    # Only apply for these exact action strings
    if act not in ("pour jug into flower pot 1", "pour jug into flower pot 3"):
        return 0.0

    # Expected observation text depends on pot number in action
    if act.endswith("flower pot 1"):
        expected_obs = "You pour the contents of the jug into the flower pot 1."
        expected_reward_pos = 0.12
        # no negative-known value listed for pot1 in the merged rule
        known_negative = None
    else:
        expected_obs = "You pour the contents of the jug into the flower pot 3."
        expected_reward_pos = 0.14
        # per-source information, 0.12 is a known conflicting value for pot 3
        known_negative = 0.12

    # Parse the choice text for labeled fields
    if choice is None:
        return 0.0
    lines = choice.splitlines()

    obs_text = None
    reward_text = None
    inv_index = None
    # Find indices/lines that start with the exact prefixes
    for i, ln in enumerate(lines):
        if ln.startswith("predicted_observation:"):
            # take remainder of the same line as the observation
            obs_text = ln.split("predicted_observation:", 1)[1].strip()
        elif ln.startswith("predicted_reward:"):
            reward_text = ln.split("predicted_reward:", 1)[1].strip()
        elif ln.startswith("predicted_inventory_diff:"):
            inv_index = i

    # Required fields must be present
    if obs_text is None or reward_text is None or inv_index is None:
        return 0.0

    # Determine whether inventory diff is empty: no non-blank lines after the header,
    # and no same-line content after the header.
    inv_same_line = lines[inv_index].split("predicted_inventory_diff:", 1)[1]
    inv_after = []
    if inv_same_line is not None and inv_same_line.strip() != "":
        inv_after.append(inv_same_line.strip())
    for j in range(inv_index + 1, len(lines)):
        if lines[j].strip() != "":
            inv_after.append(lines[j].strip())
    inv_empty = (len(inv_after) == 0)

    # Require exact observation match and empty inventory diff; otherwise abstain
    if obs_text != expected_obs or not inv_empty:
        return 0.0

    # Parse reward as float
    try:
        pred_reward = float(reward_text)
    except Exception:
        return 0.0

    # Scoring logic
    if math.isclose(pred_reward, expected_reward_pos, rel_tol=0.0, abs_tol=TOL):
        return 1.0
    if known_negative is not None and math.isclose(pred_reward, known_negative, rel_tol=0.0, abs_tol=TOL):
        return -1.0

    # Conservative default: do not score other numeric values to avoid false positives
    return 0.0

# Rule 31
# Task group: measure the
def rule_reward(state, action, choice):
    """
    Returns a float in [-1, 1] according to the merged conservative rule.
    Triggers only for exact actions described in the rule. If parsing fails
    or the action does not match exactly, returns 0.0 (do not apply).
    """
    try:
        # Extract action from parameters or from state if missing/empty
        act = action.strip() if action is not None and action.strip() != "" else None
        if not act:
            for line in state.splitlines():
                if line.strip().startswith("current_step_action:"):
                    act = line.split("current_step_action:", 1)[1].strip()
                    break
        if not act:
            return 0.0

        # Set of move-target actions we handle
        move_prefix = "move unknown substance B in inventory to "
        focus_action = "focus on green box"
        move_colors = {"green", "yellow", "blue", "purple"}

        # Only apply for the exact actions we support
        is_focus = (act == focus_action)
        is_move = False
        target_color = None
        if act.startswith(move_prefix):
            # Expect exact form: move unknown substance B in inventory to <color> box
            suffix = act[len(move_prefix):]
            # suffix should be like "<color> box"
            if suffix.endswith(" box"):
                color = suffix[:-len(" box")]
                if color in move_colors:
                    is_move = True
                    target_color = color

        if not (is_focus or is_move):
            return 0.0

        # Parse the choice into three parts: predicted_observation:, predicted_reward:, predicted_inventory_diff:
        lines = choice.splitlines()
        obs_idx = None
        rew_idx = None
        inv_idx = None
        for i, ln in enumerate(lines):
            if ln.startswith("predicted_observation:") and obs_idx is None:
                obs_idx = i
            elif ln.startswith("predicted_reward:") and rew_idx is None:
                rew_idx = i
            elif ln.startswith("predicted_inventory_diff:") and inv_idx is None:
                inv_idx = i

        # Require all three headers to be present
        if obs_idx is None or rew_idx is None or inv_idx is None:
            return 0.0

        # Extract observation text (text after the header on that line)
        predicted_observation = lines[obs_idx].split("predicted_observation:", 1)[1].strip()

        # Extract reward text and parse float
        rew_text = lines[rew_idx].split("predicted_reward:", 1)[1].strip()
        try:
            predicted_reward = float(rew_text)
        except Exception:
            return 0.0

        # Extract inventory diff: inline text after header plus any subsequent lines until end
        inv_header_tail = lines[inv_idx].split("predicted_inventory_diff:", 1)[1].strip()
        inv_lines = []
        if inv_header_tail != "":
            inv_lines.append(inv_header_tail)
        for ln in lines[inv_idx+1:]:
            # stop if a new top-level key appears (defensive)
            if ln.startswith("predicted_observation:") or ln.startswith("predicted_reward:") or ln.startswith("predicted_inventory_diff:"):
                break
            if ln.strip() == "":
                continue
            inv_lines.append(ln.rstrip())

        # Normalize inventory lines (strip only leading/trailing whitespace)
        inv_lines = [ln.strip() for ln in inv_lines]

        # Now apply the action-specific expectations
        if is_focus:
            expected_obs = "You focus on the green box."
            expected_reward = 0.21
            # Only apply when observation exactly matches
            if predicted_observation != expected_obs:
                return 0.0
            # Inventory diff must be empty (no lines or only whitespace)
            if any(ln.strip() != "" for ln in inv_lines):
                return 0.0
            # Reward check
            if abs(predicted_reward - expected_reward) < 1e-9:
                return 1.0
            else:
                return -1.0

        if is_move:
            expected_obs = f"You move the unknown substance B to the {target_color} box."
            expected_reward = 0.08
            # Only apply when observation exactly matches
            if predicted_observation != expected_obs:
                return 0.0
            # Inventory diff must include the removal line "- unknown substance B"
            has_removal = any(ln.strip() == "- unknown substance B" for ln in inv_lines)
            if not has_removal:
                # Observation matched but removal missing -> penalize
                return -1.0
            # If observation and inventory removal match, require correct reward
            if abs(predicted_reward - expected_reward) < 1e-9:
                return 1.0
            else:
                return -1.0

        # Should not reach here; be conservative
        return 0.0

    except Exception:
        return 0.0

# Rule 32
# Task group: melt
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    import math

    def extract_action_from_state(s):
        for line in s.splitlines():
            if line.strip().startswith("current_step_action:"):
                return line.split("current_step_action:",1)[1].strip()
        return None

    def parse_choice(choice_text):
        # Expect three labeled sections. Return (obs_str, reward_float, inv_lines_list)
        if choice_text is None:
            return None
        key_obs = "predicted_observation:"
        key_rew = "predicted_reward:"
        key_inv = "predicted_inventory_diff:"
        i_obs = choice_text.find(key_obs)
        i_rew = choice_text.find(key_rew)
        i_inv = choice_text.find(key_inv)
        if i_obs == -1 or i_rew == -1 or i_inv == -1:
            return None
        # Ensure order obs -> reward -> inv
        if not (i_obs < i_rew < i_inv):
            return None
        obs_text = choice_text[i_obs + len(key_obs):i_rew].strip()
        rew_text = choice_text[i_rew + len(key_rew):i_inv].strip()
        inv_text = choice_text[i_inv + len(key_inv):].strip()
        # parse reward
        try:
            rew_val = float(rew_text)
        except Exception:
            return None
        # Normalize inventory diff lines: collect non-empty lines, prefer lines starting with +/-
        inv_lines = []
        for ln in inv_text.splitlines():
            ln = ln.rstrip()
            if ln.strip() == "":
                continue
            inv_lines.append(ln.strip())
        return (obs_text, rew_val, inv_lines)

    # Determine action string
    act = action.strip() if action and isinstance(action, str) and action.strip() != "" else None
    if not act:
        act = extract_action_from_state(state if state is not None else "")

    # Only apply to the specified exact actions
    if act not in {
        "open door to outside",
        "use thermometer in inventory on soap in metal pot",
        "use thermometer in inventory on soap",
        "use thermometer in inventory on rubber",
        "look around"
    }:
        return 0.0

    # Parse choice
    parsed = parse_choice(choice)
    if parsed is None:
        # conservative: don't score if we cannot reliably parse
        return 0.0
    pred_obs, pred_rew, pred_inv = parsed

    # Small tolerance for exact reward comparisons
    eps = 1e-9

    # Case: open door to outside
    if act == "open door to outside":
        expected_obs = "The door is now open."
        expected_rew = 0.28
        expected_plus = "+ a metal pot (containing liquid ice cream)"
        expected_minus = "- a metal pot (containing ice cream)"
        obs_ok = (pred_obs == expected_obs)
        reward_ok = (abs(pred_rew - expected_rew) <= eps)
        inv_set = set(pred_inv)
        plus_ok = expected_plus in inv_set
        minus_ok = expected_minus in inv_set
        if obs_ok and reward_ok and plus_ok and minus_ok:
            return 1.0
        else:
            # parsed but deviates from the single correct continuation -> strong penalty
            return -1.0

    # Case: use thermometer in inventory on soap in metal pot
    if act == "use thermometer in inventory on soap in metal pot":
        expected_obs = "the thermometer measures a temperature of 138 degrees celsius"
        expected_rew = 0.25
        obs_match = (pred_obs == expected_obs)
        rew_match = (abs(pred_rew - expected_rew) <= eps)
        if obs_match and rew_match:
            return 1.0
        if obs_match != rew_match:
            return -0.5
        return -1.0

    # Case: use thermometer in inventory on soap
    if act == "use thermometer in inventory on soap":
        expected_obs = "the thermometer measures a temperature of 127 degrees celsius"
        expected_rew = 0.23
        if pred_obs == expected_obs:
            # inventory should be empty for this expected continuation
            if len(pred_inv) > 0:
                # unexpected inventory changes: moderate penalty
                return -0.6
            # linear scaling of score by reward proximity:
            diff = abs(pred_rew - expected_rew)
            # diff 0 -> 1.0 ; diff >= 0.1 -> 0.0 ; scale linearly in between
            if diff >= 0.1:
                score = 0.0
            else:
                score = 1.0 - (diff / 0.1)
            # ensure in [-1,1]
            return max(-1.0, min(1.0, score))
        else:
            # wrong observation is considered a substantial negative (conservative)
            return -0.8

    # Case: use thermometer in inventory on rubber
    if act == "use thermometer in inventory on rubber":
        expected_obs = "the thermometer measures a temperature of 179 degrees celsius"
        expected_rew = 0.25
        obs_ok = (pred_obs == expected_obs)
        rew_ok = (abs(pred_rew - expected_rew) <= eps)
        inv_empty = (len(pred_inv) == 0)
        if obs_ok and rew_ok and inv_empty:
            return 1.0
        else:
            # action matches but deviation from expected -> conservative penalty
            return -0.5

    # Case: look around
    if act == "look around":
        # required substrings in observation for correct greenhouse description
        required_subs = [
            "This room is called the greenhouse.",
            "a substance called air",
            "a bee hive. The bee hive door is closed.",
            "a jug (containing nothing)",
            "a sink, which is turned off. In the sink is: nothing.",
            "A door to the hallway (that is closed)",
            "A door to the outside (that is open)"
        ]
        obs_ok = all(sub in pred_obs for sub in required_subs)
        reward_ok = abs(pred_rew - 0.0) <= eps
        inv_ok = (len(pred_inv) == 0)
        if obs_ok and reward_ok and inv_ok:
            return 1.0
        else:
            return -1.0

    # Fallback (shouldn't reach here because we guarded actions above)
    return 0.0

# Rule 33
# Task group: turn on
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    try:
        # Normalize/obtain current action string
        act = action.strip() if action and action.strip() != "" else None
        if not act:
            # try to extract from state: look for line starting with 'current_step_action:'
            for line in state.splitlines():
                if line.strip().startswith("current_step_action:"):
                    act = line.split("current_step_action:", 1)[1].strip()
                    break
        if not act:
            return 0.0

        # Supported target actions and their expected observations/rewards
        mapping = {
            "connect red wire terminal 2 to anode in blue light bulb": (
                "terminal 2 on red wire is now connected to anode on blue light bulb",
                0.23
            ),
            "connect black wire terminal 2 to anode in blue light bulb": (
                "terminal 2 on black wire is now connected to anode on blue light bulb",
                0.47
            )
        }

        if act not in mapping:
            return 0.0

        expected_observation, expected_reward = mapping[act]

        # Parse the choice into the three required fields
        lines = choice.splitlines()
        po_prefix = "predicted_observation:"
        pr_prefix = "predicted_reward:"
        pid_prefix = "predicted_inventory_diff:"

        po_idx = pr_idx = pid_idx = None
        for i, ln in enumerate(lines):
            if ln.startswith(po_prefix) and po_idx is None:
                po_idx = i
            elif ln.startswith(pr_prefix) and pr_idx is None:
                pr_idx = i
            elif ln.startswith(pid_prefix) and pid_idx is None:
                pid_idx = i

        # All three headers must be present
        if po_idx is None or pr_idx is None or pid_idx is None:
            return 0.0

        # Extract predicted_observation (remainder of the po line)
        predicted_observation = lines[po_idx][len(po_prefix):].lstrip()

        # Extract predicted_reward and parse float
        pr_text = lines[pr_idx][len(pr_prefix):].strip()
        try:
            predicted_reward = float(pr_text)
        except:
            return 0.0

        # Inventory diff are lines after the pid_idx; require inventory to be empty to apply rule
        inv_lines = []
        for ln in lines[pid_idx + 1:]:
            if ln.strip() != "":
                inv_lines.append(ln.rstrip())
        inventory_empty = (len(inv_lines) == 0)

        if not inventory_empty:
            # Do not apply this rule when the model reports inventory changes (avoid false positives)
            return 0.0

        # If the observation exactly matches the expected observation for this action
        if predicted_observation == expected_observation:
            # correct reward -> strong positive
            if abs(predicted_reward - expected_reward) < 1e-9:
                return 1.0
            # observation correct but reward wrong -> penalize
            return -1.0

        # Otherwise, be conservative and do not apply the rule
        return 0.0

    except Exception:
        return 0.0

# Rule 34
# Task group: turn on
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    expected_actions = {
        "connect battery anode to orange wire terminal 1":
            "anode on battery is now connected to terminal 1 on orange wire",
        "connect battery cathode to black wire terminal 1":
            "cathode on battery is now connected to terminal 1 on black wire"
    }

    # If action empty, try to extract from state
    act = (action or "").strip()
    if not act:
        for line in (state or "").splitlines():
            if line.strip().startswith("current_step_action:"):
                act = line.split("current_step_action:", 1)[1].strip()
                break

    if act not in expected_actions:
        return 0.0

    # Parse the choice for labeled fields (allow leading whitespace)
    lines = (choice or "").splitlines()
    obs = None
    rew_str = None
    inv_index = None
    for i, ln in enumerate(lines):
        s = ln.lstrip()
        if s.startswith("predicted_observation:"):
            obs = ln.split("predicted_observation:", 1)[1].strip()
        elif s.startswith("predicted_reward:"):
            rew_str = ln.split("predicted_reward:", 1)[1].strip()
        elif s.startswith("predicted_inventory_diff:"):
            # inventory diff may continue on this and subsequent lines
            inv_first = ln.split("predicted_inventory_diff:", 1)[1].strip()
            inv_rest = "\n".join(l.rstrip() for l in lines[i+1:]) if i+1 < len(lines) else ""
            inventory_text = (inv_first + ("\n" + inv_rest if inv_rest else "")).strip()
            inv_index = i
            break

    # Require all three labeled fields to be present and reward parseable
    if obs is None or rew_str is None or inv_index is None:
        return 0.0
    try:
        rew = float(rew_str)
    except:
        return 0.0

    # Determine if inventory diff is empty (no meaningful content)
    inv_empty = (inventory_text.strip() == "")

    expected_obs = expected_actions[act]

    # Outcomes:
    # exact correct observation, empty inventory, zero reward => 1.0
    if obs == expected_obs and inv_empty and abs(rew - 0.0) < 1e-9:
        return 1.0
    # observation correct and inventory empty but nonzero reward => moderate penalty
    if obs == expected_obs and inv_empty:
        return -0.4
    # parsed but observation or inventory incorrect => stronger penalty
    return -0.6

# Rule 35
# Task group: use chemistry
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    import math

    TOL = 1e-6

    def extract_current_action_from_state(state_text):
        if not state_text:
            return ""
        for line in state_text.splitlines()[::-1]:
            line = line.strip()
            if line.startswith("current_step_action:"):
                return line.split("current_step_action:", 1)[1].strip()
        return ""

    # Determine action: prefer provided 'action' param; if empty, extract from state.
    act = action.strip() if action is not None else ""
    if not act:
        act = extract_current_action_from_state(state)

    if act not in ("open cupboard", "open door to outside"):
        return 0.0

    # Parse the choice text into three headers' contents
    try:
        lines = choice.splitlines() if choice is not None else []
        obs_text = None
        rew_text = None
        inv_lines = None

        # Locate headers anywhere in the block
        obs_idx = rew_idx = inv_idx = None
        for i, ln in enumerate(lines):
            s = ln.lstrip()
            if s.startswith("predicted_observation:") and obs_idx is None:
                obs_idx = i
                # capture same-line content after colon
                obs_text = ln.split("predicted_observation:", 1)[1].lstrip()
            elif s.startswith("predicted_reward:") and rew_idx is None:
                rew_idx = i
                rew_text = ln.split("predicted_reward:", 1)[1].strip()
            elif s.startswith("predicted_inventory_diff:") and inv_idx is None:
                inv_idx = i
                # capture same-line content after colon as first inventory line (if any)
                inv_first = ln.split("predicted_inventory_diff:", 1)[1].lstrip()
                inv_lines = []
                if inv_first != "":
                    inv_lines.append(inv_first)
                # collect any following lines as inventory diff
                for j in range(i+1, len(lines)):
                    inv_lines.append(lines[j])
                break  # inventory diff is treated as the final section
        # Ensure reward and inventory header present (observation may be optional for door)
        if rew_idx is None or inv_idx is None:
            return 0.0

        # Normalize parsed values
        predicted_observation = obs_text.strip() if obs_text is not None else ""
        try:
            predicted_reward = float(rew_text)
        except Exception:
            return 0.0
        predicted_inventory_lines = [ln.rstrip() for ln in (inv_lines or [])]

        # Determine if inventory diff is empty: no non-empty lines and no + or - prefixed lines
        has_inventory_change = False
        for ln in predicted_inventory_lines:
            if ln is None:
                continue
            s = ln.strip()
            if s == "":
                continue
            # any non-empty line counts as a change; be conservative
            has_inventory_change = True
            break

    except Exception:
        return 0.0

    # Now apply action-specific conditions
    if act == "open cupboard":
        # require exact observation and no inventory changes
        if predicted_observation != "The cupboard is now open.":
            return 0.0
        if has_inventory_change:
            return 0.0
    else:  # act == "open door to outside"
        # require no inventory changes; do not enforce observation text
        if has_inventory_change:
            return 0.0

    # If we reach here, the observation/inventory conditions are satisfied -> judge reward
    if math.isclose(predicted_reward, 0.0, abs_tol=TOL):
        return 1.0
    else:
        return -1.0

# Rule 36
# Task group: use chemistry
def rule_reward(state, action, choice):
    """
    Returns a float in [-1, 1] according to the rule description above.
    Conservative behavior: if parsing fails or action doesn't exactly match,
    return 0.0 to avoid false positives.
    """
    try:
        expected_action = "focus on fruit salad"
        expected_observation = "You focus on the fruit salad."
        expected_reward = 0.50
        EPS = 1e-6

        # Determine action: prefer explicit argument, otherwise try to extract from state
        act = None
        if action is not None:
            a = action.strip()
            if a != "":
                act = a
        if not act:
            # search for the last occurrence of 'current_step_action:' in state
            try:
                for line in reversed(state.splitlines()):
                    if line.strip().startswith("current_step_action:"):
                        act = line.split("current_step_action:", 1)[1].strip()
                        break
            except Exception:
                return 0.0

        if act != expected_action:
            return 0.0

        # Parse the choice block for the three expected fields.
        lines = choice.splitlines()
        po_idx = pr_idx = pid_idx = None
        for i, ln in enumerate(lines):
            if ln.startswith("predicted_observation:"):
                po_idx = i
            elif ln.startswith("predicted_reward:"):
                pr_idx = i
            elif ln.startswith("predicted_inventory_diff:"):
                pid_idx = i

        # require all three sections to be present for a definite judgment
        if po_idx is None or pr_idx is None or pid_idx is None:
            return 0.0

        # Extract predicted_observation (text after the first colon on that line)
        try:
            predicted_observation = lines[po_idx].split("predicted_observation:", 1)[1].strip()
        except Exception:
            return 0.0

        # Extract predicted_reward as float
        try:
            pr_text = lines[pr_idx].split("predicted_reward:", 1)[1].strip()
            predicted_reward = float(pr_text)
        except Exception:
            return 0.0

        # Collect inventory-diff lines: any non-empty lines after predicted_inventory_diff:
        inv_lines = []
        for ln in lines[pid_idx+1:]:
            if ln.strip() == "":
                continue
            inv_lines.append(ln.rstrip())

        # Now apply checks
        if predicted_observation != expected_observation:
            # Strong penalty for wrong observation
            return -1.0

        # Observation matches; check reward
        if abs(predicted_reward - expected_reward) > EPS:
            # Moderate penalty for wrong reward
            return -0.5

        # Observation and reward match. Determine whether there are explicit inventory changes.
        # To avoid false positives, only treat lines that explicitly start with '+' or '-' (after optional whitespace) as changes.
        explicit_changes = False
        for ln in inv_lines:
            stripped = ln.lstrip()
            if stripped.startswith("+") or stripped.startswith("-"):
                explicit_changes = True
                break

        if explicit_changes:
            # Moderate penalty if unexpected inventory +/- lines are present
            return -0.8

        # Everything as expected: positive score
        return 1.0

    except Exception:
        # Conservative fallback: no judgment on unexpected errors
        return 0.0

# Rule 37
# Task group: use chemistry
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    import re

    # Determine the current action; allow extraction from state if action is empty/None
    act = action.strip() if action is not None else ""
    if act == "":
        m = re.search(r"current_step_action:\s*(.+)", state or "")
        if m:
            act = m.group(1).strip()
        else:
            return 0.0

    # Apply only for the exact target action
    if act != "focus on paint in bowl":
        return 0.0

    # Find labeled lines in the choice text
    try:
        lines = [ln.rstrip("\n") for ln in (choice or "").splitlines()]
        po_idx = pr_idx = pid_idx = None
        for i, ln in enumerate(lines):
            s = ln.lstrip()
            if s.startswith("predicted_observation:"):
                po_idx = i
            elif s.startswith("predicted_reward:"):
                pr_idx = i
            elif s.startswith("predicted_inventory_diff:"):
                pid_idx = i
        # If any required label missing, do not apply this rule
        if po_idx is None or pr_idx is None or pid_idx is None:
            return 0.0

        # Extract observation (text after the colon)
        po_line = lines[po_idx]
        obs = po_line.split("predicted_observation:", 1)[1].lstrip()

        # Extract reward and parse to float
        pr_line = lines[pr_idx]
        reward_str = pr_line.split("predicted_reward:", 1)[1].strip()
        predicted_reward = float(reward_str)

        # Inventory diff lines are any lines after the predicted_inventory_diff: label
        inv_lines = lines[pid_idx+1:] if pid_idx+1 < len(lines) else []
        # Conservative: treat inventory as changed only if any line explicitly starts with '+' or '-'
        inv_nonempty = any(l.lstrip().startswith(("+", "-")) for l in inv_lines)
    except Exception:
        # Parsing problems: do not apply the rule
        return 0.0

    # Expected values
    expected_obs = "You focus on the yellow-orange paint."
    expected_rewards = (0.17, 0.20)
    eps = 1e-6

    # Scoring logic
    if obs == expected_obs and not inv_nonempty:
        # Observation and inventory diff acceptable
        for er in expected_rewards:
            if abs(predicted_reward - er) <= eps:
                return 1.0
        # Reward does not match expected canonical values
        return -0.5
    else:
        # Observation mismatch or explicit inventory changes: stronger penalty
        return -1.0

# Rule 38
def rule_reward(state, action, choice):
    import re
    # Helper to clamp
    def clamp(x, a=-1.0, b=1.0):
        return max(a, min(b, x))

    # If action not provided, extract from state
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''

    # Parse choice fields
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)\n\s*predicted_reward:', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)

    if not (obs_m and rew_m and diff_m is not None):
        return -0.5  # malformed choice

    obs = obs_m.group(1).strip().lower()
    try:
        predicted_reward = float(rew_m.group(1))
    except:
        predicted_reward = 0.0
    inv_diff = diff_m.group(1).strip()

    # Match applicable actions: "focus on <obj>" or "examine <obj>"
    m = re.match(r'(?i)^\s*(focus on|examine)\s+(.+)$', action.strip())
    if not m:
        return 0.0  # rule not applicable

    act_type = m.group(1).strip().lower()  # 'focus on' or 'examine'
    obj = m.group(2).strip().lower().rstrip('.').strip()

    score = 0.0

    # Check that observation mentions the object
    if obj and obj in obs:
        score += 0.5
    else:
        # If object not mentioned, heavy penalty
        score -= 0.6

    # Action-specific observation expectations
    if act_type.startswith('focus'):
        if 'focus' in obs or 'you focus' in obs:
            score += 0.3
        else:
            # mild penalty if focus action doesn't indicate focusing
            score -= 0.15
    else:  # examine
        if 'examin' in obs:  # matches 'examine' or 'examining'
            score += 0.3
        else:
            # it's acceptable for examine to just return the object name; reward a bit if obs equals object or starts with it
            obs_first_line = obs.splitlines()[0].strip()
            if obs_first_line == obj or obs_first_line.startswith(obj):
                score += 0.2
            else:
                score -= 0.1

    # Inventory diff should be empty for examine/focus
    inv_lines = [ln for ln in inv_diff.splitlines() if ln.strip()]
    if len(inv_lines) == 0:
        score += 0.2
    else:
        score -= 0.4

    # Predicted reward should be positive (these are perceptual actions)
    if predicted_reward > 0.0:
        score += 0.2
    else:
        score -= 0.6

    return clamp(score)

# Rule 39
def rule_reward(state, action, choice):
    import re

    # If action not provided, extract from state
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state or '')
        action = m.group(1).strip() if m else ''

    # Only apply this rule to recipe-read actions
    if not re.search(r'(?i)\bread\b.*\brecipe\b|\bread\b.*\binventory\b.*\brecipe\b', action):
        return 0.0

    # Parse predicted_observation, predicted_reward, predicted_inventory_diff
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)\npredicted_reward:', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)

    if not (obs_m and rew_m and diff_m is not None):
        return -0.5

    obs = obs_m.group(1).strip().lower()
    invdiff_text = diff_m.group(1).strip()

    # Find candidate ingredient list in the predicted_observation.
    # Look for phrases like "you need to mix ..." or "mix ..." or "to make ... you need ..."
    ing_text = None
    for patt in [r'(?i)you need to mix\s*(.*?)(?:[\.!\n]|$)',
                 r'(?i)mix\s*(.*?)(?:[\.!\n]|$)',
                 r'(?i)you need\s*(.*?)(?:[\.!\n]|$)',
                 r'(?i)to make [^,\.]*,\s*you need to mix\s*(.*?)(?:[\.!\n]|$)']:
        m = re.search(patt, obs)
        if m:
            ing_text = m.group(1)
            break
    # Fallback: try to find anything after "the recipe reads:" or after colon
    if not ing_text:
        m = re.search(r'(?i)the recipe reads:?(.*)', obs)
        if m:
            ing_text = m.group(1)
    if not ing_text:
        # Could not parse ingredients -> neutral/low penalty
        return -0.5

    # Split ingredient text by commas and 'and', remove parentheses and non-alpha
    # Produce tokens like 'jam', 'bread', 'butter'
    # Keep multi-word tokens (e.g., 'peanut butter') intact
    # Normalize separators
    ing_text_clean = re.sub(r'[\(\)\[\]]', ' ', ing_text)
    parts = re.split(r',|\band\b|\b&\b', ing_text_clean)
    ingredients = []
    for p in parts:
        # remove trailing/leading junk and take contiguous alpha+space tokens
        tok = re.sub(r'[^a-z0-9 \-]', ' ', p.lower()).strip()
        tok = re.sub(r'\s+', ' ', tok)
        if tok:
            ingredients.append(tok.strip())

    if not ingredients:
        return -0.5

    # Build a set of object/name tokens that appear in the state's visible text (rooms + inventory)
    # Extract phrases introduced by "a ", "an ", "the " and also standalone words that look like item names.
    present_objs = set()
    for m in re.finditer(r'\b(?:a|an|the)\s+([a-z0-9][a-z0-9 \-\']{0,80}?)\b', (state or '').lower()):
        name = m.group(1).strip()
        # strip trailing descriptors like '(containing...)'
        name = re.sub(r'\s*\(.*$', '', name).strip()
        present_objs.add(name)
    # Also add lone words that appear as single tokens after commas/periods in state lines (heuristic)
    for m in re.finditer(r'\b([a-z0-9][a-z0-9\-]{0,40})\b', (state or '').lower()):
        present_objs.add(m.group(1))

    # Helper to check if an ingredient token matches any present object (whole-word substring)
    def ingredient_present(ing):
        # check exact presence
        if ing in present_objs:
            return True
        # check any present object contains the ingredient as a whole word
        for obj in present_objs:
            # full-word match
            if re.search(r'\b' + re.escape(ing) + r'\b', obj):
                return True
        # also try splitting ingredient into words and check any word appears
        for word in ing.split():
            for obj in present_objs:
                if re.search(r'\b' + re.escape(word) + r'\b', obj):
                    return True
        return False

    # Check each ingredient
    missing = []
    for ing in ingredients:
        if not ingredient_present(ing):
            missing.append(ing)

    # If any missing ingredients, strong penalty
    if missing:
        return -1.0

    # If all ingredients are present, prefer no inventory changes (reading shouldn't change inventory)
    inv_lines = [ln.strip() for ln in invdiff_text.splitlines() if ln.strip()]
    if len(inv_lines) == 0:
        return 1.0
    else:
        # small penalty if predicted inventory changes during reading
        return 0.6

# Rule 40
def rule_reward(state, action, choice):
    import re
    # Extract action from state if not provided
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''
    # Parse choice into fields
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)\npredicted_reward:', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff:\s*(.*)$', choice)
    if not (obs_m and rew_m and diff_m is not None):
        return -0.5
    obs = obs_m.group(1).strip().lower()
    inv = diff_m.group(1)
    # Check action pattern "focus on <obj>"
    m = re.match(r'(?i)^\s*focus on\s+(.+)$', action.strip())
    if not m:
        return 0.0
    obj = m.group(1).strip().lower()
    # Prepare candidate key tokens from object: take last two meaningful tokens
    tokens = re.findall(r"[a-zA-Z0-9\-]+", obj)
    if len(tokens) >= 2:
        key = ' '.join(tokens[-2:])
    elif tokens:
        key = tokens[-1]
    else:
        key = obj
    # Observation check: must contain "you focus on" and mention the key tokens
    obs_ok = ('you focus on' in obs) and (key in obs)
    # Inventory diff check: should be empty (no '+' or '-' lines with content)
    inv_lines = [ln.strip() for ln in inv.splitlines() if ln.strip()]
    inv_ok = len(inv_lines) == 0
    # Scoring
    if obs_ok and inv_ok:
        return 1.0
    if obs_ok and not inv_ok:
        return 0.5
    # observation incorrect -> negative signal
    return -0.5

# Rule 41
def rule_reward(state, action, choice):
    import re
    # Extract action from state if not provided
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''
    # Parse predicted_observation, predicted_reward, predicted_inventory_diff
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)\s*?\n\s*predicted_reward:', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)
    if not (obs_m and rew_m and diff_m is not None):
        # Can't parse choice properly
        return -0.5
    obs = obs_m.group(1).strip().lower()
    inv_text = diff_m.group(1) or ''
    inv_lines = [ln.strip() for ln in inv_text.splitlines() if ln.strip()]
    # Match the action pattern "move <obj> to <dest>"
    m = re.match(r'(?i)^\s*move\s+(.+?)\s+to\s+(.+?)\s*$', action.strip())
    if not m:
        # Rule not applicable
        return 0.0
    obj = m.group(1).strip().lower()
    dest = m.group(2).strip().lower()
    # Determine whether destination is inventory-like
    dest_is_inventory = bool(re.search(r'\binventory\b|\bto inventory\b|\bin my inventory\b|\binto inventory\b', dest))
    # Observation should mention moving the object to destination
    # Check both object and destination substrings appear in observation and "you move"
    obs_ok = ('you move' in obs) and (obj in obs) and (dest in obs)
    # Detect if predicted_inventory_diff has any "+ " additions (unexpected)
    has_plus = any(ln.startswith('+ ') for ln in inv_lines)
    # Also check if there's a "+ " referencing the object specifically
    plus_refs_obj = any(ln.startswith('+ ') and obj.split()[0] in ln.lower() for ln in inv_lines)
    score = 0.0
    # Major error: model adds a '+ ...' when move is not to inventory
    if has_plus and not dest_is_inventory:
        # Strong penalty for adding inventory when not moving to inventory
        return -0.8
    # If destination is inventory but model did not add a + entry referencing the object, partial penalty
    if dest_is_inventory and not plus_refs_obj:
        score -= 0.4
    # Reward correct observation mention
    if obs_ok:
        score += 0.8
    else:
        score -= 0.3
    # Cap to [-1,1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return float(score)

# Rule 42
def rule_reward(state, action, choice):
    import re, math

    # Helper: clamp to [-1,1]
    def clamp(x):
        return max(-1.0, min(1.0, x))

    # If action not provided, try to extract it from state
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''

    # Parse choice fields
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)\npredicted_reward:', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)

    # If any field missing, give a mild negative signal
    if not (obs_m and rew_m and diff_m is not None):
        return -0.5

    obs = obs_m.group(1).strip().lower()
    try:
        predicted_reward = float(rew_m.group(1))
    except:
        predicted_reward = 0.0
    inv_diff = diff_m.group(1).strip()

    # Only apply this rule for tasks about shortest life span
    if 'shortest life span' not in state.lower():
        return 0.0

    # Match action "focus on <obj>"
    m = re.match(r'(?i)^\s*focus on\s+(.+)$', action.strip())
    if not m:
        return 0.0

    obj = m.group(1).strip().lower()

    score = 0.0

    # Observation should state focusing on the object
    if f'you focus on the {obj}' in obs or f'you focus on {obj}' in obs:
        score += 0.6
    else:
        score -= 0.6

    # Reward should be substantial for this task (expect >= 0.5)
    if predicted_reward >= 0.5:
        score += 0.4
    else:
        score -= 0.8

    # Inventory diff should be empty (focusing shouldn't change inventory)
    # If there are any non-empty lines that look like +/- inventory changes, penalize
    inv_lines = [ln for ln in inv_diff.splitlines() if ln.strip()]
    if len(inv_lines) == 0:
        score += 0.0
    else:
        score -= 0.2

    return clamp(score)

# Rule 43
def rule_reward(state, action, choice):
    import re
    # Helper to safely extract current_step_action from state if action not provided
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''

    # Parse choice fields
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)(?:\npredicted_reward:|\npredicted_inventory_diff:|$)', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    inv_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)

    if not obs_m or inv_m is None:
        return -0.5  # couldn't parse required fields

    obs = obs_m.group(1).strip().lower()
    inv_text = inv_m.group(1).strip()

    # Only apply this rule to "move ... to ..." actions
    m = re.match(r'(?i)\s*move\s+(.+?)\s+to\s+(.+)$', action.strip())
    if not m:
        return 0.0  # not applicable

    obj = m.group(1).strip().lower()
    dest = m.group(2).strip().lower()

    score = 0.0
    # 1) Check the movement phrase appears in predicted_observation
    expected_move_phrase = f'you move the {obj} to the {dest}'
    if expected_move_phrase in obs:
        score += 0.5
    else:
        # Sometimes observation may include parentheses before the phrase, so also allow that form
        # but still require the core phrase present; if missing, strong negative.
        return -0.9

    # 2) Determine if the state shows the object was connected earlier
    state_lower = state.lower()
    connected = False
    # Patterns that indicate the object was connected in the history
    conn_patterns = [
        rf'\b{re.escape(obj)}\b.*\bconnected\b',
        rf'\bconnected\b.*\b{re.escape(obj)}\b',
        rf'connect(?:ed|)\b.*\b{re.escape(obj)}\b',
        rf'terminal .* on {re.escape(obj)} .*connected',
        rf'{re.escape(obj)} .*is now connected',
    ]
    for p in conn_patterns:
        if re.search(p, state_lower):
            connected = True
            break

    # If it was connected, predicted_observation must indicate a disconnection
    if connected:
        if 'disconnect' in obs or '(disconnecting' in obs:
            score += 0.4
        else:
            return -1.0  # strong penalty: moving a connected object without disconnect mention

    # 3) predicted_inventory_diff should be empty (no plus/minus lines)
    inv_lines = [ln.strip() for ln in inv_text.splitlines() if ln.strip()]
    has_inventory_change = any(ln.startswith('+') or ln.startswith('-') for ln in inv_lines)
    if not has_inventory_change:
        score += 0.1
    else:
        # small negative if inventory incorrectly shows change
        score -= 0.6

    # Clamp to [-1, 1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return float(score)

# Rule 44
def rule_reward(state, action, choice):
    import re
    # Helper to normalize text: lower, remove punctuation, remove articles
    def normalize(s):
        s = s.lower()
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # Extract action from state if not provided
    if not action:
        m = re.search(r'(?mi)^current_step_action:\s*(.+)$', state)
        action = m.group(1).strip() if m else ''

    # Parse choice into predicted_observation, predicted_reward, predicted_inventory_diff
    obs_m = re.search(r'(?s)predicted_observation:\s*(.*?)(?:\r?\n)predicted_reward:', choice)
    rew_m = re.search(r'predicted_reward:\s*([-+]?\d*\.?\d+)', choice)
    diff_m = re.search(r'(?s)predicted_inventory_diff\s*:\s*(.*)$', choice)

    if not (obs_m and rew_m and diff_m is not None):
        # malformed choice -> mild negative
        return -0.5

    obs = obs_m.group(1).strip().lower()
    try:
        pred_reward = float(rew_m.group(1))
    except:
        pred_reward = None
    inv_diff = diff_m.group(1).strip()

    # Only apply this rule for 'focus on <obj>' actions
    mact = re.match(r'(?i)^\s*focus on\s+(.+)$', action.strip())
    if not mact:
        return 0.0

    obj = mact.group(1).strip().lower()
    # Normalize both for robust matching
    norm_obj = normalize(obj)
    norm_obs = normalize(obs)

    score = 0.0

    # Check observation: contains 'you focus on' and mentions object words in order
    obs_contains_focus = 'you focus on' in obs
    obs_mentions_obj = False
    if norm_obj:
        # require the normalized object string to appear as a substring in normalized observation
        obs_mentions_obj = norm_obj in norm_obs
    else:
        obs_mentions_obj = True  # no object to check (unlikely)

    if obs_contains_focus and obs_mentions_obj:
        score += 0.5

    # Check predicted reward approximately 0.5
    if pred_reward is not None and abs(pred_reward - 0.5) <= 1e-2:
        score += 0.5
    else:
        # penalize incorrect reward for a focus action
        score -= 0.5

    # Inventory diff should be empty for focus actions
    # Consider non-empty if any non-blank lines exist (e.g., '+' or '-' lines)
    inv_lines = [ln for ln in inv_diff.splitlines() if ln.strip()]
    if inv_lines:
        score -= 0.5

    # Clip to [-1, 1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return score

