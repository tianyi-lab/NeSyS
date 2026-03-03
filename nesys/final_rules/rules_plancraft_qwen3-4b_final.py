# WMQA Improved Rules
# Improved from (2 files):
#   - transition_mcq/rules_plancraft_qwen3-4b_new.py
#   - transition_mcq/smelt_combined_rules_qwen3-4b.py
# Dev unit-weight improvement vs original: +1.71%
# Dev unit-weight accuracy (improved rules): 83.45%
# Dev weighted accuracy (learned on dev): 86.87%
# Test baseline accuracy: 85.10%
# Test weighted accuracy: 87.37%
# Test weighted improvement: +2.28%

# Rule 1
def rule_reward(state, action, choice):
    import re

    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z0-9]+\])\s*to\s*(\[[A-Z0-9]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, 0
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns dict slot -> (name, qty)
        items = {}
        # match lines like "- name [SLOT] quantity N"
        for m in re.finditer(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = m.group(1).strip()
            slot = f'[{m.group(2)}]'
            qty = int(m.group(3))
            items[slot] = (name, qty)
        return items

    src_slot, dst_slot, qty = parse_move_action(action)
    if src_slot is None:
        # Not a move action: this rule doesn't apply
        return 0.0

    state_items = parse_items(state)
    choice_items = parse_items(choice)

    # source item must exist in the original state
    if src_slot not in state_items:
        return -1.0

    moved_name, src_prev_qty = state_items[src_slot]

    # find src and dst in the choice; if absent, treat qty as 0
    src_choice_name_qty = choice_items.get(src_slot, (None, 0))
    dst_choice_name_qty = choice_items.get(dst_slot, (None, 0))
    dst_state_name_qty = state_items.get(dst_slot, (None, 0))

    src_choice_name, src_new_qty = src_choice_name_qty
    dst_choice_name, dst_new_qty = dst_choice_name_qty
    dst_prev_name, dst_prev_qty = dst_state_name_qty

    checks = 0

    # Check 1: destination increased by qty and has the same item name as the moved item
    # Destination may be newly created (dst_prev_qty == 0)
    dst_name_matches = (dst_choice_name == moved_name)
    dst_increased = (dst_new_qty - dst_prev_qty) == qty
    if dst_name_matches and dst_increased:
        checks += 1

    # Check 2: source decreased by qty (or removed if reaches zero) and name consistent
    src_name_ok = (src_choice_name in (moved_name, None))  # allow disappearance (None)
    src_decreased = (src_prev_qty - src_new_qty) == qty
    if src_name_ok and src_decreased:
        checks += 1

    # Check 3: unrelated item totals unchanged, ignoring the moved item and ignoring slot [0]
    def totals(items):
        d = {}
        for slot, (name, count) in items.items():
            if slot == '[0]':
                continue
            if name == moved_name:
                continue
            d[name] = d.get(name, 0) + count
        return d

    tot_state = totals(state_items)
    tot_choice = totals(choice_items)
    if tot_state == tot_choice:
        checks += 1

    # Map checks (0..3) to score in [-1, 1]
    score = (checks / 3.0) * 2.0 - 1.0
    # Ensure bounds
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return score

# Rule 2
def rule_reward(state, action, choice):
    import re

    # Parse move action: expected format "move: from [S] to [D] with quantity q"
    m = re.search(r'move:\s*from\s*(\[[^\]]+\])\s*to\s*(\[[^\]]+\])\s*with\s*quantity\s*(\d+)', action)
    if not m:
        # This rule applies only for move actions; return neutral small negative value
        return -0.5

    src_slot = m.group(1)
    dst_slot = m.group(2)
    qty = int(m.group(3))

    # Parse inventory lines into dict slot -> (name, count)
    def parse_items(inv_text):
        items = {}
        # match lines like "- item name [SLOT] quantity N"
        for name, slot, q in re.findall(r'-\s+([^\[]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', inv_text):
            slot_label = f'[{slot}]'
            items[slot_label] = (name.strip(), int(q))
        return items

    state_items = parse_items(state)
    choice_items = parse_items(choice)

    # Source must exist in original state
    if src_slot not in state_items:
        # cannot validate move if source missing: strong penalty
        return -1.0

    moved_name, src_prev = state_items[src_slot]
    dst_prev = choice_prev_dst = 0
    # get previous destination count in state (0 if absent or different name)
    if dst_slot in state_items and state_items[dst_slot][0] == moved_name:
        dst_prev = state_items[dst_slot][1]
    elif dst_slot in state_items:
        # destination has different item before action; that is allowed only if move will create same-name entry,
        # but changing the name of an unrelated slot is disallowed by the rule -> treat as mismatch later.
        dst_prev = state_items[dst_slot][1]

    # Get new values in choice (dst/src may be absent)
    src_new_name, src_new = choice_items.get(src_slot, (None, 0))
    dst_new_name, dst_new = choice_items.get(dst_slot, (None, 0))

    # Check 1: source decreased by exactly qty and name unchanged or removed
    src_ok = False
    # If src_new_name is None (slot removed), treat count as 0
    if src_new_name is None:
        src_new_count = 0
    else:
        src_new_count = src_new
    if (src_prev - src_new_count) == qty:
        # name at source either remains same or was removed entirely -> acceptable
        if src_new_name in (moved_name, None):
            src_ok = True

    # Check 2: destination increased by exactly qty and item name matches moved item
    dst_ok = False
    # Destination previous name might have been different; we require that the new destination contains moved_name
    if dst_new_name == moved_name and (dst_new - dst_prev) == qty:
        dst_ok = True

    # Check 3: unrelated slots unchanged (ignore src_slot, dst_slot, and slot "[0]")
    unrelated_ok = True
    for slot, (name, count) in state_items.items():
        if slot in (src_slot, dst_slot, '[0]'):
            continue
        # In choice, the same slot must exist with same name and count
        c = choice_items.get(slot)
        if c is None:
            unrelated_ok = False
            break
        c_name, c_count = c
        if c_name != name or c_count != count:
            unrelated_ok = False
            break
    # Also ensure the choice does not introduce new unrelated slots (except dst_slot or '[0]')
    for slot in choice_items:
        if slot in (src_slot, dst_slot, '[0]'):
            continue
        if slot not in state_items:
            # new slot introduced that's not allowed
            unrelated_ok = False
            break

    # Compose score: full positive if all checks pass; partial reward if 2/3 pass; penalty otherwise
    checks = sum([src_ok, dst_ok, unrelated_ok])
    if checks == 3:
        return 1.0
    if checks == 2:
        return 0.5
    if checks == 1:
        return -0.5
    return -1.0

# Rule 3
def rule_reward(state, action, choice):
    import re

    # Parse move action: returns src_slot like [I6], dst_slot like [A1], qty as int
    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z0-9]+\])\s*to\s*(\[[A-Z0-9]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, 0
        return m.group(1), m.group(2), int(m.group(3))

    # Parse inventory into list of tuples: (name, slot, qty)
    def parse_items(s):
        items = []
        # Matches lines like "- blue_dye [I2] quantity 1"
        for name, slot, qty in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            items.append((name.strip(), f'[{slot}]', int(qty)))
        return items

    src_slot, dst_slot, qty = parse_move_action(action)
    # If not a move action, this rule does not apply -> return 0 (neutral)
    if not src_slot:
        return 0.0

    # Build dicts slot -> (name, qty)
    state_items = parse_items(state)
    choice_items = parse_items(choice)
    s_map = {slot: (name, q) for (name, slot, q) in state_items}
    c_map = {slot: (name, q) for (name, slot, q) in choice_items}

    # Source must exist in state
    if src_slot not in s_map:
        # invalid move from non-existent slot -> strong penalty
        return -1.0

    moved_name, src_prev = s_map[src_slot]
    dst_prev_name, dst_prev_q = c_map.get(dst_slot, (None, 0))
    # Destination previous from state (may exist in state)
    dst_prev_name_state, dst_prev_q_state = s_map.get(dst_slot, (None, 0))

    # Retrieve new quantities/names in choice
    src_new_name, src_new_q = c_map.get(src_slot, (None, 0))
    dst_new_name, dst_new_q = c_map.get(dst_slot, (None, 0))

    checks = 0
    total_checks = 3  # we will compute three checks

    # Check 1: destination increased by qty and destination item name equals moved item name
    dest_ok = False
    # Destination had some previous qty in state (dst_prev_q_state)
    if dst_new_name == moved_name and (dst_new_q - dst_prev_q_state) == qty:
        dest_ok = True
    # Also accept case where destination did not exist before and now exists with qty == qty and name matches
    if dst_slot not in s_map and dst_new_name == moved_name and dst_new_q == qty:
        dest_ok = True
    if dest_ok:
        checks += 1

    # Check 2: source decreased by qty and source item name is same as moved_name (or source removed when qty==0)
    src_ok = False
    if src_new_name in (moved_name, None):
        # If source removed -> src_new_q treated as 0
        new_q = src_new_q if src_new_name is not None else 0
        if (src_prev - new_q) == qty:
            src_ok = True
    if src_ok:
        checks += 1

    # Check 3: unrelated item totals (excluding moved_name and slot '[0]') unchanged
    def totals(items):
        d = {}
        for name, slot, cnt in items:
            if name == moved_name:
                continue
            if slot == '[0]':  # allow changes in special output slot
                continue
            d[name] = d.get(name, 0) + cnt
        return d

    t_state = totals(state_items)
    t_choice = totals(choice_items)
    if all(t_state.get(k, 0) == t_choice.get(k, 0) for k in set(t_state) | set(t_choice)):
        checks += 1

    # Map checks (0..3) to score in [-1,1]
    score = checks / total_checks  # 0.0 .. 1.0
    return score * 2.0 - 1.0  # map to -1 .. 1

# Rule 4
def rule_reward(state, action, choice):
    import re

    # parse move action
    m = re.search(r'move:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', action)
    if not m:
        # Rule is only for move actions; if not move, do not apply (neutral)
        return 0.0
    src_slot, dst_slot, qty = m.group(1), m.group(2), int(m.group(3))

    # apply only for crafting tasks
    if not state.strip().startswith('Craft an item of type:'):
        return 0.0

    # parse inventory lines into list of (name, slot, qty)
    def parse_items(text):
        items = []
        for name, slot, q in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', text):
            items.append((name.strip(), f'[{slot}]', int(q)))
        return items

    s_items = parse_items(state)
    c_items = parse_items(choice)

    # build slot -> (name, qty) maps
    s_map = {slot: (name, qty) for name, slot, qty in s_items}
    c_map = {slot: (name, qty) for name, slot, qty in c_items}

    # Check that source slot existed in state
    if src_slot not in s_map:
        # cannot validate move if source absent; penalize slightly
        return -0.6

    moved_name, src_prev_qty = s_map[src_slot]
    dst_prev_name, dst_prev_qty = c_map.get(dst_slot, (None, 0))
    # But need destination previous on state (could be absent in state)
    dst_prev_state = s_map.get(dst_slot, (None, 0))[1]

    # In choice, get new src and dst entries (if missing, treat qty as 0)
    src_new_name, src_new_qty = c_map.get(src_slot, (moved_name, 0))
    dst_new_name, dst_new_qty = c_map.get(dst_slot, (moved_name, 0))

    # Move correctness checks:
    move_dst_inc_ok = (dst_new_name == moved_name) and ((dst_new_qty - dst_prev_state) == qty)
    # Source decreased by qty (allow name unchanged or absent)
    src_decreased_ok = ((src_new_qty + qty) == src_prev_qty) or (src_new_qty == 0 and src_prev_qty == qty)
    # Also ensure the source name in choice is either same or absent
    src_name_ok = (src_new_name == moved_name) or (src_new_qty == 0)

    move_ok = move_dst_inc_ok and src_decreased_ok and src_name_ok

    # Check presence of [0] in choice
    zero_present = any(slot == '[0]' for _, slot, _ in c_items)

    # Scoring logic
    if move_ok and zero_present:
        return 1.0   # best: move correct and [0] present
    if move_ok and not zero_present:
        return 0.3   # partial: move correct but missing required [0] slot
    if (not move_ok) and zero_present:
        return -0.2  # has [0] but move wrong -> negative but small
    return -1.0     # move wrong and no [0] -> strongly negative

# Rule 5
def rule_reward(state, action, choice):
    import re
    # parse target craft item
    m = re.search(r'Craft an item of type:\s*([^\n\r]+)', action)
    target = m.group(1).strip() if m else None

    # parse move if present
    m2 = re.search(r'move:\s*from\s*(\[[A-Z0-9]+\])\s*to\s*(\[[A-Z0-9]+\])\s*with\s*quantity\s*(\d+)', action)
    src_slot, dst_slot, qty = (m2.group(1), m2.group(2), int(m2.group(3))) if m2 else (None, None, 0)

    # helper to parse inventory text into list of (name, slot, qty)
    def parse_items(text):
        items = []
        # match lines like: - item_name [I15] quantity 1
        for name, slot, q in re.findall(r'-\s+([^\[\n\r]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', text):
            items.append((name.strip(), f'[{slot}]', int(q)))
        return items

    state_items = parse_items(state)
    choice_items = parse_items(choice)

    # map slot -> (name, qty) for easy lookup
    def slot_map(items):
        d = {}
        for name, slot, q in items:
            d[slot] = (name, q)
        return d

    s_map = slot_map(state_items)
    c_map = slot_map(choice_items)

    # map name -> total qty (across non-output slots) for conservation check
    def name_totals(items, ignore_slots=None, ignore_names=None):
        ignore_slots = set(ignore_slots or [])
        ignore_names = set(ignore_names or [])
        tot = {}
        for name, slot, q in items:
            if slot in ignore_slots or name in ignore_names:
                continue
            tot[name] = tot.get(name, 0) + q
        return tot

    # 1) Output check: target must appear at slot [0] with quantity >=1 and name exact match
    output_ok = 0.0
    if target is not None:
        out = c_map.get('[0]')
        if out and out[0] == target and out[1] >= 1:
            # prefer quantity 1 but accept >=1
            output_ok = 1.0
        else:
            output_ok = 0.0
    else:
        # if no explicit craft target parsed, be neutral (no strong signal)
        output_ok = 0.5

    # 2) Move check: ensure moved item name M is decreased at src by qty and increased at dst by qty
    move_ok = 0.0
    if src_slot is None:
        # if no move present in action, we consider move check neutral
        move_ok = 0.5
        moved_name = None
    else:
        src = s_map.get(src_slot)
        if not src:
            # source slot not present in original state -> invalid action; penalize
            move_ok = 0.0
            moved_name = None
        else:
            moved_name, src_prev = src
            # quantities in choice
            c_src = c_map.get(src_slot)
            c_dst = c_map.get(dst_slot)
            src_new_qty = c_src[1] if c_src and c_src[0] == moved_name else 0
            # if the dst slot in choice has a different name, it's wrong
            dst_name_ok = (c_dst is not None and c_dst[0] == moved_name) or (c_dst is None)
            dst_prev_qty = s_map.get(dst_slot, (moved_name, 0))[1] if s_map.get(dst_slot, (None,0))[0] == moved_name else s_map.get(dst_slot, (None,0))[1] if s_map.get(dst_slot) else 0
            dst_new_qty = c_dst[1] if c_dst and c_dst[0] == moved_name else (0 if c_dst is None else 0)

            # check decreases/increases by exactly qty
            src_decreased = (src_prev - src_new_qty) == qty
            dst_increased = (dst_new_qty - dst_prev_qty) == qty

            if src_decreased and dst_increased and dst_name_ok:
                move_ok = 1.0
            else:
                move_ok = 0.0

    # 3) Conservation check: totals of all names except moved_name and the output slot must remain equal
    # Build totals ignoring slot [0] and ignoring moved_name
    ignore_names = set()
    if src_slot and moved_name:
        ignore_names.add(moved_name)
    s_tot = name_totals(state_items, ignore_slots=['[0]'], ignore_names=ignore_names)
    c_tot = name_totals(choice_items, ignore_slots=['[0]'], ignore_names=ignore_names)

    conservation_ok = 1.0 if s_tot == c_tot else 0.0

    # combine scores with weights: output highest, then move, then conservation
    score = 0.55 * output_ok + 0.30 * move_ok + 0.15 * conservation_ok
    # map from [0,1] to [-1,1]
    final = score * 2.0 - 1.0
    # clamp
    if final > 1.0: final = 1.0
    if final < -1.0: final = -1.0
    return final

# Rule 6
def rule_reward(state, action, choice):
    import re

    def parse_items(s):
        # returns dict slot -> (name, qty)
        items = {}
        for name, slot, qty in re.findall(r'-\s+([^\[]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = name.strip()
            slot = f'[{slot}]'
            qty = int(qty)
            items[slot] = (name, qty)
        return items

    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z0-9]+\])\s*to\s*(\[[A-Z0-9]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None
        return m.group(1), m.group(2), int(m.group(3))

    # Only apply this rule when crafting is requested and action is a move
    if 'Craft an item of type:' not in state:
        return 0.0

    mv = parse_move_action(action)
    if not mv:
        return 0.0
    src_slot, dst_slot, qty = mv

    s_items = parse_items(state)
    c_items = parse_items(choice)

    # moved item must exist in source in the original state
    if src_slot not in s_items:
        # invalid move source; strongly penalize
        return -1.0

    moved_name, src_prev = s_items[src_slot]
    # get destination previous qty (could be absent)
    dst_prev = s_items.get(dst_slot, (moved_name, 0))[1]

    # get new values in choice (if absent treat qty 0)
    c_src_name, c_src_qty = c_items.get(src_slot, (moved_name, 0))
    c_dst_name, c_dst_qty = c_items.get(dst_slot, (moved_name, 0))

    # 1) Check move correctness (name consistency and exact quantity change)
    src_decreased = (src_prev - c_src_qty) == qty
    dst_increased = (c_dst_qty - dst_prev) == qty
    names_consistent = (c_dst_name == moved_name or c_dst_name is None or dst_prev == 0)

    move_score = 0.0
    if src_decreased and dst_increased and names_consistent:
        move_score = 1.0
    else:
        # give partial credit for one-sided correctness
        move_score = 0.5 * (1.0 if src_decreased else 0.0) + 0.5 * (1.0 if dst_increased else 0.0)

    # 2) Check output slot [0] exists or increased
    s_out_qty = s_items.get('[0]', (None, 0))[1]
    c_out_qty = c_items.get('[0]', (None, 0))[1]
    output_created_or_increased = c_out_qty > s_out_qty

    output_score = 1.0 if output_created_or_increased else 0.0

    # 3) Check that unrelated items did not change totals
    # build totals by name excluding moved_name and excluding slot [0]
    def totals(items):
        d = {}
        for slot, (name, qty) in items.items():
            if slot == '[0]':
                continue
            if name == moved_name:
                continue
            d[name] = d.get(name, 0) + qty
        return d

    s_tot = totals(s_items)
    c_tot = totals(c_items)
    unrelated_unchanged = 1.0 if s_tot == c_tot else 0.0

    # Combine scores with weights (favor move correctness and presence of output)
    base = 0.6 * move_score + 0.3 * output_score + 0.1 * unrelated_unchanged

    # Map base [0,1] to [-1,1] so positive indicates likely correct
    reward = base * 2.0 - 1.0

    # If move was completely wrong (no src decrease and no dst increase), strongly penalize
    if move_score == 0.0:
        reward = -1.0

    return float(max(-1.0, min(1.0, reward)))

# Rule 7
def rule_reward(state, action, choice):
    import re

    def parse_action(a):
        # returns src_slot like '[I24]', dst_slot like '[A1]', and qty int
        m = re.search(r'move:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, None
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns list of (name, slot, qty) with slot like '[I17]', '[A1]', '[0]'
        items = []
        # pattern matches "- name [SLOT] quantity N"
        for name, slot, qty in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            items.append((name.strip(), f'[{slot}]', int(qty)))
        return items

    src_slot, dst_slot, qty = parse_action(action)
    # Only apply this rule for move actions of the observed pattern
    if src_slot is None:
        return 0.0

    q_items = parse_items(state)
    c_items = parse_items(choice)

    # Build slot->(name,qty) maps
    q_slot_map = {slot: (name, count) for (name, slot, count) in q_items}
    c_slot_map = {slot: (name, count) for (name, slot, count) in c_items}

    # If source slot not present in original state, cannot validate; return neutral small negative
    if src_slot not in q_slot_map:
        return -0.6

    moved_name, src_prev = q_slot_map[src_slot]
    # destination previous quantity for the moved item (may be absent)
    dst_prev_name, dst_prev = c_name_prev = (None, 0)
    if dst_slot in q_slot_map:
        dst_prev_name, dst_prev = q_slot_map[dst_slot]
    else:
        # dst absent in original -> dst_prev stays 0
        dst_prev_name, dst_prev = (None, 0)

    # Find destination entry in choice (may be present or new)
    dst_new_name, dst_new = c_slot_map.get(dst_slot, (None, 0))
    # Find source entry in choice
    src_new_name, src_new = c_slot_map.get(src_slot, (None, 0))

    checks = 0

    # Condition 1: destination increased by qty and item name matches moved item
    # Accept if dst_new_name == moved_name and dst_new - dst_prev == qty
    if dst_new_name == moved_name and (dst_new - dst_prev) == qty:
        checks += 1

    # Condition 2: source decreased by qty (or removed). Accept if src_prev - src_new == qty
    # If source name changed to something else, treat as failed (unless it's absent which is ok if decreased)
    src_name_ok = (src_new_name in (moved_name, None))
    if src_name_ok and (src_prev - src_new) == qty:
        checks += 1

    # Condition 3: No changes to unrelated items except slot '[0]'.
    # Compute totals per item-name across all slots except slot '[0]' and excluding moved_name.
    def totals(items):
        d = {}
        for name, slot, count in items:
            if slot == '[0]':
                continue
            if name == moved_name:
                continue
            d[name] = d.get(name, 0) + count
        return d

    tq = totals(q_items)
    tc = totals(c_items)
    # If totals equal for all item names, condition satisfied
    if tq == tc:
        checks += 1

    # Map checks (0..3) to score in [-1, 1]
    score = (2.0 * (checks / 3.0)) - 1.0
    # Ensure bounds
    if score < -1.0:
        score = -1.0
    if score > 1.0:
        score = 1.0
    return float(score)

# Rule 8
def rule_reward(state, action, choice):
    import re

    def parse_action(a):
        # match moves of form: move: from [I2] to [B1] with quantity 1
        m = re.search(r'move:\s*from\s*\[([A-Z]\d+)\]\s*to\s*\[([A-Z]\d+)\]\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None
        return (m.group(1), m.group(2), int(m.group(3)))

    def parse_items(s):
        # returns dict slot -> (name, qty)
        items = {}
        for name, slot, qty in re.findall(r'-\s+([^\[]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            items[f'[{slot}]'] = (name.strip(), int(qty))
        return items

    act = parse_action(action)
    if not act:
        return 0.0
    src_slot, dst_slot, q = act
    src_slot = f'[{src_slot}]'
    dst_slot = f'[{dst_slot}]'

    q_items = parse_items(state)
    c_items = parse_items(choice)

    # helper to get qty/name with defaults
    def get_slot(items, slot):
        return items.get(slot, (None, 0))

    # If source not present in state, rule not applicable
    src_name, src_qty = get_slot(q_items, src_slot)
    if src_name is None:
        return 0.0

    # Only apply rule when destination is a crafting-grid slot (A/B/C prefix)
    def is_craft_slot(slot):
        return re.match(r'^\[[ABC]\d+\]$', slot) is not None

    if not is_craft_slot(dst_slot):
        return 0.0

    # Source must have enough quantity to move
    if src_qty < q:
        return 0.0

    # find other crafting-grid slots that already contain the same item type
    same_in_craft = 0
    for s, (n, qty) in q_items.items():
        if is_craft_slot(s) and n == src_name:
            same_in_craft += qty

    # If destination already had some of this item in the craft grid, that counts too
    # Evaluate completion by checking same_in_craft + q >= 2 (for 2-item recipes)
    if (same_in_craft + q) < 2:
        # rule not applicable, move does not (plausibly) complete a 2-item craft
        return 0.0

    # Now the rule applies: check move correctness
    # Source should decrease by q
    _, src_qty_choice = get_slot(c_items, src_slot)
    # Destination should increase by q for same item name (destination may have been empty or same name)
    dst_name_choice, dst_qty_choice = get_slot(c_items, dst_slot)
    dst_name_state, dst_qty_state = get_slot(q_items, dst_slot)

    move_ok = -1.0  # default penalty for incorrect move
    # Validate that destination ends up with the same item name as source and quantity increased by q
    # Accept cases where destination was empty (dst_name_state is None) or already had the same name
    if dst_name_choice == src_name and (dst_qty_choice - dst_qty_state) == q:
        # Validate source decreased by q (source may be absent in choice, treated as 0)
        if (src_qty - src_qty_choice) == q:
            move_ok = 1.0

    # Check for crafted output at [0]: quantity increased (or new item appears)
    out_name_state, out_qty_state = get_slot(q_items, '[0]')
    out_name_choice, out_qty_choice = get_slot(c_items, '[0]')

    # Be conservative: do not penalize the absence of an output increase.
    # Reward only if output increased, and penalize only if output decreased unexpectedly.
    crafted_ok = 0.0
    if out_qty_choice > out_qty_state:
        crafted_ok = 1.0
    elif out_qty_choice < out_qty_state:
        crafted_ok = -1.0
    else:
        crafted_ok = 0.0

    # Penalize unrelated changes: allow changes only to src slot, dst slot, and [0]
    unrelated_ok = 1.0
    # Build mapping of name+slot -> qty to compare
    for slot, (name, qty) in q_items.items():
        if slot in (src_slot, dst_slot, '[0]'):
            continue
        # if a slot changed in choice compared to state, consider it unrelated change
        c_name, c_qty = get_slot(c_items, slot)
        if c_name != name or c_qty != qty:
            unrelated_ok = -1.0
            break
    # Also ensure choice has no extra new slots (except dst, src, [0])
    if unrelated_ok > 0:
        for slot, (name, qty) in c_items.items():
            if slot in (src_slot, dst_slot, '[0]'):
                continue
            if slot not in q_items:
                # a new slot appeared that's unrelated
                unrelated_ok = -1.0
                break

    # Combine scores and normalize to [-1,1]
    score = (move_ok + crafted_ok + unrelated_ok) / 3.0

    # Clamp
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return float(score)

# Rule 9
def rule_reward(state, action, choice):
    import re

    def parse_action(a):
        # expects: move: from [I17] to [B2] with quantity 1
        m = re.search(r'move:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, None
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns dict slot -> (name, qty)
        items = {}
        for name, slot, qty in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            slot_id = f'[{slot}]'
            items[slot_id] = (name.strip(), int(qty))
        return items

    src_slot, dst_slot, qty = parse_action(action)
    if not src_slot:
        # not a move action we handle
        return -1.0

    state_map = parse_items(state)
    choice_map = parse_items(choice)

    # Source must exist in the state
    if src_slot not in state_map:
        return -1.0

    moved_name, src_prev = state_map[src_slot]
    # quantities in choice (missing treated as 0)
    src_new = choice_map.get(src_slot, (moved_name, 0))[1]
    dst_prev = state_map.get(dst_slot, (moved_name, 0))[1]
    dst_entry = choice_map.get(dst_slot, (None, 0))
    dst_new_name, dst_new = dst_entry

    checks = 0
    total_checks = 5

    # C1: destination increased by q
    if (dst_new - dst_prev) == qty:
        checks += 1

    # C2: source decreased by q
    if (src_prev - src_new) == qty:
        checks += 1

    # C3: destination item name matches moved_name
    if dst_new_name == moved_name:
        checks += 1

    # C4: other slots that had moved_name remain unchanged
    ok_c4 = True
    for s, (n, q_prev) in state_map.items():
        if s in (src_slot, dst_slot):
            continue
        if n == moved_name:
            q_choice = choice_map.get(s, (n, 0))[1]
            if q_choice != q_prev:
                ok_c4 = False
                break
    if ok_c4:
        checks += 1

    # C5: no other slot (except [0]) changed name or quantity
    ok_c5 = True
    for s, (n, q_prev) in state_map.items():
        if s == '[0]':
            continue
        if s in (src_slot, dst_slot):
            # we've already checked these
            continue
        choice_entry = choice_map.get(s)
        if choice_entry is None:
            # slot removed -> only allowed if it was the source and decreased to 0 (handled above).
            ok_c5 = False
            break
        cname, cqty = choice_entry
        if cname != n or cqty != q_prev:
            ok_c5 = False
            break
    # Also ensure choice did not introduce new unrelated slots (except destination and [0])
    for s, (n, cqty) in choice_map.items():
        if s == '[0]' or s == dst_slot:
            continue
        if s not in state_map:
            ok_c5 = False
            break

    if ok_c5:
        checks += 1

    # Map checks (0..total_checks) to [-1,1]
    score = (checks / float(total_checks)) * 2.0 - 1.0
    return float(score)

# Rule 10
def rule_reward(state, action, choice):
    import re

    def parse_items(text):
        # returns dict slot -> (name, qty)
        items = {}
        for name, slot, qty in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', text):
            slot_key = f'[{slot}]'
            items[slot_key] = (name.strip(), int(qty))
        return items

    # parse craft target from state's first line if present
    m_target = re.search(r'Craft an item of type:\s*([^\n\r]+)', state)
    craft_target = m_target.group(1).strip() if m_target else None

    # parse action: move
    m_action = re.search(r'move:\s*from\s*(\[[A-Z0-9]+\])\s*to\s*(\[[A-Z0-9]+\])\s*with\s*quantity\s*(\d+)', action)
    if not m_action:
        return 0.0
    src_slot, dst_slot, qty = m_action.group(1), m_action.group(2), int(m_action.group(3))

    s_items = parse_items(state)
    c_items = parse_items(choice)

    # helper to get name/qty
    def get(items, slot):
        return items.get(slot, (None, 0))

    score = 0.0

    # 1) Basic correctness: moved item reflected (src decreased by qty, dst increased by qty, same name)
    src_name, src_prev = get(s_items, src_slot)
    dst_name, dst_prev = get(s_items, dst_slot)
    src_new_name, src_new = get(c_items, src_slot)
    dst_new_name, dst_new = get(c_items, dst_slot)

    moved_reflected = False
    if src_name is not None:
        # destination may previously hold different name; require destination after move to have the moved item's name
        if dst_new_name == src_name and (dst_new - dst_prev) == qty:
            # source should decrease by qty (or disappear)
            if (src_prev - src_new) == qty or (src_new_name in (None, src_name) and (src_prev - src_new) == qty):
                moved_reflected = True
    else:
        # if source not present in state (rare), require destination decreased or increased appropriately
        if dst_new - dst_prev == qty:
            moved_reflected = True

    if moved_reflected:
        score += 0.4
    else:
        score -= 0.6  # fairly strong penalty if move quantities not reflected

    # 2) If destination is a crafting-slot (A/B/C), require output [0] to appear/update for craft_target
    if re.match(r'\[[ABC]\d*\]', dst_slot):
        if craft_target:
            out_name, out_prev = get(s_items, '[0]')
            out_new_name, out_new = get(c_items, '[0]')
            # Accept either creation of the craft_target at [0] or increase in its qty
            if out_new_name == craft_target and out_new >= 1:
                score += 0.5
            else:
                score -= 0.8

    # 3) If source is [0] (taking crafted item into inventory), require crafting inputs (A/B/C slots) to be consumed
    if src_slot == '[0]':
        # check that at least one A/B/C slot decreases in quantity in choice vs state
        consumed_any = False
        for slot, (name, prev_q) in s_items.items():
            if re.match(r'\[[ABC]\d*\]', slot):
                new_q = c_items.get(slot, (None, 0))[1]
                if new_q < prev_q:
                    consumed_any = True
                    break
        if consumed_any:
            score += 0.5
        else:
            score -= 0.9

    # 4) Penalize large unexpected changes in unrelated items
    # Build totals for unrelated items (exclude moved item name, exclude crafting slots A/B/C and slot [0])
    def totals(items):
        d = {}
        for slot, (name, qty_) in items.items():
            if re.match(r'\[[ABC]\d*\]', slot):
                continue
            if slot == '[0]':
                continue
            d[name] = d.get(name, 0) + qty_
        return d

    moved_name = src_name
    s_tot = totals(s_items)
    c_tot = totals(c_items)

    # ignore differences for the moved item (it may appear in totals under different slots)
    if moved_name in s_tot:
        s_tot[moved_name] = 0
    if moved_name in c_tot:
        c_tot[moved_name] = 0

    # count how many item totals changed
    changed = 0
    keys = set(s_tot.keys()) | set(c_tot.keys())
    for k in keys:
        if s_tot.get(k, 0) != c_tot.get(k, 0):
            changed += 1

    # allow 0-1 unrelated changes; penalize many
    if changed == 0:
        score += 0.1
    elif changed == 1:
        score += 0.0
    else:
        score -= min(0.7, 0.2 * (changed - 1))

    # clamp to [-1, 1]
    if score > 1:
        score = 1.0
    if score < -1:
        score = -1.0

    return float(score)

# Rule 11
def rule_reward(state, action, choice):
    import re

    def parse_action(a):
        # match "smelt:" or "move:" with from/to/quantity
        m = re.search(r'(smelt|move):\s*from\s*(\[[^\]]+\])\s*to\s*(\[[^\]]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None
        return {'type': m.group(1), 'src_slot': m.group(2), 'dst_slot': m.group(3), 'q': int(m.group(4))}

    def parse_items(s):
        # returns dict slot -> (name, qty)
        items = {}
        # pattern: - <name> [SLOT] quantity N
        for name, slot, qty in re.findall(r'-\s+([^\[]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            items[f'[{slot}]'] = (name.strip(), int(qty))
        return items

    act = parse_action(action)
    if not act:
        return 0.0

    src_slot = act['src_slot']
    dst_slot = act['dst_slot']
    q = act['q']
    typ = act['type']

    # basic validation
    if q <= 0:
        return 0.0

    s_items = parse_items(state)
    c_items = parse_items(choice)

    def get_slot(items, slot):
        return items.get(slot, (None, 0))

    src_name, src_qty = get_slot(s_items, src_slot)
    # Be conservative: if the source is missing in the original state, avoid a maximal -1.
    # Return a mild penalty so we don't flip correct answers due to parsing/format differences.
    if src_name is None:
        return -0.5

    c_src_name, c_src_qty = get_slot(c_items, src_slot)
    c_dst_name, c_dst_qty = get_slot(c_items, dst_slot)
    _, s_dst_qty = get_slot(s_items, dst_slot)

    checks = []

    # Source decreased by q (allow exact decrease). If candidate didn't reflect the decrease,
    # be forgiving but mark as failed.
    src_decreased = (c_src_qty == src_qty - q)
    checks.append(('src_decreased', src_decreased))

    # Destination increased by q (allow when original destination slot is missing -> s_dst_qty==0)
    dst_increased = (c_dst_qty == s_dst_qty + q)
    checks.append(('dst_increased', dst_increased))

    # Name checks: conservative
    if typ == 'move':
        # prefer destination name equals source name; if destination name missing, don't immediately fatal
        name_ok = (c_dst_name == src_name)
        checks.append(('move_dst_name_matches_src', name_ok))
    else:  # smelt
        # prefer destination name different from source name; allow missing destination name to be treated as unknown (fail but not fatal)
        name_ok = (c_dst_name is not None) and (c_dst_name != src_name)
        checks.append(('smelt_dst_name_not_src', name_ok))

    # Totals per name helper
    def totals_per_name(items):
        d = {}
        for (name, qty) in items.values():
            d[name] = d.get(name, 0) + qty
        return d

    s_tot = totals_per_name(s_items)
    c_tot = totals_per_name(c_items)

    # Determine names to ignore:
    ignore_names = set()
    # Always ignore the source name for move (it moves)
    if typ == 'move':
        ignore_names.add(src_name)
    else:
        # for smelt, ignore source name and the resulting destination name if present in candidate
        ignore_names.add(src_name)
        if c_dst_name:
            ignore_names.add(c_dst_name)

    # Additionally ignore any name that appears only in slot "[0]" on either side (typical output/result slot)
    def names_only_in_slot_zero(items):
        res = set()
        for slot, (name, qty) in items.items():
            if slot == '[0]':
                res.add(name)
        return res

    s_slot0_names = names_only_in_slot_zero(s_items)
    c_slot0_names = names_only_in_slot_zero(c_items)
    # If a name appears only as slot 0 in either side (and not elsewhere), ignore it for totals comparison:
    for name in s_slot0_names | c_slot0_names:
        # only ignore if that name does not also appear in other slots (i.e., only in slot 0)
        in_s_elsewhere = any((n == name and slot != '[0]') for slot, (n, _) in s_items.items())
        in_c_elsewhere = any((n == name and slot != '[0]') for slot, (n, _) in c_items.items())
        if not in_s_elsewhere and not in_c_elsewhere:
            ignore_names.add(name)

    # Totals comparison: compare only names that appear in both state and candidate (intersection),
    # excluding ignore_names. This avoids penalizing names that appear only on one side (e.g., result slot).
    all_names_intersection = set(s_tot.keys()) & set(c_tot.keys())
    # Exclude ignored names
    compare_names = [name for name in all_names_intersection if name not in ignore_names]

    totals_ok = True
    for name in compare_names:
        if s_tot.get(name, 0) != c_tot.get(name, 0):
            totals_ok = False
            break
    checks.append(('totals_unchanged_intersection', totals_ok))

    # Compute score: ratio of passed checks
    passed = sum(1 for _, v in checks if v)
    total = len(checks)
    score = 2.0 * (passed / total) - 1.0

    # Strong contradictions should still be penalized:
    # If move and destination increased but destination name != source name -> strong wrong
    if typ == 'move' and dst_increased and (c_dst_name is not None) and (c_dst_name != src_name):
        return -1.0
    # If smelt and destination increased but destination name == source name -> strong wrong
    if typ == 'smelt' and dst_increased and (c_dst_name == src_name):
        return -1.0

    # Avoid returning extreme values for small failures; clamp to [-1,1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0

    return score

# Rule 12
def rule_reward(state, action, choice):
    import re

    def parse_crafted_item(s):
        m = re.search(r'Craft an item of type:\s*([^\n\r]+)', s)
        return m.group(1).strip() if m else None

    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z0-9]+\])\s*to\s*(\[[A-Z0-9]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, 0
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        items = []
        for name, slot, qty in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            items.append((name.strip(), f'[{slot}]', int(qty)))
        return items

    crafted = parse_crafted_item(state)
    if not crafted:
        # This rule only applies to craft states
        return 0.0

    src, dst, q = parse_move_action(action)
    if not src:
        return 0.0

    s_items = parse_items(state)
    c_items = parse_items(choice)

    s_by_slot = {slot: (name, qty) for (name, slot, qty) in s_items}
    c_by_slot = {slot: (name, qty) for (name, slot, qty) in c_items}

    def qty_in(d, slot):
        return d.get(slot, (None, 0))[1]

    def name_in(d, slot):
        return d.get(slot, (None, 0))[0]

    # Move correctness (strict)
    src_prev_name = name_in(s_by_slot, src)
    src_prev_q = qty_in(s_by_slot, src)
    dst_prev_name = name_in(s_by_slot, dst)
    dst_prev_q = qty_in(s_by_slot, dst)
    src_new_name = name_in(c_by_slot, src)
    src_new_q = qty_in(c_by_slot, src)
    dst_new_name = name_in(c_by_slot, dst)
    dst_new_q = qty_in(c_by_slot, dst)

    move_ok = False
    # If the source had an item before, require it decreased by q; destination must increase by q and preserve the moved item's name
    if src_prev_name:
        if (src_prev_q - src_new_q == q):
            # destination should increase by q
            if (dst_new_q - dst_prev_q == q):
                # destination's resulting name should match the moved item name when possible
                if dst_new_name == src_prev_name:
                    move_ok = True
    else:
        # If source had no item previously, at minimum destination should have increased by q
        if dst_new_q - dst_prev_q == q:
            move_ok = True

    # Output ([0]) handling: strict only if move touches [0]; optional bonus otherwise
    output_prev_name = name_in(s_by_slot, '[0]')
    output_prev_q = qty_in(s_by_slot, '[0]')
    output_new_name = name_in(c_by_slot, '[0]')
    output_new_q = qty_in(c_by_slot, '[0]')

    output_ok_strict = False
    output_optional_bonus = False
    move_touches_output = (src == '[0]' or dst == '[0]')

    if move_touches_output:
        if src == '[0]':
            # Moving from [0] must reduce [0] by q (name may become None if removed)
            if output_prev_q - output_new_q == q:
                output_ok_strict = True
        elif dst == '[0]':
            # Moving into [0] must increase [0] by q and name should match moved item name when possible
            if output_new_q - output_prev_q == q:
                if (src_prev_name is None) or (output_new_name == src_prev_name):
                    output_ok_strict = True
    else:
        # Do not penalize absence of crafted item at [0]. Give a small positive signal if the crafted item appears there.
        if output_new_name == crafted and output_new_q >= 1:
            output_optional_bonus = True

    # Unrelated items unchanged (conservative):
    # If move does not touch [0], compare totals by name excluding slot [0] (so outputs placed only at [0] do not trigger a penalty).
    # If move touches [0], compare totals across whole inventory (strict).
    def totals_by_name(items, exclude_slots=None):
        exclude_slots = exclude_slots or set()
        d = {}
        for name, slot, count in items:
            if slot in exclude_slots:
                continue
            if not name:
                continue
            d[name] = d.get(name, 0) + count
        return d

    if move_touches_output:
        tot_s = totals_by_name(s_items, exclude_slots=set())
        tot_c = totals_by_name(c_items, exclude_slots=set())
    else:
        # Exclude counts in slot [0] when checking unrelated totals
        tot_s = totals_by_name(s_items, exclude_slots={'[0]'})
        tot_c = totals_by_name(c_items, exclude_slots={'[0]'})

    moved_names = set()
    if src_prev_name:
        moved_names.add(src_prev_name)
    if dst_prev_name:
        moved_names.add(dst_prev_name)

    # remove crafted and moved names from comparison
    unrelated_ok = True
    all_names = set(tot_s.keys()) | set(tot_c.keys())
    for name in all_names:
        if name == crafted:
            continue
        if name in moved_names:
            continue
        s_count = tot_s.get(name, 0)
        c_count = tot_c.get(name, 0)
        if s_count != c_count:
            unrelated_ok = False
            break

    # Aggregate checks conservatively:
    # - move correctness is mandatory (1)
    # - unrelated_ok is mandatory (1)
    # - output_ok_strict is mandatory only when move touches [0] (1)
    total_checks = 1 + 1 + (1 if move_touches_output else 0)
    checks = 0
    if move_ok:
        checks += 1
    if unrelated_ok:
        checks += 1
    if move_touches_output:
        if output_ok_strict:
            checks += 1

    # Map checks to [-1, 1]
    if total_checks == 0:
        score = 0.0
    else:
        score = (checks / total_checks) * 2.0 - 1.0

    # If move didn't touch [0], give a small bonus for producing the crafted item at [0] (do not penalize its absence)
    if (not move_touches_output) and output_optional_bonus:
        score += 0.2

    # Clamp to [-1, 1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0

    return float(score)

# Rule 13
def rule_reward(state, action, choice):
    import re

    # Parse move action
    m = re.search(r'move:\s*from\s*(\[[A-Z0-9]+\])\s*to\s*(\[[A-Z0-9]+\])\s*with\s*quantity\s*(\d+)', action, re.IGNORECASE)
    if not m:
        return 0.0

    src_slot, dst_slot, q = m.group(1), m.group(2), int(m.group(3))

    # Parse inventories into slot -> (name, qty)
    def parse_items(s):
        items = {}
        for name, slot, qty in re.findall(r'-\s+([^\[]+?)\s*\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            slot_token = f'[{slot}]'
            items[slot_token] = (name.strip(), int(qty))
        return items

    prior = parse_items(state)
    cand = parse_items(choice)

    # Source must exist in prior to validate the "from"
    if src_slot not in prior:
        return 0.0

    moved_name, prior_src_qty = prior[src_slot]

    # Basic q validation: impossible move
    if q <= 0 or q > prior_src_qty:
        return -1.0

    # Helper: total for a given name
    def total_for_name(d, name):
        return sum(qty for (nm, qty) in d.values() if nm == name)

    prior_total_moved = total_for_name(prior, moved_name)
    cand_total_moved = total_for_name(cand, moved_name)

    # Clear contradiction: conservation violated
    if cand_total_moved != prior_total_moved:
        return -1.0

    prior_dst_qty = prior.get(dst_slot, (None, 0))[1] if dst_slot in prior else 0

    # Helper: compute fraction of non-moved-name slots preserved exactly
    def non_moved_preservation_fraction(prior_map, cand_map, moved_name):
        prior_non_moved_slots = [s for s, (n, q) in prior_map.items() if n != moved_name]
        cand_non_moved_new = [s for s, (n, q) in cand_map.items() if n != moved_name and s not in prior_map]
        preserved = 0
        for slot in prior_non_moved_slots:
            p_name, p_qty = prior_map[slot]
            if slot in cand_map:
                c_name, c_qty = cand_map[slot]
                if c_name == p_name and c_qty == p_qty:
                    preserved += 1
        total_expected = len(prior_non_moved_slots) + len(cand_non_moved_new)
        if total_expected == 0:
            return 1.0
        return preserved / float(total_expected)

    # Decide strict mode
    strict_mode = (prior_total_moved == prior_src_qty) or (dst_slot in prior)

    # Strict mode: require explicit changes at DST and SRC; soften using other_fraction
    if strict_mode:
        dst_ok = False
        if dst_slot in cand:
            c_dst_name, c_dst_qty = cand[dst_slot]
            if c_dst_name == moved_name and c_dst_qty == prior_dst_qty + q:
                dst_ok = True
        # If DST was present in prior but candidate changed name at DST, that is a contradiction for strict check
        # (we leave dst_ok False and let combined score reflect mismatch)

        src_ok = False
        if src_slot in cand:
            c_src_name, c_src_qty = cand[src_slot]
            if c_src_name == moved_name and c_src_qty == prior_src_qty - q:
                src_ok = True
        else:
            # allowed only if fully moved
            if prior_src_qty == q:
                src_ok = True

        other_fraction = non_moved_preservation_fraction(prior, cand, moved_name)

        checks_sum = (1.0 if dst_ok else 0.0) + (1.0 if src_ok else 0.0) + other_fraction
        # Map 0..3 -> -1..1 linearly
        score = (checks_sum / 3.0) * 2.0 - 1.0
        # Be conservative: do not give a positive score unless at least src_ok is true (we expect the 'from' to be effected)
        if score > 0 and not src_ok:
            # If dst_ok but src_ok failed, reduce reward to 0.0 (inconclusive)
            return 0.0
        # Clamp
        if score < -1.0:
            score = -1.0
        if score > 1.0:
            score = 1.0
        return float(score)

    # Relaxed mode: moved_name appears multiple times in prior and DST not present in prior
    # We require explicit evidence at the declared SRC and DST (to avoid ambiguous +q/-q elsewhere)
    # Ensure DST is present in candidate with moved_name and increased by q
    if dst_slot not in cand:
        # No explicit DST evidence -> inconclusive
        return 0.0
    c_dst_name, c_dst_qty = cand[dst_slot]
    # Prior DST qty is 0 in this branch (dst not in prior), but keep general check
    if c_dst_name != moved_name or c_dst_qty != prior_dst_qty + q:
        # Candidate's DST does not show the expected increase -> inconclusive
        return 0.0

    # Ensure SRC decreased by q (or removed if fully moved)
    src_delta_ok = False
    if src_slot in cand:
        c_src_name, c_src_qty = cand[src_slot]
        if c_src_name == moved_name and c_src_qty == prior_src_qty - q:
            src_delta_ok = True
    else:
        if prior_src_qty == q:
            src_delta_ok = True

    if not src_delta_ok:
        # If the named source didn't show the expected decrease, be conservative
        return 0.0

    # Now compute non-moved preservation fraction to scale a modest positive reward
    other_fraction = non_moved_preservation_fraction(prior, cand, moved_name)
    # Give modest positive score scaled by preservation (more preserved -> higher)
    base = 0.5
    score = base * other_fraction
    # Ensure within [-1,1]
    if score < -1.0:
        score = -1.0
    if score > 1.0:
        score = 1.0
    return float(score)

# Rule 14
def rule_reward(state, action, choice):
    import re
    from collections import defaultdict

    def parse_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z0-9]+\])\s*to\s*(\[[A-Z0-9]+\])\s*with\s*quantity\s*(\d+)', a, re.IGNORECASE)
        if m:
            return ('move', m.group(1), m.group(2), int(m.group(3)))
        m = re.search(r'smelt:\s*from\s*(\[[A-Z0-9]+\])\s*to\s*(\[[A-Z0-9]+\])\s*with\s*quantity\s*(\d+)', a, re.IGNORECASE)
        if m:
            return ('smelt', m.group(1), m.group(2), int(m.group(3)))
        return (None, None, None, 0)

    def parse_items(s):
        # returns slot_map: slot -> (name, qty), and totals: name -> total_qty
        slot_map = {}
        totals = defaultdict(int)
        # pattern matches lines like "- slime_ball [A1] quantity 1"
        for name, slot, qty in re.findall(r'-\s+([^\[\n]+?)\s*\[([^\]]+)\]\s*quantity\s*(\d+)', s):
            name = name.strip()
            slot_token = f'[{slot}]'
            qty = int(qty)
            slot_map[slot_token] = (name, qty)
            totals[name] += qty
        return slot_map, dict(totals)

    act_type, src_slot, dst_slot, q = parse_action(action)
    state_slots, state_totals = parse_items(state)
    choice_slots, choice_totals = parse_items(choice)

    score = 0.0

    # If we cannot parse the action, give neutral score
    if act_type is None:
        return 0.0

    # Verify source exists in state
    if src_slot not in state_slots:
        # invalid action: source not present => strong penalty
        return -1.0

    src_name, src_prev_qty = state_slots[src_slot]
    src_choice_name, src_choice_qty = choice_slots.get(src_slot, (None, 0))

    # Check source decreased by q (be strict here — moving should reduce source)
    src_decreased = (src_prev_qty - src_choice_qty) == q and (src_choice_name in (src_name, None))
    if src_decreased:
        score += 0.4
    else:
        # less strict penalty for small mismatches; very bad if source increased or unchanged when expected to decrease
        if (src_choice_qty is not None) and (src_choice_qty >= src_prev_qty):
            score -= 0.7
        else:
            score -= 0.3

    # For move, check destination increased by q and name preserved (but be tolerant if destination appears under a different slot)
    if act_type == 'move':
        dst_prev_name, dst_prev_qty = state_slots.get(dst_slot, (None, 0))
        dst_choice_name, dst_choice_qty = choice_slots.get(dst_slot, (None, 0))
        # destination increase must equal q and name should match moved name
        dst_increased = (dst_choice_qty - dst_prev_qty) == q
        dst_name_preserved = (dst_choice_name == src_name)
        if dst_increased and dst_name_preserved:
            score += 0.4
        else:
            # allow some flexibility: maybe the moved item ended up in a different slot (same-name present elsewhere).
            # If some slot (other than dst_slot) gained exactly q of the same name, accept.
            moved_elsewhere = False
            for s, (nm, qty) in choice_slots.items():
                if s == dst_slot:  # already checked
                    continue
                if nm == src_name:
                    prev_qty = state_slots.get(s, (None, 0))[1]
                    if (qty - prev_qty) == q:
                        moved_elsewhere = True
                        break
            if moved_elsewhere:
                score += 0.3
            else:
                # only a mild penalty — be conservative
                score -= 0.4

    # For smelt, destination should increase by q (name may differ)
    if act_type == 'smelt':
        dst_prev_name, dst_prev_qty = state_slots.get(dst_slot, (None, 0))
        dst_choice_name, dst_choice_qty = choice_slots.get(dst_slot, (None, 0))
        dst_increased_correct = (dst_choice_qty - dst_prev_qty) == q
        if dst_increased_correct:
            score += 0.4
        else:
            score -= 0.6

    # Now check for produced items that are unaccounted for:
    # Compute per-item net changes
    net_changes = {}
    names = set(list(state_totals.keys()) + list(choice_totals.keys()))
    for n in names:
        net_changes[n] = choice_totals.get(n, 0) - state_totals.get(n, 0)

    # Helper: get slots for a given name in a slot_map
    def slots_for_name(slot_map, name):
        return [s for s, (nm, qty) in slot_map.items() if nm == name]

    produced_items = []
    weak_produced_items = []  # produced items that are only in result slot [0]
    for name, delta in net_changes.items():
        if delta <= 0:
            continue

        state_name_slots = set(slots_for_name(state_slots, name))
        choice_name_slots = set(slots_for_name(choice_slots, name))
        new_slots = choice_name_slots - state_name_slots

        # If the positive change is entirely in the special result slot "[0]", treat as expected artifact.
        if new_slots and all(s == '[0]' for s in new_slots):
            weak_produced_items.append((name, delta, sorted(list(new_slots))))
            continue

        # For smelt, if the increase corresponds exactly to the declared dst slot name and qty, it's accounted for
        if act_type == 'smelt':
            dst_choice_name = choice_slots.get(dst_slot, (None, None))[0]
            if name == dst_choice_name and delta == q:
                continue

        # For move: be conservative — only suspicious if the net increase cannot be reconciled
        if act_type == 'move':
            # If the net increase > 0 but there exists a corresponding decrease in totals that can explain it, allow.
            # We'll collect total decreases across all names (later) to compare.
            produced_items.append((name, delta, sorted(list(new_slots))))
            continue

        # Default: treat as produced/unexpected
        produced_items.append((name, delta, sorted(list(new_slots))))

    # Compute total decrease across all names (positive number means items were consumed)
    total_decrease = sum(max(0, state_totals.get(n, 0) - choice_totals.get(n, 0)) for n in names)

    # Now decide penalties/rewards conservatively:
    # Filter produced_items to exclude ones that are exclusively in [0] (weak_produced_items already separated).
    nonresult_produced = []
    for name, delta, new_slots in produced_items:
        # treat new_slots that are only '[0]' as weak (shouldn't be here, but be safe)
        if new_slots and all(s == '[0]' for s in new_slots):
            weak_produced_items.append((name, delta, new_slots))
            continue
        nonresult_produced.append((name, delta, new_slots))

    if nonresult_produced:
        total_produced_qty = sum(p[1] for p in nonresult_produced)
        includes_moved_name = any(p[0] == src_name for p in nonresult_produced)
        # Conservative decision: require clear mismatch: produced quantity strictly greater than decreases
        # or multiple produced items, or produced includes moved name.
        if total_produced_qty > total_decrease:
            # Now require stronger signal to apply strong penalty
            if (total_produced_qty >= 2) or includes_moved_name or (len(nonresult_produced) > 1):
                # Moderate penalty (less severe than before)
                score -= 0.7
            else:
                # single small-produced item that slightly exceeds decreases: mild penalty
                score -= 0.35
        else:
            # There is some decrease that can explain production — mild reward for plausible transformation
            score += 0.15

    # Weak produced items (only in result slot [0]) should not be penalized.
    # If there are both weak_produced_items and nonresult_produced, we already handled nonresult ones above.
    # If only weak_produced_items exist and everything else looks consistent, give a small positive boost.
    if (not nonresult_produced) and weak_produced_items:
        # If source decreased appropriately (we added +0.4 above) and destination is consistent, give small extra credit
        score += 0.1

    # Cap to [-1, 1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0

    return float(score)

# Rule 15
def rule_reward(state, action, choice):
    """
    Returns a float in [-1, 1] indicating how likely the choice is correct under the move/smelt rule.
    Positive values favor correct choices (source decreased by q, dest increased by q, no unrelated changes).
    """
    import re

    def parse_move_or_smelt(a):
        m = re.search(r'(?:move|smelt):\s*from\s*(\[[A-Za-z0-9]+\])\s*to\s*(\[[A-Za-z0-9]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns dict slot -> (name, qty)
        items = {}
        for m in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = m[0].strip()
            slot = f'[{m[1]}]'
            qty = int(m[2])
            items[slot] = (name, qty)
        return items

    parsed = parse_move_or_smelt(action)
    if not parsed:
        # not a move/smelt action -> this rule doesn't apply; return neutral 0.0
        return 0.0

    src_slot, dst_slot, q = parsed
    state_items = parse_items(state)
    choice_items = parse_items(choice)

    # helper to get qty (0 if missing)
    def qty_at(d, slot):
        return d.get(slot, (None, 0))[1]

    src_prev = qty_at(state_items, src_slot)
    src_new = qty_at(choice_items, src_slot)
    dst_prev = qty_at(state_items, dst_slot)
    dst_new = qty_at(choice_items, dst_slot)

    src_ok = (src_prev - src_new) == q
    dst_ok = (dst_new - dst_prev) == q

    # detect unrelated changes: any slot other than src_slot, dst_slot, or '[0]' whose qty changed
    unrelated_changed = False
    all_slots = set(state_items.keys()) | set(choice_items.keys())
    for s in all_slots:
        if s in (src_slot, dst_slot, '[0]'):
            continue
        if qty_at(state_items, s) != qty_at(choice_items, s):
            unrelated_changed = True
            break

    # scoring logic
    if src_ok and dst_ok and not unrelated_changed:
        score = 1.0
    elif not src_ok and not dst_ok:
        # completely wrong transfer
        score = -1.0
    else:
        # one side correct (src or dst) but not both, or other minor issue
        score = -0.6

    # penalize unrelated changes strongly
    if unrelated_changed:
        score -= 0.8

    # clamp to [-1, 1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0

    return float(score)

# Rule 16
def rule_reward(state, action, choice):
    import re

    def parse_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z0-9]+\])\s*to\s*(\[[A-Z0-9]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, 0
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns dict slot -> (name, qty)
        items = {}
        # Matches lines like: - item_name [SLOT] quantity N
        for name, slot, qty in re.findall(r'-\s+([^\[]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            items[f'[{slot}]'] = (name.strip(), int(qty))
        return items

    # Only apply when this is a craft state
    if "Craft an item" not in state:
        return 0.0

    src_slot, dst_slot, qty = parse_action(action)
    if src_slot is None:
        return 0.0

    state_items = parse_items(state)
    choice_items = parse_items(choice)

    # Move correctness check
    moved = state_items.get(src_slot)
    if moved is None:
        # no item to move from source in state -> invalid
        return -1.0
    moved_name, src_prev = moved
    # In choice, source may be missing (treated as qty 0) or present
    choice_src = choice_items.get(src_slot, (moved_name, 0))
    src_new_name, src_new = choice_src
    # Destination previous qty (0 if absent)
    dst_prev = state_items.get(dst_slot, (moved_name, 0))[1]
    choice_dst = choice_items.get(dst_slot, (moved_name, 0))
    dst_new_name, dst_new = choice_dst

    # Check same item name appears at dst and src (or allowed to be absent if qty 0)
    move_src_ok = (src_new_name in (moved_name, None) or src_slot not in choice_items) and (src_prev - src_new) == qty
    move_dst_ok = (dst_new_name == moved_name) and (dst_new - dst_prev) == qty

    if not (move_src_ok and move_dst_ok):
        # Move not implemented correctly -> penalize strongly
        return -1.0

    # Count additional changes outside of src and dst
    extra_changes = 0
    considered_slots = set(state_items.keys()) | set(choice_items.keys())
    for s in considered_slots:
        if s == src_slot or s == dst_slot:
            continue
        s_state = state_items.get(s)
        s_choice = choice_items.get(s)
        if s_state != s_choice:
            extra_changes += 1

    # Reward if there is at least one extra change (craft side-effect)
    if extra_changes > 0:
        return 1.0
    else:
        return -1.0

# Rule 17
def rule_reward(state, action, choice):
    import re
    from collections import defaultdict

    # parse action like: move: from [I13] to [A3] with quantity 1
    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[^\]]+\])\s*to\s*(\[[^\]]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None
        return m.group(1), m.group(2), int(m.group(3))

    # parse inventory lists: return list of (name, slot, qty)
    def parse_items(s):
        items = []
        # match lines like: - white_wool [A1] quantity 1
        for m in re.findall(r'-\s+([^\[]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = m[0].strip()
            slot = f'[{m[1]}]'
            qty = int(m[2])
            items.append((name, slot, qty))
        return items

    parsed = parse_move_action(action)
    if not parsed:
        # not a move action: neutral score
        return 0.0
    src_slot, dst_slot, q = parsed

    state_items = parse_items(state)
    choice_items = parse_items(choice)

    # build slot->(name,qty) maps
    s_slot_map = {slot: (name, qty) for (name, slot, qty) in state_items}
    c_slot_map = {slot: (name, qty) for (name, slot, qty) in choice_items}

    # moved item must exist at src in state
    if src_slot not in s_slot_map:
        return -1.0

    moved_name, src_prev_qty = s_slot_map[src_slot]
    dst_prev_name, dst_prev_qty = c_slot_map.get(dst_slot, (None, 0))
    # but dst_prev as in state:
    dst_prev_in_state = s_slot_map.get(dst_slot, (None, 0))[1]

    # get new quantities and names for src/dst in choice (default name from state if absent)
    src_new_name, src_new_qty = c_slot_map.get(src_slot, (moved_name, 0))
    dst_new_name, dst_new_qty = c_slot_map.get(dst_slot, (moved_name, 0))

    checks = 0
    total_checks = 3

    # Check 1: destination contains same moved_name and increased by q relative to state's dst
    # Use state's dst_prev_qty (0 if absent)
    if dst_new_name == moved_name and (dst_new_qty - dst_prev_in_state) == q:
        checks += 1

    # Check 2: source decreased by q and still same name or absent
    # Allow src to disappear if qty becomes 0
    if (src_new_name in (moved_name, None)) and (src_prev_qty - src_new_qty) == q:
        checks += 1

    # Check 3: totals of other item names (excluding moved_name) must remain same,
    # ignoring any slot entries whose slot is exactly '[0]' (we don't penalize changes in slot [0])
    def totals(items):
        d = defaultdict(int)
        for name, slot, qty in items:
            if slot == '[0]':
                continue
            d[name] += qty
        return d

    s_tot = totals(state_items)
    c_tot = totals(choice_items)

    # subtract moved item totals since those are allowed to change
    s_tot.pop(moved_name, None)
    c_tot.pop(moved_name, None)

    if s_tot == c_tot:
        checks += 1

    # Map checks (0..3) into [-1,1]
    score = (checks / total_checks) * 2.0 - 1.0
    return float(score)

# Rule 18
def rule_reward(state, action, choice):
    import re
    # Apply rule only for craft tasks with a move action
    if "Craft an item" not in state:
        return 0.0

    m = re.search(r'move:\s*from\s*(\[[A-Z0-9]+\])\s*to\s*(\[[A-Z0-9]+\])\s*with\s*quantity\s*(\d+)', action)
    if not m:
        return 0.0
    src_slot, dst_slot, qty = m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns list of (name, slot, qty) where slot like [I17], [A1], [0]
        items = []
        # Match lines like "- item_name [SLOT] quantity N"
        for name, slot, q in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            items.append((name.strip(), f'[{slot}]', int(q)))
        return items

    q_items = parse_items(state)
    c_items = parse_items(choice)

    slot_q = {slot: (name, qty) for (name, slot, qty) in q_items}
    slot_c = {slot: (name, qty) for (name, slot, qty) in c_items}

    # Check move correctness
    if src_slot not in slot_q:
        # source missing in initial state -> penalize
        move_ok = False
    else:
        moved_name, src_prev = slot_q[src_slot]
        dst_prev = slot_q.get(dst_slot, (moved_name, 0))[1]
        src_new_name, src_new = slot_c.get(src_slot, (moved_name, 0))
        dst_new_name, dst_new = slot_c.get(dst_slot, (moved_name, 0))

        # Require names at source/destination to match moved item (or be missing at source when reduced to 0)
        src_name_ok = (src_new_name == moved_name) or (src_slot not in slot_c) or (src_new == 0)
        dst_name_ok = (dst_new_name == moved_name)

        src_qty_ok = (src_prev - src_new) == qty
        dst_qty_ok = (dst_new - dst_prev) == qty

        move_ok = (src_name_ok and dst_name_ok and src_qty_ok and dst_qty_ok)

    # Check slot [0] updated/added to represent crafted output
    state_slot0 = slot_q.get('[0]')  # may be None
    choice_slot0 = slot_c.get('[0]')  # may be None

    # Good if choice has slot [0] and it is newly present or changed (name changed or quantity increased)
    slot0_ok = False
    if choice_slot0 is not None:
        if state_slot0 is None:
            slot0_ok = True
        else:
            # different crafted item name or increased qty
            if choice_slot0[0] != state_slot0[0]:
                slot0_ok = True
            elif choice_slot0[1] > state_slot0[1]:
                slot0_ok = True

    # Scoring: both checks needed for strong positive. Penalize failures.
    score = 0.0
    if move_ok:
        score += 0.6
    else:
        score -= 0.6

    if slot0_ok:
        score += 0.4
    else:
        score -= 0.4

    # Clamp to [-1, 1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return float(score)

# Rule 19
def rule_reward(state, action, choice):
    import re

    # parse action of form: move: from [I8] to [A1] with quantity 1
    m = re.search(r'move:\s*from\s*(\[[A-Z0-9]+\])\s*to\s*(\[[A-Z0-9]+\])\s*with\s*quantity\s*(\d+)', action)
    if not m:
        return 0.0  # rule not applicable

    src_slot = m.group(1)
    dst_slot = m.group(2)
    qty = int(m.group(3))

    # parse inventory lines like: - diamond_block [I8] quantity 1
    def parse_items(text):
        items = []
        for name, slot, q in re.findall(r'-\s+([^\[]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', text):
            items.append((name.strip(), f'[{slot}]', int(q)))
        return items

    s_items = parse_items(state)
    c_items = parse_items(choice)

    s_map = {slot: (name, q) for (name, slot, q) in s_items}
    c_map = {slot: (name, q) for (name, slot, q) in c_items}

    # source must exist in original state
    if src_slot not in s_map:
        # can't verify move if source absent; treat as incorrect
        return -1.0

    moved_name, src_prev_q = s_map[src_slot]
    # destination previous quantity (might be 0 if missing)
    dst_prev_name, dst_prev_q = c_prev = s_map.get(dst_slot, (None, 0))

    # choice destination entry must exist and must contain moved_name
    if dst_slot not in c_map:
        # destination missing in choice is allowed only if moved qty is 0 (not our case)
        return -1.0

    dst_new_name, dst_new_q = c_map[dst_slot]
    # identity check: destination must contain the same item name as moved item
    if dst_new_name != moved_name:
        return -1.0

    # source in choice (may be missing => qty 0)
    src_new_name, src_new_q = c_map.get(src_slot, (moved_name, 0))

    # Quantity checks
    dst_increase_ok = (dst_new_q - (s_map.get(dst_slot, (None, 0))[1])) == qty
    src_decrease_ok = ((s_map[src_slot][1] - src_new_q) == qty)

    # Unrelated changes: for all slots except src_slot, dst_slot, and slot [0],
    # the multiset of (slot, name, qty) should be unchanged.
    def key_items(items):
        d = {}
        for name, slot, q in items:
            if slot in (src_slot, dst_slot) or slot == '[0]':
                continue
            d[(slot, name)] = q
        return d

    s_other = key_items(s_items)
    c_other = key_items(c_items)
    unrelated_unchanged = (s_other == c_other)

    # scoring:
    # if identity (dst contains moved_name) failed, we already returned -1
    # now reward partial/full correctness
    score = 0.0
    if dst_increase_ok:
        score += 0.4
    if src_decrease_ok:
        score += 0.4
    if unrelated_unchanged:
        score += 0.2

    # normalize to range [-1, 1]; since score in [0,1], map directly
    # but if neither primary checks true, penalize
    if score == 0.0:
        return -1.0
    return round(score, 3)

# Rule 20
def rule_reward(state, action, choice):
    import re

    def parse_action(a):
        # expects form: move: from [0] to [I2] with quantity 3
        m = re.search(r'move:\s*from\s*(\[[^\]]+\])\s*to\s*(\[[^\]]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, None
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns dict slot -> (name, qty)
        items = {}
        for m in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = m[0].strip()
            slot = f'[{m[1]}]'
            qty = int(m[2])
            items[slot] = (name, qty)
        return items

    src_slot, dst_slot, qty = parse_action(action)
    if not src_slot:
        return 0.0  # rule not applicable

    s_items = parse_items(state)
    c_items = parse_items(choice)

    # Must have a moved item at source in state
    if src_slot not in s_items:
        return -1.0

    moved_name, src_prev = s_items[src_slot]

    # Check 1: destination in choice must contain moved_name and increased by qty
    dst_prev = s_items.get(dst_slot, (moved_name, 0))[1]
    dst_new_entry = c_items.get(dst_slot)
    if not dst_new_entry:
        return -1.0
    dst_new_name, dst_new_qty = dst_new_entry
    if dst_new_name != moved_name:
        return -1.0
    if (dst_new_qty - dst_prev) != qty:
        return -1.0

    score = 0.0
    score += 0.4  # reward for correct destination name & qty change

    # Check 2: source [0] should be removed (in these examples it is removed)
    if src_slot in c_items:
        # if present, ensure it decreased by qty (but prefer it absent)
        c_name, c_qty = c_items[src_slot]
        if c_name != moved_name:
            return -1.0
        if (src_prev - c_qty) == qty:
            score += 0.1
        else:
            return -1.0
    else:
        score += 0.3

    # Check 3: no non-I slots (except source which we already handled) should remain in choice
    for slot in c_items:
        # slot format like [A1], [I2], [0]
        inner = slot.strip('[]')
        # allow only slots that start with 'I' (I...) in the final inventory
        if not inner.startswith('I'):
            # if it's the source it was already checked; otherwise penalize heavily
            return -1.0

    score += 0.2

    # Check 4: other I* items (excluding dst_slot) should be unchanged
    for slot, (name, qty_s) in s_items.items():
        inner = slot.strip('[]')
        if not inner.startswith('I'):
            continue
        if slot == dst_slot:
            continue
        # must exist in choice with same name and qty
        if slot not in c_items:
            return -1.0
        name_c, qty_c = c_items[slot]
        if name_c != name or qty_c != qty_s:
            return -1.0

    score = min(1.0, score)
    # map to [-1,1]: positive if passing checks
    return score

# Rule 21
def rule_reward(state, action, choice):
    import re

    def parse_action(a):
        # expects: move: from [I5] to [A1] with quantity 3
        m = re.search(r'move:\s*from\s*(\[[A-Z0-9]+\])\s*to\s*(\[[A-Z0-9]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, None
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns list of (name, slot, qty) where slot like [I17], [A1], [0]
        items = []
        for m in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = m[0].strip()
            slot = f'[{m[1]}]'
            qty = int(m[2])
            items.append((name, slot, qty))
        return items

    # parse action
    src_slot, dst_slot, qty = parse_action(action)
    if src_slot is None:
        # action not in expected move format -> don't apply this rule
        return 0.0

    # Restrict rule to intended pattern: inventory -> auxiliary (e.g., [I...] -> [A...])
    if not (src_slot.startswith('[I') and dst_slot.startswith('[A')):
        return 0.0

    state_items = parse_items(state)
    choice_items = parse_items(choice)

    # Build maps slot -> (name, qty)
    s_map = {slot: (name, q) for (name, slot, q) in state_items}
    c_map = {slot: (name, q) for (name, slot, q) in choice_items}

    # If source missing in original, be conservative and don't apply this rule
    if src_slot not in s_map:
        return 0.0

    moved_name, src_prev_q = s_map[src_slot]
    dst_prev_name, dst_prev_q = s_map.get(dst_slot, (None, 0))

    # If the action tries to move more than exists, do not apply (invalid action)
    if qty > src_prev_q:
        return 0.0

    # 1) A-slot preservation detection (only consider A-slots present in original state,
    #    and exclude the destination slot of this action — we only care about other pre-existing A-slots).
    a_slots = [(name, slot, q) for (name, slot, q) in state_items if slot.startswith('[A') and slot != dst_slot]
    a_violations = []
    for name, a_slot, a_q in a_slots:
        if a_slot in c_map:
            c_name, c_q = c_map[a_slot]
            # suspicious if unchanged or increased with same name (could indicate not consumed)
            if c_name == name and c_q >= a_q:
                a_violations.append((a_slot, name, a_q, c_q))

    # 2) Destination correctness: must end up holding moved_name and increase by qty relative to previous dst quantity
    dst_choice = c_map.get(dst_slot)
    dest_ok = False
    if dst_choice is not None:
        dst_choice_name, dst_choice_q = dst_choice
        prev_q = dst_prev_q
        if dst_choice_name == moved_name and (dst_choice_q - prev_q) == qty:
            dest_ok = True

    # 3) Source decrement correctness: source slot must be decreased by qty or removed when consumed exactly
    src_choice = c_map.get(src_slot)
    src_ok = False
    if src_choice is None:
        # removed from choice; acceptable only if src_prev_q == qty
        if src_prev_q == qty:
            src_ok = True
    else:
        src_choice_name, src_choice_q = src_choice
        if src_choice_name == moved_name and (src_prev_q - src_choice_q) == qty:
            src_ok = True

    # 4) Unrelated items unchanged: compare totals for names excluding moved_name,
    #    excluding A-slot items, excluding src_slot and excluding the production/output slot [0].
    def totals(items):
        d = {}
        for name, slot, q in items:
            if slot == src_slot:
                continue
            if slot.startswith('[A'):
                continue
            if slot == '[0]':
                # explicitly ignore the crafted-output line
                continue
            if name == moved_name:
                continue
            d[name] = d.get(name, 0) + q
        return d

    s_tot = totals(state_items)
    c_tot = totals(choice_items)
    tot_ok = (s_tot == c_tot)

    # Conservative scoring policy:
    # - If all three core checks pass -> full positive
    # - If two pass -> strong positive
    # - If one pass -> small positive (we don't want to overly punish small differences)
    # - If none pass -> moderate negative, but if there are additional A-slot violations then slightly stronger negative
    passed = sum([1 for v in (dest_ok, src_ok, tot_ok) if v])

    if passed == 3:
        return 1.0
    if passed == 2:
        return 0.7
    if passed == 1:
        # small positive: partial correctness (avoid strong penalties for small mismatches)
        return 0.1

    # passed == 0: be conservative but penalize; consider A-slot preservation as additional evidence
    # If there are clear A-slot violations (pre-existing auxiliary slots that appear preserved), increase penalty
    if a_violations:
        # moderate negative, but not extreme to avoid false negatives
        return -0.6
    # no primary checks passed and no A-slot evidence -> moderate negative
    return -0.5

# Rule 22
def rule_reward(state, action, choice):
    import re

    def is_smelt_action(a):
        return re.search(r'smelt:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', a) is not None

    def parse_items(txt):
        items = {}
        # matches lines like "- cobblestone [A0] quantity 3"
        for name, slot, qty in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', txt):
            items['[' + str(slot) + ']'] = (name.strip(), int(qty))
        return items

    # Only consider explicit smelt actions
    m = re.search(r'smelt:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', action)
    if not m:
        return 0.0
    act_src_slot, act_dst_slot, act_qty = m.group(1), m.group(2), int(m.group(3))

    # Known conservative smelt mappings
    smelt_map = {
        'cobblestone': 'stone',
        'oak_wood': 'charcoal',
        'diamond_ore': 'diamond'
    }

    state_map = parse_items(state)
    choice_map = parse_items(choice)

    state_src_name, state_src_qty = state_map.get(act_src_slot, (None, 0))
    _choice_src_name, choice_src_qty = choice_map.get(act_src_slot, (None, 0))
    _state_dst_name, state_dst_qty = state_map.get(act_dst_slot, (None, 0))
    choice_dst_name, choice_dst_qty = choice_map.get(act_dst_slot, (None, 0))

    # Only trigger when the source slot in the initial state contains a known smeltable item
    if state_src_name not in smelt_map:
        return 0.0
    expected_src_name = state_src_name
    expected_dst_name = smelt_map[expected_src_name]

    # Expected deltas derived from the action quantity (conservative and slot-aware)
    expected_delta_src = act_qty
    expected_delta_dst = act_qty

    # Compute slot-specific deltas (only at the exact slots mentioned in the action)
    delta_src = max(0, int(state_src_qty) - int(choice_src_qty))
    delta_dst = max(0, int(choice_dst_qty) - int(state_dst_qty))

    # If neither source nor destination slot changed, treat as unrelated/no-evidence and do not penalize
    if delta_src == 0 and delta_dst == 0:
        return 0.0

    score = 0.0
    # Reward correct product appearing at the destination slot with the expected name and exact quantity
    if (choice_dst_name == expected_dst_name) and (delta_dst == expected_delta_dst):
        score += 0.7
    else:
        # Only penalize if the destination slot actually changed or the observed name differs from the prior name
        if delta_dst > 0 or (choice_dst_name != _state_dst_name and choice_dst_name is not None):
            score -= 0.7

    # Give a smaller positive reward if the source slot decreased by the expected amount; do not penalize if it didn't
    if delta_src == expected_delta_src:
        score += 0.3

    # Clamp score to [-1, 1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return float(score)

# Rule 23
def rule_reward(state, action, choice):
    import re

    # Only apply to smelt actions of the exact form used in examples
    m = re.search(r'smelt:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', action)
    if not m:
        return 0.0
    act_src_slot, act_dst_slot, _ = m.group(1), m.group(2), int(m.group(3))

    # Known conservative smelt mappings (source_name -> (dest_name, expected_src_decrease, expected_dst_increase))
    mappings = {
        'nether_quartz_ore': ('quartz', 1, 1),
        'nether_gold_ore': ('gold_ingot', 1, 1),
        'golden_helmet': ('gold_nugget', 1, 1),
    }

    # Parse simple item lists of the form "- NAME [SLOT] quantity N"
    def parse_items(txt):
        items = {}
        for name, slot, qty in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', txt):
            items['[' + str(slot) + ']'] = (name.strip(), int(qty))
        return items

    state_map = parse_items(state)
    choice_map = parse_items(choice)

    # Get names and quantities at the exact slots referenced by the action (default to None/0)
    state_src_name, state_src_qty = state_map.get(act_src_slot, (None, 0))
    choice_src_name, choice_src_qty = choice_map.get(act_src_slot, (None, 0))
    state_dst_name, state_dst_qty = state_map.get(act_dst_slot, (None, 0))
    choice_dst_name, choice_dst_qty = choice_map.get(act_dst_slot, (None, 0))

    # Only trigger if the source slot in the state is one of the known smeltable items
    if state_src_name not in mappings:
        return 0.0

    expected_dst_name, expected_delta_src, expected_delta_dst = mappings[state_src_name]

    # Compute slot-specific non-negative deltas (only consider increases for dest and decreases for src)
    delta_src = max(0, int(state_src_qty) - int(choice_src_qty))
    delta_dst = max(0, int(choice_dst_qty) - int(state_dst_qty))

    score = 0.0

    # Destination: require the product name at the destination slot and exact expected increase to reward;
    # otherwise penalize (we are conservative about incorrect products placed in the target slot).
    if choice_dst_name == expected_dst_name and delta_dst == expected_delta_dst:
        score += 0.7
    else:
        score -= 0.7

    # Source: reward (but do not penalize) if the source slot decreased by the expected amount
    if delta_src == expected_delta_src:
        score += 0.3

    # Clip to [-1, 1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0

    return float(score)

# Rule 24
def rule_reward(state, action, choice):
    import re

    # Only consider actions that exactly match the expected smelt syntax.
    m = re.search(r'smelt:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', action)
    if not m:
        return 0.0

    act_src_slot, act_dst_slot, act_qty = m.group(1), m.group(2), int(m.group(3))

    # Allowed smelt mappings (conservative set from examples). Add mappings here if needed.
    allowed_mappings = {
        'cobblestone': 'stone',
        'chorus_fruit': 'popped_chorus_fruit',
    }

    # Parse inventories: only collect items that include a slot, name, and quantity in the expected format.
    def parse_items(txt):
        items = {}
        for name, slot, qty in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', txt):
            items['[' + str(slot) + ']'] = (name.strip(), int(qty))
        return items

    state_map = parse_items(state)
    choice_map = parse_items(choice)

    # Get names/quantities at the exact slots referenced by the action (default to None/0).
    state_src_name, state_src_qty = state_map.get(act_src_slot, (None, 0))
    choice_src_name, choice_src_qty = choice_map.get(act_src_slot, (None, 0))
    state_dst_name, state_dst_qty = state_map.get(act_dst_slot, (None, 0))
    choice_dst_name, choice_dst_qty = choice_map.get(act_dst_slot, (None, 0))

    # Require that the source slot in the prior state contains one of the allowed source items.
    if state_src_name not in allowed_mappings:
        return 0.0

    expected_dst_name = allowed_mappings[state_src_name]
    expected_delta_src = act_qty
    expected_delta_dst = act_qty

    # Compute slot-specific deltas only for the two referenced slots to avoid penalizing unrelated changes.
    delta_src = max(0, int(state_src_qty) - int(choice_src_qty))
    delta_dst = max(0, int(choice_dst_qty) - int(state_dst_qty))

    score = 0.0
    # Reward correctly named product increasing by exactly the action quantity at the destination slot.
    if (choice_dst_name == expected_dst_name) and (delta_dst == expected_delta_dst):
        score += 0.7
    else:
        # Only penalize if the destination did not show the expected product increase.
        score -= 0.7

    # Soft reward exact source decrease at the source slot; do not penalize if source change is different.
    if delta_src == expected_delta_src:
        score += 0.3

    # Clamp to [-1, 1] and return a float.
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return float(score)

# Rule 25
def rule_reward(state, action, choice):
    """
    Returns a float in [-1,1]. Positive if the choice follows the rule (likely correct),
    negative if it violates the rule (likely incorrect).
    Applies when action is 'smelt' or 'move' (to an A-slot). Otherwise returns 0.0 (rule not applicable).

    Heuristics:
      - For smelt: destination item name must differ from source item name and quantities should adjust.
      - For move -> A-slot: moved item must appear at A-slot AND there must be a produced output:
          either slot '[0]' quantity increases OR at least one new item name appears in choice
          that wasn't present in the state (excluding the moved item).
    """
    import re

    def parse_items(s):
        # returns dict slot -> (name, qty) and a name->total_qty map
        slot_map = {}
        name_totals = {}
        for m in re.findall(r'-\s+([^\[]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = m[0].strip()
            slot = f'[{m[1]}]'
            qty = int(m[2])
            slot_map[slot] = (name, qty)
            name_totals[name] = name_totals.get(name, 0) + qty
        return slot_map, name_totals

    def parse_smelt(a):
        m = re.search(r'smelt:\s*from\s*(\[[^\]]+\])\s*to\s*(\[[^\]]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None
        return (m.group(1), m.group(2), int(m.group(3)))

    def parse_move(a):
        m = re.search(r'move:\s*from\s*(\[[^\]]+\])\s*to\s*(\[[^\]]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None
        return (m.group(1), m.group(2), int(m.group(3)))

    state_slots, state_totals = parse_items(state)
    choice_slots, choice_totals = parse_items(choice)

    sm = parse_smelt(action)
    mv = parse_move(action)

    # Helper to get slot info (name, qty) or (None,0)
    def slot_info(slots, slot):
        return slots.get(slot, (None, 0))

    # Apply smelt rule
    if sm:
        src_slot, dst_slot, qty = sm
        src_name, src_prev = slot_info(state_slots, src_slot)
        # If source missing in state, cannot apply rule robustly
        if src_name is None:
            return 0.0
        # Destination in choice must exist and have different name
        dst_name_choice, dst_choice_qty = slot_info(choice_slots, dst_slot)
        src_name_choice, src_choice_qty = slot_info(choice_slots, src_slot)
        # If dst became same name as source -> wrong
        if dst_name_choice == src_name:
            return -1.0
        # If destination name is different, reward; but check quantities for stronger signal
        # Check expected quantity changes
        src_decrease = src_prev - src_choice_qty
        dst_prev = slot_info(state_slots, dst_slot)[1]
        dst_increase = dst_choice_qty - dst_prev
        score = 0.0
        if dst_name_choice is not None and dst_name_choice != src_name:
            score += 0.6  # name changed -> good
        # check quantity consistency
        if src_decrease == qty and dst_increase == qty:
            score += 0.4
        elif (src_decrease >= qty and dst_increase >= 0) or (src_decrease == qty or dst_increase == qty):
            score += 0.1
        return max(-1.0, min(1.0, score * 1.0))  # map to [-1,1], mostly positive if conditions met

    # Apply move -> A-slot rule
    if mv:
        src_slot, dst_slot, qty = mv
        # Only apply the "produced output" expectation when destination is an A-slot
        if not re.match(r'\[A', dst_slot):
            return 0.0
        src_name, src_prev = slot_info(state_slots, src_slot)
        if src_name is None:
            return 0.0
        # Verify moved item appears at destination in choice
        dst_name_choice, dst_choice_qty = slot_info(choice_slots, dst_slot)
        src_name_choice, src_choice_qty = slot_info(choice_slots, src_slot)
        moved_ok = False
        if dst_name_choice == src_name and (dst_choice_qty - slot_info(state_slots, dst_slot)[1]) == qty:
            # destination shows same item and increased by qty
            if (src_prev - src_choice_qty) == qty or src_choice_qty == 0:
                moved_ok = True
        # If moved not ok, penalize
        if not moved_ok:
            return -1.0
        # Now check for produced output:
        # Condition A: slot [0] quantity increased compared to state
        zero_prev = slot_info(state_slots, '[0]')[1]
        zero_choice = slot_info(choice_slots, '[0]')[1]
        zero_increased = zero_choice > zero_prev
        # Condition B: there exists a new item name in choice not present in state (excluding moved item)
        new_name_found = False
        for name in choice_totals:
            if name == src_name:
                continue
            if name not in state_totals:
                new_name_found = True
                break
            # also if total quantity increased for some name (excluding moved item)
            if choice_totals.get(name, 0) > state_totals.get(name, 0):
                new_name_found = True
                break
        if zero_increased or new_name_found:
            return 1.0
        else:
            # moved ingredient present but no produced output -> likely incorrect
            return -1.0

    # If action is neither applicable, don't apply this rule
    return 0.0

# Rule 26
def rule_reward(state, action, choice):
    """
    Return a float in [-1, 1] indicating how likely the choice is correct for
    move-into-crafting-grid actions, according to the rule described above.
    """

    import re

    def parse_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns list of (name, slot, qty) with slot like [I17], [A1], [0]
        items = []
        for name, slot, qty in re.findall(r'-\s+([^\[]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            items.append((name.strip(), f'[{slot}]', int(qty)))
        return items

    parsed = parse_action(action)
    if not parsed:
        return 0.0  # Not applicable

    src_slot, dst_slot, qty = parsed

    state_items = parse_items(state)
    choice_items = parse_items(choice)

    # Build dicts for quick lookup: slot -> (name, qty)
    def slot_map(items):
        return {slot: (name, count) for (name, slot, count) in items}

    s_map = slot_map(state_items)
    c_map = slot_map(choice_items)

    # Must have the moved item in source in the state
    if src_slot not in s_map:
        return -1.0

    moved_name, src_prev = s_map[src_slot]
    dst_prev = s_map.get(dst_slot, (moved_name, 0))[1]  # destination may be empty in state

    # Find new counts in choice (if missing, treat as 0)
    src_new_name, src_new = c_map.get(src_slot, (moved_name, 0))
    dst_new_name, dst_new = c_map.get(dst_slot, (moved_name, 0))

    # Check move correctness: same item occupies dst and src adjustments by qty
    move_ok = False
    if dst_new_name == moved_name and (dst_new - dst_prev) == qty:
        # source decreased by qty (allow source to disappear)
        if src_new_name in (moved_name, None) and (src_prev - src_new) == qty:
            move_ok = True

    # If move failed, strongly penalize
    if not move_ok:
        return -1.0

    # Helper: totals of all items excluding a set of names and excluding slot [0] (output)
    def totals(items, exclude_names=None):
        exclude_names = set(exclude_names or ())
        d = {}
        for name, slot, count in items:
            if slot == '[0]':
                continue
            if name in exclude_names:
                continue
            d[name] = d.get(name, 0) + count
        return d

    # Known reversible/group recipes observed in the error examples:
    # - slime_block -> 9 slime_ball
    # - bamboo -> stick (1 per bamboo in these examples)
    # - (white_dye + gray_dye) -> light_gray_dye yields 2 per pair moved
    expected_output_name = None
    expected_output_increase = None

    mn = moved_name.lower()

    if mn == 'slime_block':
        expected_output_name = 'slime_ball'
        expected_output_increase = 9 * qty
    elif mn == 'bamboo':
        expected_output_name = 'stick'
        expected_output_increase = 1 * qty
    elif mn in ('gray_dye', 'white_dye'):
        # If the other dye is present anywhere in state (or in the destination after move),
        # expect light_gray_dye to be produced. We assume 2 per pair moved (as in examples).
        other = 'white_dye' if mn == 'gray_dye' else 'gray_dye'
        # check presence in state or destination (after move)
        has_other = any(name == other for name, slot, count in state_items)
        # if other dye is on the destination in choice, consider that too
        if has_other:
            expected_output_name = 'light_gray_dye'
            expected_output_increase = 2 * qty

    # Check unrelated items unchanged (ignore moved_name and ignore output slot)
    s_tot = totals(state_items, exclude_names={moved_name})
    c_tot = totals(choice_items, exclude_names={moved_name})
    if s_tot != c_tot:
        # unrelated items changed -> penalize
        return -1.0

    # If we don't have an expected output mapping, reward the correct pure-move case modestly
    if expected_output_name is None:
        # Move was valid and nothing else changed -> good partial score
        return 0.6

    # Now check output slot [0] in choice: it must contain expected_output_name with the expected increase
    # Compute prior output qty for that name in state (slot [0] may contain something else or be absent)
    prev_out_qty = 0
    # find output slot in state: if slot [0] exists and has expected_output_name, take its count
    for name, slot, count in state_items:
        if slot == '[0]' and name == expected_output_name:
            prev_out_qty = count
            break
    # find output in choice
    out_name = None
    out_qty = 0
    for name, slot, count in choice_items:
        if slot == '[0]':
            out_name = name
            out_qty = count
            break

    # The output slot must be the expected item and its qty must have increased by expected_output_increase
    if out_name == expected_output_name and (out_qty - prev_out_qty) == expected_output_increase:
        return 1.0
    else:
        # If output slot used a different item or wrong amount, penalize
        return -1.0

# Rule 27
def rule_reward(state, action, choice):
    """
    Returns a float in [-1, 1]. Positive values indicate the choice
    looks consistent with a move action according to the rule; negative
    values indicate likely incorrect choices.

    state, action, choice are strings with the same format as in the examples.
    """
    import re
    from collections import defaultdict

    def parse_action(a):
        # matches "move: from [I22] to [C1] with quantity 1"
        m = re.search(r'move:\s*from\s*(\[[^\]]+\])\s*to\s*(\[[^\]]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns dict slot -> (name, qty) and name->total_qty
        slot_map = {}
        name_totals = defaultdict(int)
        # We look for lines like "- jungle_planks [A1] quantity 1" or "- sticky_piston [0] quantity 1"
        for name, slot, qty in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = name.strip()
            slot_lbl = f'[{slot}]'
            qty = int(qty)
            slot_map[slot_lbl] = (name, qty)
            name_totals[name] += qty
        return slot_map, dict(name_totals)

    pa = parse_action(action)
    if not pa:
        # not a move action we don't evaluate here
        return 0.0
    src_slot, dst_slot, qty = pa

    s_slots, s_totals = parse_items(state)
    c_slots, c_totals = parse_items(choice)

    # Find moved item name in source slot of state
    if src_slot not in s_slots:
        # source slot had no item in state -> invalid action
        return -1.0
    moved_name, src_prev = s_slots[src_slot]

    # Get quantities in candidate (0 if absent)
    c_src_name, c_src_qty = c_slots.get(src_slot, (None, 0))
    c_dst_name, c_dst_qty = c_slots.get(dst_slot, (None, 0))
    s_dst_name, s_dst_qty = s_slots.get(dst_slot, (None, 0))

    score = 0.0

    # 1) Source decreased by exactly qty for the same item
    src_decreased = False
    if c_src_name in (moved_name, None) and (src_prev - c_src_qty) == qty:
        src_decreased = True
        score += 0.45  # reward for correct source decrement
    else:
        # heavy penalty if source not decreased properly
        score -= 0.6

    # 2) Destination increased by exactly qty and contains same item
    dst_increased = False
    # Destination in state may have been empty or had some item. We require the destination in choice to contain moved_name.
    # Accept also if destination previously had moved_name (stacking).
    prev_dst_qty = s_dst_qty if s_dst_name == moved_name else 0
    if c_dst_name == moved_name and (c_dst_qty - prev_dst_qty) == qty:
        dst_increased = True
        score += 0.45  # reward for correct destination increment
    else:
        score -= 0.6

    # 3) Penalize any changes to other item names (total counts), except slot [0] is exempt
    # Build totals for names excluding moved_name and excluding items only in slot [0].
    def totals_excluding_moved_and_output(slot_map, name_totals):
        totals = {}
        for name, tot in name_totals.items():
            if name == moved_name:
                continue
            # Determine whether all of this item's presence is only in slot [0] in the corresponding slot_map.
            # If an item appears in non-[0] slot(s), we include it in totals comparison.
            has_non_output = False
            for slot, (nm, qtyv) in slot_map.items():
                if nm == name and slot != '[0]':
                    has_non_output = True
                    break
            if has_non_output:
                totals[name] = tot
            # otherwise we ignore items that exist only at [0]
        return totals

    s_other = totals_excluding_moved_and_output(s_slots, s_totals)
    c_other = totals_excluding_moved_and_output(c_slots, c_totals)

    # If any other item total changed, penalize
    other_changed = False
    # Compare keys union
    keys = set(s_other.keys()) | set(c_other.keys())
    for k in keys:
        if s_other.get(k, 0) != c_other.get(k, 0):
            other_changed = True
            break
    if other_changed:
        score -= 0.25
    else:
        score += 0.05  # small bonus for minimal unrelated changes

    # Clamp score to [-1, 1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return float(score)

# Rule 28
def rule_reward(state, action, choice):
    import re

    # parse move action
    m = re.search(r'move:\s*from\s*(\[[^\]]+\])\s*to\s*(\[[^\]]+\])\s*with\s*quantity\s*(\d+)', action)
    if not m:
        # This rule only applies to move actions
        return 0.0

    src_slot, dst_slot, q = m.group(1), m.group(2), int(m.group(3))

    # helper to parse inventory lists into list of tuples (name, slot, qty)
    def parse_items(block):
        items = []
        for name, slot, qty in re.findall(r'-\s+([^\[]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', block):
            items.append((name.strip(), f'[{slot}]', int(qty)))
        return items

    q_items = parse_items(state)
    c_items = parse_items(choice)

    # build maps: slot -> (name, qty); name -> total_qty (across all slots)
    q_slot = {slot: (name, qty) for name, slot, qty in q_items}
    c_slot = {slot: (name, qty) for name, slot, qty in c_items}

    def totals(items):
        d = {}
        for name, slot, qty in items:
            d[name] = d.get(name, 0) + qty
        return d

    q_tot = totals(q_items)
    c_tot = totals(c_items)

    # moved item must exist in original src slot
    if src_slot not in q_slot:
        return -1.0  # invalid move: nothing to move

    moved_name, src_prev = q_slot[src_slot]
    # get dest previous qty and name (may not exist)
    dst_prev_name, dst_prev = c_slot.get(dst_slot, q_slot.get(dst_slot, (moved_name, 0)))
    # In many correct cases destination may be empty in the original state; handle missing.

    # find new quantities for source and destination in candidate (0 if missing)
    src_new_name, src_new = c_slot.get(src_slot, (moved_name, 0))
    dst_new_name, dst_new = c_slot.get(dst_slot, (dst_prev_name, 0))

    # Check 1: move correctness (source decreased by q, destination increased by q, item name preserved)
    move_ok = False
    try:
        src_delta = src_prev - src_new
        dst_delta = dst_new - dst_prev
        # validate that the item moved is the same name at src and destination after move
        dst_name_matches = (dst_new_name == moved_name) or (dst_prev_name == moved_name) or (dst_prev_name is None)
        src_name_ok = (src_new_name == moved_name) or (src_new_name == None)
        move_ok = (src_delta == q) and (dst_delta == q) and dst_name_matches and src_name_ok
    except Exception:
        move_ok = False

    # Identify crafted target name T from the state first line "Craft an item of type: X"
    tmatch = re.search(r'Craft an item of type:\s*([^\n\r]+)', state)
    target_name = tmatch.group(1).strip() if tmatch else None

    # Determine change in crafted target at slot [0]
    q_target_prev = 0
    c_target_new = 0
    # find any slot '[0]' in parsed items
    for name, slot, qty in q_items:
        if slot == '[0]':
            q_target_prev = qty
            break
    for name, slot, qty in c_items:
        if slot == '[0]':
            if target_name and name.strip() == target_name:
                c_target_new = qty
            else:
                # slot [0] contains something else (unexpected), treat as change to other item
                pass

    craft_increase = 0
    if target_name:
        craft_increase = max(0, c_target_new - q_target_prev)

    # Compute total decreases across all items excluding the moved item
    total_decrease_excl_moved = 0
    unrelated_increase_exists = False
    for name, slot, qty in q_items:
        # skip moved item contributions; we will handle moved item separately
        if name == moved_name:
            continue
        c_qty = c_tot.get(name, 0)
        if c_qty < qty:
            total_decrease_excl_moved += (qty - c_qty)
    # Check for any increases in names other than moved_name and target_name
    for name, new_qty in c_tot.items():
        if name == moved_name:
            continue
        if name == target_name:
            continue
        prev_qty = q_tot.get(name, 0)
        if new_qty > prev_qty:
            unrelated_increase_exists = True
            break

    # Check 2 and 3: unrelated changes validity
    # - If there's any unrelated increase (other than dst moved item or target slot), fail.
    # - If craft_increase > 0, require total_decrease_excl_moved >= craft_increase
    # - If craft_increase == 0, require total_decrease_excl_moved == 0 (no unrelated decreases)
    if unrelated_increase_exists:
        unrelated_ok = False
    else:
        if craft_increase > 0:
            unrelated_ok = (total_decrease_excl_moved >= craft_increase)
        else:
            # no crafting; no other item should change aside from moved item
            unrelated_ok = (total_decrease_excl_moved == 0)

    # craft_ok: if candidate added crafted items, they must be justified
    craft_ok = True
    if craft_increase > 0:
        craft_ok = (total_decrease_excl_moved >= craft_increase)

    # Combine checks into a score in [-1,1]
    checks = 0
    checks += 1 if move_ok else 0
    checks += 1 if unrelated_ok else 0
    checks += 1 if craft_ok else 0

    # Map checks 0..3 to -1..1 linearly
    score = -1.0 + (checks / 3.0) * 2.0
    # small safety clamp
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return score

# Rule 29
def rule_reward(state, action, choice):
    import re

    def parse_move_action(a):
        # returns src_slot, dst_slot, qty or (None,None,0) if not a move action
        m = re.search(r'move:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, 0
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns dict slot -> (name, qty)
        items = {}
        # matches lines like: - quartz [I4] quantity 1
        for m in re.finditer(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = m.group(1).strip()
            slot = f'[{m.group(2)}]'
            qty = int(m.group(3))
            items[slot] = (name, qty)
        return items

    src_slot, dst_slot, qty = parse_move_action(action)
    # Only apply this rule for move actions
    if src_slot is None:
        return 0.0

    q_items = parse_items(state)
    c_items = parse_items(choice)

    # Source must exist in the current state
    if src_slot not in q_items:
        return -1.0

    moved_name, src_prev = q_items[src_slot]
    dst_prev = c_items.get(dst_slot, (None, 0))[1] if dst_slot in c_items else q_items.get(dst_slot, (None, 0))[1] if dst_slot in q_items else 0
    # But better to get previous dst quantity from original state (0 if absent)
    dst_prev = q_items.get(dst_slot, (None, 0))[1]

    # Get new values from the choice (treat absent as (None,0))
    src_new_name, src_new_qty = c_items.get(src_slot, (None, 0))
    dst_new_name, dst_new_qty = c_items.get(dst_slot, (None, 0))

    # Check source decreased correctly
    src_ok = False
    if src_new_name in (moved_name, None):
        if (src_prev - src_new_qty) == qty:
            src_ok = True

    # Check destination increased correctly and holds the same item name
    dst_ok = False
    if dst_new_name == moved_name:
        # previous dst quantity is dst_prev (0 if absent)
        if (dst_new_qty - dst_prev) == qty:
            dst_ok = True

    # Check no other slots changed
    violation = False
    # For every slot present in the original state except src and dst: must be identical in choice
    for slot, (name, cnt) in q_items.items():
        if slot in (src_slot, dst_slot):
            continue
        c = c_items.get(slot)
        if c is None:
            violation = True
            break
        if c[0] != name or c[1] != cnt:
            violation = True
            break

    # For every slot present in the choice except src and dst: it must either have been present before unchanged, or be the dst slot
    if not violation:
        for slot, (name, cnt) in c_items.items():
            if slot in (src_slot, dst_slot):
                continue
            if slot not in q_items:
                # new slot introduced that's not the destination -> violation
                violation = True
                break
            # if present before we've already checked equality above

    # Final scoring
    if src_ok and dst_ok and not violation:
        return 1.0  # perfect match
    if (src_ok or dst_ok) and not violation:
        return 0.0  # partially correct (one side correct) but not fully
    # Otherwise clearly incorrect (changed unrelated slots, wrong amounts, or new unrelated items)
    return -1.0

# Rule 30
def rule_reward(state, action, choice):
    import re

    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[^\]]+\])\s*to\s*(\[[^\]]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, 0
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns dict slot -> (name, qty)
        items = {}
        for name, slot, qty in re.findall(r'-\s+([^\[]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            slot_label = f'[{slot}]'
            items[slot_label] = (name.strip(), int(qty))
        return items

    src_slot, dst_slot, qty = parse_move_action(action)
    # If not a move action with expected format, rule not applicable -> neutral 0.0
    if not src_slot:
        return 0.0

    s_items = parse_items(state)
    c_items = parse_items(choice)

    # Moved item must exist in source in the original state
    if src_slot not in s_items:
        return -1.0

    moved_name, src_prev_qty = s_items[src_slot]
    # Destination previous qty and name (if exists)
    dst_prev = c_prev = None
    if dst_slot in s_items:
        dst_prev_name, dst_prev_qty = s_items[dst_slot]
    else:
        dst_prev_name, dst_prev_qty = None, 0

    # Destination in choice must exist with moved_name
    if dst_slot not in c_items:
        return -1.0
    dst_choice_name, dst_choice_qty = c_items[dst_slot]
    if dst_choice_name != moved_name:
        return -1.0

    # Source in choice may be missing (treated as qty 0) or present with same name
    if src_slot in c_items:
        src_choice_name, src_choice_qty = c_items[src_slot]
    else:
        src_choice_name, src_choice_qty = None, 0

    # Check 1: destination increased by exactly qty
    check_dst = (dst_choice_qty - dst_prev_qty) == qty and dst_choice_name == moved_name

    # Check 2: source decreased by exactly qty and name consistent (or removed)
    src_name_ok = (src_choice_name is None) or (src_choice_name == moved_name)
    check_src = src_name_ok and (src_prev_qty - src_choice_qty) == qty

    # Check 3: no changes to any other slots (same name and qty)
    other_ok = True
    for slot, (name, q) in s_items.items():
        if slot == src_slot or slot == dst_slot:
            continue
        if slot not in c_items:
            other_ok = False
            break
        c_name, c_q = c_items[slot]
        if c_name != name or c_q != q:
            other_ok = False
            break
    # also ensure choice does not introduce extra unrelated slots
    for slot in c_items:
        if slot == src_slot or slot == dst_slot:
            continue
        if slot not in s_items:
            other_ok = False
            break

    # Check 4: no creation or modification of special output slot "[0]"
    zero_slot_changed = False
    if '[0]' in s_items:
        s0_name, s0_q = s_items['[0]']
    else:
        s0_name, s0_q = None, 0
    if '[0]' in c_items:
        c0_name, c0_q = c_items['[0]']
    else:
        c0_name, c0_q = None, 0
    if (s0_name != c0_name) or (s0_q != c0_q):
        zero_slot_changed = True

    # Compose checks
    checks_passed = 0
    checks_passed += 1 if check_dst else 0
    checks_passed += 1 if check_src else 0
    checks_passed += 1 if other_ok else 0
    checks_passed += 1 if not zero_slot_changed else 0

    # Map 0..4 -> -1..1 linearly
    score = (checks_passed / 4.0) * 2.0 - 1.0
    return float(score)

