# WMQA Improved Rules
# Improved from: transition_mcq/rules_plancraft_llama3-2-1b-instruct_30pct_new.py
# Dev unit-weight improvement vs original: +1.71%
# Dev unit-weight accuracy (improved rules): 82.31%
# Dev weighted accuracy (learned on dev): 86.99%
# Test baseline accuracy: 75.43%
# Test weighted accuracy: 83.39%
# Test weighted improvement: +7.96%

# Rule 1
def rule_reward(state, action, choice):
    import re

    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[^\]]+\])\s*to\s*(\[[^\]]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None
        return (m.group(1), m.group(2), int(m.group(3)))

    def parse_items(s):
        # returns list of (name, slot, qty)
        items = []
        # matches lines like "- item name [I23] quantity 5"
        for name, slot, qty in re.findall(r'-\s+([^\[]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            items.append((name.strip(), f'[{slot}]', int(qty)))
        return items

    # Only apply the rule when state is a crafting task and action is a move
    if not re.search(r'Craft an item of type:', state):
        return 0.0
    mv = parse_move_action(action)
    if not mv:
        return 0.0
    src_slot, dst_slot, qty = mv

    state_items = parse_items(state)
    choice_items = parse_items(choice)

    # Build slot -> (name, qty) maps
    state_slot_map = {slot: (name, q) for name, slot, q in state_items}
    choice_slot_map = {slot: (name, q) for name, slot, q in choice_items}

    # The moved item must exist in the source in the original state
    if src_slot not in state_slot_map:
        return -1.0

    moved_name, src_prev = state_slot_map[src_slot]

    # Get destination previous qty (could be absent)
    dst_prev = state_slot_map.get(dst_slot, (moved_name, 0))[1]

    # In the choice, the source may be missing (treated as zero) or present
    src_choice_name, src_new = choice_slot_map.get(src_slot, (moved_name, 0))
    dst_choice_name, dst_new = choice_slot_map.get(dst_slot, (moved_name, 0))

    # Check that the item name at src in state matches moved_name
    # and that the move in choice applies to the same item name
    # Allow destination/source to have the moved item name or be absent.
    move_valid = True
    # Source decrease by qty for the moved item (or removed)
    # Accept if the source in choice has same name or missing (removed)
    if src_choice_name != moved_name and src_choice_name != 0:
        # if src slot changed to a different item name, treat as invalid move
        move_valid = False
    if (src_prev - src_new) != qty:
        move_valid = False
    # Destination increase by qty for moved item
    if dst_choice_name != moved_name and dst_choice_name != 0:
        # destination may be new item or same name; if different name, invalid
        move_valid = False
    if (dst_new - dst_prev) != qty:
        move_valid = False

    if not move_valid:
        return -1.0

    # Build name -> total qty mapping excluding the moved item
    def totals(items):
        d = {}
        for name, slot, q in items:
            if name == moved_name:
                continue
            d[name] = d.get(name, 0) + q
        return d

    totals_state = totals(state_items)
    totals_choice = totals(choice_items)

    # Check whether any non-moved item total changed
    all_names = set(totals_state.keys()) | set(totals_choice.keys())
    non_moved_changed = any(totals_state.get(n, 0) != totals_choice.get(n, 0) for n in all_names)

    if non_moved_changed:
        return 1.0
    else:
        # Move was applied but no other item changed -> likely omitted craft side-effect
        return -0.8

# Rule 2
def rule_reward(state, action, choice):
    import re

    def parse_craft_target(s):
        m = re.search(r'Craft an item of type:\s*([^\n\r]+)', s)
        return m.group(1).strip() if m else None

    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, 0
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns dict slot -> (name, qty) and also totals_by_name across non-[0] slots
        items = {}
        for m in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = m[0].strip()
            slot = f'[{m[1]}]'
            qty = int(m[2])
            items[slot] = (name, qty)
        return items

    # parse inputs
    craft_target = parse_craft_target(state)
    src_slot, dst_slot, qty = parse_move_action(action)
    state_items = parse_items(state)
    choice_items = parse_items(choice)

    # If not a craft instruction or not a move action, we do not apply this rule
    if craft_target is None or src_slot is None:
        return 0.0

    # Check crafted output presence and correct name
    crafted_ok = False
    if '[0]' in choice_items:
        name0, qty0 = choice_items['[0]']
        if name0 == craft_target and qty0 > 0:
            crafted_ok = True

    # Check move correctness
    move_ok = False
    # source must exist in state
    if src_slot in state_items:
        moved_name, src_prev_q = state_items[src_slot]
        # after choice, source slot may be absent (treated as 0) or present
        dst_prev_q = state_items.get(dst_slot, (None, 0))[1]
        src_new_name, src_new_q = choice_items.get(src_slot, (moved_name, 0))
        dst_new_name, dst_new_q = choice_items.get(dst_slot, (moved_name, 0))

        # destination name must be same moved_name (or previously different but becomes moved_name) and increase by qty
        dst_increased = (dst_new_q - dst_prev_q) == qty and (dst_new_name == moved_name)
        # source decreased by qty (allow source to disappear)
        src_decreased = (src_prev_q - src_new_q) == qty and (src_new_name in (moved_name, None))

        if dst_increased and src_decreased:
            move_ok = True

    # Check unrelated items unchanged (across non-[0] slots) except for the moved item
    def totals_excluding_moved(items_dict, moved_name):
        d = {}
        for slot, (name, q) in items_dict.items():
            if slot == '[0]':
                continue
            if name == moved_name:
                continue
            d[name] = d.get(name, 0) + q
        return d

    unrelated_unchanged = True
    if src_slot in state_items:
        moved_name = state_items[src_slot][0]
        state_totals = totals_excluding_moved(state_items, moved_name)
        choice_totals = totals_excluding_moved(choice_items, moved_name)
        if state_totals != choice_totals:
            unrelated_unchanged = False
    else:
        # if moved item not present in state, be conservative
        unrelated_unchanged = False

    # Scoring:
    # Start neutral 0. Reward presence of crafted output and correct move.
    # Penalize missing crafted output, incorrect move, or unrelated changes.
    score = 0.0
    score += 0.6 if crafted_ok else -0.6
    score += 0.4 if move_ok else -0.4
    if not unrelated_unchanged:
        score -= 0.4

    # clamp to [-1, 1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0

    return float(score)

# Rule 3
def rule_reward(state, action, choice):
    import re
    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, 0
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns dict slot -> (name, qty) and also list of (name, slot, qty)
        items = []
        for m in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = m[0].strip()
            slot = f'[{m[1]}]'
            qty = int(m[2])
            items.append((name, slot, qty))
        slot_map = {slot: (name, qty) for (name, slot, qty) in items}
        return slot_map, items

    # parse
    src_slot, dst_slot, qty = parse_move_action(action)
    slot_map_state, items_state = parse_items(state)
    slot_map_choice, items_choice = parse_items(choice)

    # If not a move action, give neutral small negative (rule not applicable)
    if src_slot is None:
        return 0.0

    # Get moved item info from state
    if src_slot not in slot_map_state:
        # source absent in state -> unlikely correct
        return -0.8

    moved_name, src_prev = slot_map_state[src_slot]

    # destination previous quantity (may be 0)
    dst_prev = slot_map_state.get(dst_slot, (None, 0))[1]
    dst_new_name, dst_new = slot_map_choice.get(dst_slot, (None, 0))

    # source new quantity in choice (treat missing as 0)
    src_new_name, src_new = slot_map_choice.get(src_slot, (None, 0))

    # Quick duplication check: if source unchanged (same qty) then strongly penalize
    if src_prev == src_new and src_prev > 0:
        return -1.0

    # Check destination increment by exactly qty and same item name
    dst_ok = False
    if dst_new_name == moved_name and (dst_new - dst_prev) == qty:
        dst_ok = True

    # Check source decreased by exactly qty (allow missing slot treated as 0)
    src_ok = False
    if src_new_name in (None, moved_name):
        if (src_prev - src_new) == qty:
            src_ok = True

    # Check slot [0] presence/increase (craft output must be added or increased)
    total0_prev = sum(q for (n, s, q) in items_state if s == '[0]')
    total0_new = sum(q for (n, s, q) in items_choice if s == '[0]')
    zero_ok = (total0_new > total0_prev)

    # Check unrelated items unchanged (exclude moved_name and exclude slot [0])
    def totals_excluding(items, exclude_name):
        d = {}
        for name, slot, q in items:
            if slot == '[0]':
                continue
            if name == exclude_name:
                continue
            d[name] = d.get(name, 0) + q
        return d

    totals_state = totals_excluding(items_state, moved_name)
    totals_choice = totals_excluding(items_choice, moved_name)
    unrelated_same = (totals_state == totals_choice)

    # Compose score from checks (weights chosen to strongly prefer correct behavior)
    score = 0.0
    if dst_ok:
        score += 0.45
    if src_ok:
        score += 0.35
    if zero_ok:
        score += 0.3
    if unrelated_same:
        score += 0.2

    # Penalties for obvious failures
    if not dst_ok:
        score -= 0.6
    if not src_ok:
        score -= 0.5
    if not zero_ok:
        score -= 0.6
    if not unrelated_same:
        score -= 0.3

    # Clamp to [-1, 1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0

    return float(score)

# Rule 4
def rule_reward(state, action, choice):
    import re

    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, None
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns dict mapping slot -> (name, qty)
        items = {}
        for m in re.findall(r'-\s+([^\[]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = m[0].strip()
            slot = f'[{m[1]}]'
            qty = int(m[2])
            items[slot] = (name, qty)
        return items

    src, dst, q = parse_move_action(action)
    if src is None:
        # rule not applicable
        return 0.0

    s_items = parse_items(state)
    c_items = parse_items(choice)

    # Get moved item name and previous amounts
    src_prev = s_items.get(src)
    if src_prev is None:
        # nothing to move in source -> invalid
        return -1.0
    moved_name, src_prev_qty = src_prev

    # Destination previous qty (may be absent)
    dst_prev = s_items.get(dst)
    dst_prev_name, dst_prev_qty = (dst_prev[0], dst_prev[1]) if dst_prev else (None, 0)

    # Destination in choice
    dst_new = c_items.get(dst)
    if dst_new is None:
        # destination missing after move -> error
        return -1.0
    dst_new_name, dst_new_qty = dst_new

    # Source in choice (may be absent if qty becomes 0)
    src_new = c_items.get(src)
    if src_new is None:
        src_new_name, src_new_qty = (None, 0)
    else:
        src_new_name, src_new_qty = src_new

    # 1) Destination must contain same item name and increased by q
    if dst_new_name != moved_name:
        return -1.0
    if (dst_new_qty - dst_prev_qty) != q:
        return -1.0

    # 2) Source must be decreased by q (or removed)
    if src_new_name not in (moved_name, None):
        return -1.0
    if (src_prev_qty - src_new_qty) != q:
        return -1.0

    # 3) No other slots/items (except [0]) should change
    def compare_except_allowed(old, new, allowed_slots):
        # returns True if all slots other than allowed_slots have identical (name, qty)
        old_keys = set(old.keys())
        new_keys = set(new.keys())
        keys = old_keys | new_keys
        for k in keys:
            if k in allowed_slots:
                continue
            o = old.get(k)
            n = new.get(k)
            if o != n:
                return False
        return True

    allowed = {src, dst, '[0]'}
    others_ok = compare_except_allowed(s_items, c_items, allowed)

    # 4) Check whether output [0] was created/changed (positive signal)
    out_prev = s_items.get('[0]')
    out_new = c_items.get('[0]')
    output_changed = (out_prev != out_new)

    # Scoring:
    # If core invariants satisfied (src/dst correct), we reward more if no unrelated changes and output changed.
    if not others_ok and output_changed:
        # some unrelated changes present but output did change -> modest positive
        return 0.3
    if not others_ok and not output_changed:
        # unrelated changes and no output change -> weak negative
        return -0.6
    # others_ok == True
    if output_changed:
        return 1.0  # best: only src/dst and output changed as expected
    else:
        return 0.5  # acceptable: src/dst changed correctly, but no output produced (still plausible)

# Rule 5
def rule_reward(state, action, choice):
    import re

    def parse_target(s):
        m = re.search(r'Craft an item of type:\s*([^\n\r]+)', s)
        return m.group(1).strip() if m else None

    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[^\]]+\])\s*to\s*(\[[^\]]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, None
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns dict slot -> (name, qty)
        items = {}
        # match lines like: - item_name [SLOT] quantity N
        for name, slot, qty in re.findall(r'-\s+([^\[]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            items[f'[{slot}]'] = (name.strip(), int(qty))
        return items

    # Only apply this rule for move actions in craft tasks
    target = parse_target(state)
    src_slot, dst_slot, qty = parse_move_action(action)
    if (src_slot is None) or (target is None):
        return 0.0  # rule not applicable, return neutral

    before = parse_items(state)
    after = parse_items(choice)

    # Helper to get name and count (missing slot -> (None,0))
    def get_slot(d, slot):
        return d.get(slot, (None, 0))

    src_name_before, src_qty_before = get_slot(before, src_slot)
    dst_name_before, dst_qty_before = get_slot(before, dst_slot)
    src_name_after, src_qty_after = get_slot(after, src_slot)
    dst_name_after, dst_qty_after = get_slot(after, dst_slot)

    score_components = []

    # 1) Move source decreased by exactly qty and name preserved (or removed if 0)
    src_ok = False
    if src_name_before is None:
        src_ok = False
    else:
        # name must remain same unless quantity becomes zero and slot removed
        if src_qty_after == 0 and src_slot not in after:
            # allowed: slot removed as count reached 0
            # but check decrease amount
            if src_qty_before - 0 == qty:
                src_ok = True
        else:
            # name must be same
            if src_name_after == src_name_before and (src_qty_before - src_qty_after) == qty:
                src_ok = True
    score_components.append(1.0 if src_ok else 0.0)

    # 2) Destination increased by exactly qty and name equals moved item
    dst_ok = False
    moved_name = src_name_before
    if moved_name is not None:
        if dst_name_after == moved_name and (dst_qty_after - dst_qty_before) == qty:
            dst_ok = True
    score_components.append(1.0 if dst_ok else 0.0)

    # 3) No unrelated changes to other slots (names and quantities), ignoring [0], src_slot, dst_slot
    other_ok = True
    for slot, (name_before, qty_before) in before.items():
        if slot in (src_slot, dst_slot, '[0]'):
            continue
        name_after, qty_after = get_slot(after, slot)
        if name_after != name_before or qty_after != qty_before:
            other_ok = False
            break
    # Also ensure no new unrelated slots have appeared in after (except [0] and src/dst)
    if other_ok:
        for slot, (name_after, qty_after) in after.items():
            if slot in (src_slot, dst_slot, '[0]'):
                continue
            if slot not in before:
                other_ok = False
                break
    score_components.append(1.0 if other_ok else 0.0)

    # 4) Crafted item presence: ensure target item appears somewhere in "after" with qty >= 1
    crafted_present = any((name == target and qty >= 1) for name, qty in after.values())
    # If target existed before, ensure it was not lost entirely (i.e., after count >= before count for that item across all slots)
    # compute total counts of target in before and after
    def total_of_target(d):
        tot = 0
        for name, q in d.values():
            if name == target:
                tot += q
        return tot
    before_target_total = total_of_target(before)
    after_target_total = total_of_target(after)
    crafted_ok = False
    if before_target_total > 0:
        # should not lose previously present crafted items entirely — allow same or increased (do not require increase)
        crafted_ok = (after_target_total >= before_target_total)
    else:
        crafted_ok = (after_target_total >= 1)
    # combine both: crafted_present and non-loss
    crafted_ok = crafted_present and crafted_ok
    score_components.append(1.0 if crafted_ok else 0.0)

    # Average component score -> map from [0,1] to [-1,1]
    avg = sum(score_components) / len(score_components)
    return 2 * avg - 1

# Rule 6
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
        for m in re.findall(r'-\s+([^\[]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = m[0].strip()
            slot = f'[{m[1]}]'
            qty = int(m[2])
            items[slot] = (name, qty)
        return items

    src_slot, dst_slot, qty = parse_move_action(action)
    if src_slot is None:
        # not a move action -> this rule does not apply; be neutral
        return 0.0

    state_items = parse_items(state)
    choice_items = parse_items(choice)

    # Basic existence and moved item identity
    if src_slot not in state_items:
        # cannot find moved item in original state
        return -1.0
    moved_name, src_prev_qty = state_items[src_slot]
    dst_prev_qty = state_items.get(dst_slot, (None, 0))[1]

    # In choice, get post quantities / names (absent treated as None and qty 0)
    c_src_name, c_src_qty = choice_items.get(src_slot, (None, 0))
    c_dst_name, c_dst_qty = choice_items.get(dst_slot, (None, 0))

    checks = 0
    total_checks = 3  # name+qty on dest, qty decrease on src, unrelated slots unchanged

    # Check 1: destination contains moved_name and increased by exactly qty
    dest_name_ok = (c_dst_name == moved_name)
    dest_qty_ok = (c_dst_qty - dst_prev_qty) == qty
    if dest_name_ok and dest_qty_ok:
        checks += 1

    # Check 2: source decreased by exactly qty and kept/removed appropriately
    src_qty_ok = (src_prev_qty - c_src_qty) == qty
    # src name in choice is allowed to be absent or same moved_name
    src_name_ok = (c_src_name is None) or (c_src_name == moved_name)
    if src_qty_ok and src_name_ok:
        checks += 1

    # Check 3: no other slots changed except src, dst, and [0]
    allowed_changed = {src_slot, dst_slot, '[0]'}
    other_slots = set(state_items.keys()) | set(choice_items.keys())
    other_slots = other_slots - allowed_changed
    unchanged_ok = True
    for s in other_slots:
        s_state = state_items.get(s)
        s_choice = choice_items.get(s)
        # both must exist and match name and qty
        if s_state is None or s_choice is None:
            unchanged_ok = False
            break
        if s_state[0] != s_choice[0] or s_state[1] != s_choice[1]:
            unchanged_ok = False
            break
    if unchanged_ok:
        checks += 1

    # Map checks (0..total_checks) to score in [-1, 1]
    score = (2.0 * checks / total_checks) - 1.0
    # clamp
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return score

# Rule 7
def rule_reward(state, action, choice):
    import re

    def parse_items(s):
        # returns list of (name, slot, qty) where slot formatted like '[I1]', '[A1]', '[0]'
        items = []
        for m in re.findall(r'-\s+([^\[]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = m[0].strip()
            slot = f'[{m[1]}]'
            qty = int(m[2])
            items.append((name, slot, qty))
        return items

    # parse move action: "move: from [I35] to [A1] with quantity 1"
    m = re.search(r'move:\s*from\s*(\[[A-Z]?\d+\])\s*to\s*(\[[A-Z]?\d+\])\s*with\s*quantity\s*(\d+)', action)
    if not m:
        # not applicable rule
        return 0.0

    src_slot, dst_slot, move_qty = m.group(1), m.group(2), int(m.group(3))

    s_items = parse_items(state)
    c_items = parse_items(choice)

    # build dicts
    s_by_slot = {slot: (name, qty) for (name, slot, qty) in s_items}
    c_by_slot = {slot: (name, qty) for (name, slot, qty) in c_items}

    # find moved item in the original state
    if src_slot not in s_by_slot:
        # unexpected: source not present in initial state
        return -1.0

    moved_name, src_prev_qty = s_by_slot[src_slot]

    # Determine whether we expect a dye output based on moved item heuristics
    moved_lower = moved_name.lower()
    dye_expected = ('sunflower' in moved_lower) or ('lily' in moved_lower) or ('_dye' in moved_lower) or ('flower' in moved_lower)

    score = 0.0

    # Check move correctness: source decreased by move_qty
    src_new_qty = c_by_slot.get(src_slot, (None, 0))[1]
    src_decrease_ok = (src_prev_qty - src_new_qty) == move_qty
    if src_decrease_ok:
        score += 0.35
    else:
        score -= 0.6

    # Check destination slot now contains moved_name and increased by move_qty (or equals move_qty if absent before)
    dst_prev_name, dst_prev_qty = s_by_slot.get(dst_slot, (None, 0))
    dst_new_name, dst_new_qty = c_by_slot.get(dst_slot, (None, 0))

    dst_increase_ok = False
    if dst_new_name == moved_name:
        # compute prev qty for that slot (could be 0 if absent)
        prev_qty = dst_prev_qty if dst_prev_name == moved_name else 0
        if (dst_new_qty - prev_qty) == move_qty:
            dst_increase_ok = True

    if dst_increase_ok:
        score += 0.25
    else:
        score -= 0.4

    # If a dye output is expected, ensure slot [0] exists and its name contains "dye"
    out_name, out_qty = c_by_slot.get('[0]', (None, 0))
    if dye_expected:
        if out_name and 'dye' in out_name.lower():
            score += 0.4
        else:
            # heavy penalty if dye expected but missing or wrong type
            score -= 0.9
    else:
        # if dye is not expected, it's okay for [0] to be absent; but if present as a dye it's suspicious but not strongly penalized here
        pass

    # Penalize unrelated changes: totals for all names except moved_name and output slot [0] should be unchanged
    def totals_excluding(items, exclude_name, exclude_slot0=True):
        d = {}
        for name, slot, qty in items:
            if slot == '[0]' and exclude_slot0:
                continue
            if name == exclude_name:
                continue
            d[name] = d.get(name, 0) + qty
        return d

    s_tot = totals_excluding(s_items, moved_name, exclude_slot0=True)
    c_tot = totals_excluding(c_items, moved_name, exclude_slot0=True)

    if s_tot == c_tot:
        score += 0.3
    else:
        score -= 0.4

    # clamp to [-1, 1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0

    return float(score)

# Rule 8
def rule_reward(state, action, choice):
    import re

    # Parse craft target from state header
    m_target = re.search(r'Craft an item of type:\s*([^\n\r]+)', state)
    if not m_target:
        return 0.0
    craft_target = m_target.group(1).strip()

    # Parse move action
    m_action = re.search(r'move:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', action)
    if not m_action:
        return 0.0
    src_slot = m_action.group(1)
    dst_slot = m_action.group(2)
    qty = int(m_action.group(3))

    # Helper to parse inventory lists into dict slot -> (name, qty)
    def parse_items(block):
        items = {}
        # match lines like: - item_name [I3] quantity 1
        for name, slot, q in re.findall(r'-\s+([^\[]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', block):
            slot_id = f'[{slot}]'
            items[slot_id] = (name.strip(), int(q))
        return items

    state_items = parse_items(state)
    choice_items = parse_items(choice)

    # Only apply this rule when craft_target is expected to appear at [0] after action
    # Compute checks (4 checks total)
    checks = 0
    total_checks = 4.0

    # 1) Check craft output at [0] present and increased by 1 (or present with qty 1 if absent before)
    prev_out_name, prev_out_q = state_items.get('[0]', (None, 0))
    out_name, out_q = choice_items.get('[0]', (None, 0))
    if out_name == craft_target and (out_q - prev_out_q) == 1:
        checks += 1

    # 2) Check destination slot has moved item and quantity increased by qty (or equals qty if dst absent before)
    moved = state_items.get(src_slot)
    if moved is None:
        # if source doesn't exist in state, cannot verify; return neutral
        return 0.0
    moved_name, src_prev_q = moved
    dst_prev_name, dst_prev_q = state_items.get(dst_slot, (None, 0))
    dst_choice = choice_items.get(dst_slot)
    if dst_choice is not None:
        dst_choice_name, dst_choice_q = dst_choice
        # destination must hold the same moved item name and quantity increase should be qty
        if dst_choice_name == moved_name and (dst_choice_q - dst_prev_q) == qty:
            checks += 1

    # 3) Check source slot decreased by qty (or removed if becomes zero)
    src_choice = choice_items.get(src_slot)
    if src_choice is None:
        # removed: acceptable if src_prev_q == qty
        if src_prev_q == qty:
            checks += 1
    else:
        src_choice_name, src_choice_q = src_choice
        if src_choice_name == moved_name and (src_prev_q - src_choice_q) == qty:
            checks += 1

    # 4) All other slots (except src_slot, dst_slot, and [0]) must remain exactly unchanged
    unchanged = True
    for slot, (name, q) in state_items.items():
        if slot in (src_slot, dst_slot, '[0]'):
            continue
        # must exist in choice and match name and qty
        if slot not in choice_items:
            unchanged = False
            break
        cname, cq = choice_items[slot]
        if cname != name or cq != q:
            unchanged = False
            break
    # Also ensure choice did not introduce new unrelated slots (except maybe dst_slot or [0])
    for slot in choice_items:
        if slot in (src_slot, dst_slot, '[0]'):
            continue
        if slot not in state_items:
            unchanged = False
            break

    if unchanged:
        checks += 1

    # Map checks (0..4) to score in [-1, 1]
    score = (checks / total_checks) * 2.0 - 1.0
    # Clamp
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return score

# Rule 9
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
        for match in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = match[0].strip()
            slot = f'[{match[1]}]'
            qty = int(match[2])
            items.append((name, slot, qty))
        return items

    def slot_map(items):
        # returns dict slot -> (name, qty)
        return {slot: (name, qty) for (name, slot, qty) in items}

    # extract craft target from state header if present
    m_t = re.search(r'Craft an item of type:\s*([^\n]+)', state)
    craft_target = m_t.group(1).strip() if m_t else None

    src_slot, dst_slot, qty = parse_action(action)
    if not src_slot:
        # not a move action: not applicable
        return 0.0

    state_items = parse_items(state)
    choice_items = parse_items(choice)
    s_map = slot_map(state_items)
    c_map = slot_map(choice_items)

    # baseline score components
    score = 0.0
    max_score = 1.0

    # Check moved item existed in state
    if src_slot not in s_map:
        return -0.8  # trying to move from non-existent slot -> bad

    moved_name, src_prev_qty = s_map[src_slot]
    # Destination previous qty and name (if present)
    dst_prev = s_map.get(dst_slot, (moved_name, 0))
    dst_prev_name, dst_prev_qty = dst_prev

    # In choice, check src and dst slots
    c_src = c_map.get(src_slot)
    c_dst = c_map.get(dst_slot)

    # 1) Move correctness: src decreased by qty and dst increased by qty; names consistent
    move_ok = False
    if c_src is not None:
        c_src_name, c_src_qty = c_src
    else:
        c_src_name, c_src_qty = moved_name, 0  # missing slot treated as qty 0

    if c_dst is not None:
        c_dst_name, c_dst_qty = c_dst
    else:
        c_dst_name, c_dst_qty = moved_name, 0

    if c_src_name == moved_name and c_dst_name == moved_name and (src_prev_qty - c_src_qty) == qty and (c_dst_qty - dst_prev_qty) == qty:
        move_ok = True
        score += 0.5
    else:
        # small partial credit if names consistent and at least one side changed correctly
        partial = 0.0
        if c_src_name == moved_name and (src_prev_qty - c_src_qty) == qty:
            partial += 0.25
        if c_dst_name == moved_name and (c_dst_qty - dst_prev_qty) == qty:
            partial += 0.25
        score += partial

    # 2) Unrelated items unchanged (excluding moved item and slot [0])
    # compute totals by name across slots except [0]
    def totals_excluding(items, exclude_names):
        d = {}
        for name, slot, qty_i in items:
            if slot == '[0]':
                continue
            if name in exclude_names:
                continue
            d[name] = d.get(name, 0) + qty_i
        return d

    exclude_names = {moved_name}
    s_tot = totals_excluding(state_items, exclude_names)
    c_tot = totals_excluding(choice_items, exclude_names)

    # compare totals
    unchanged = True
    for k in set(s_tot.keys()) | set(c_tot.keys()):
        if s_tot.get(k, 0) != c_tot.get(k, 0):
            unchanged = False
            break
    if unchanged:
        score += 0.3
    else:
        # penalize small amount
        score -= 0.3

    # 3) Validate slot [0] output
    c_zero = c_map.get('[0]')
    # simulate state after move to detect plank-2x2 (heuristic): count plank-type items in A*/B* slots after the move
    # Build a map of A/B slots after move using c_map (since choice reflects after move). If c_map doesn't reflect
    # the expected move, still check using simulated application on s_map to be conservative.
    # We'll simulate by applying the move to s_map
    sim_map = dict(s_map)  # slot->(name,qty)
    # apply decrease at source
    sim_name, sim_qty = sim_map.get(src_slot, (moved_name, 0))
    sim_qty_after = sim_qty - qty
    if sim_qty_after <= 0:
        sim_map.pop(src_slot, None)
    else:
        sim_map[src_slot] = (sim_name, sim_qty_after)
    # apply increase at dest
    prev_dst_name, prev_dst_qty = sim_map.get(dst_slot, (moved_name, 0))
    # If dst existed with different name in state, we still consider the move should put moved_name there
    sim_map[dst_slot] = (moved_name, prev_dst_qty + qty)

    # count plank-type items across A* and B* in sim_map
    plank_count = 0
    for slot, (name, q_i) in sim_map.items():
        if re.match(r'^\[[AB]\d+\]$', slot):
            if 'plank' in name:  # heuristic: 'plank' substring
                plank_count += q_i

    crafting_table_expected = plank_count >= 4

    # Evaluate correctness of slot [0]
    zero_ok = True
    if c_zero is not None:
        zero_name, zero_qty = c_zero
        # Only allow zero_name if it matches craft_target or 'crafting_table'
        allowed_zero = False
        if craft_target and zero_name == craft_target:
            allowed_zero = True
        if zero_name == 'crafting_table':
            allowed_zero = True
        if not allowed_zero:
            # introducing unrelated output -> heavy penalty
            score -= 0.8
            zero_ok = False
        else:
            # reward small amount for valid output
            score += 0.2
    else:
        # no slot [0] present
        if crafting_table_expected:
            # expected crafting_table but missing -> penalty
            score -= 0.5
            zero_ok = False
        else:
            # fine to have none
            score += 0.0

    # Final clamp to [-1, 1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return float(score)

# Rule 10
def rule_reward(state, action, choice):
    """
    Returns a float in [-1, 1]. Positive => choice likely correct, negative => likely wrong.
    Applies only when state contains "Craft" and action is a move.
    """
    import re
    def parse_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z0-9]+\])\s*to\s*(\[[A-Z0-9]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, None
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns dict slot -> (name, qty)
        items = {}
        for name, slot, qty in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            slot_fmt = f'[{slot}]'
            items[slot_fmt] = (name.strip(), int(qty))
        return items

    # Only apply this rule for "Craft" states and move actions
    if "Craft" not in state:
        return 0.0

    src_slot, dst_slot, qty = parse_action(action)
    if src_slot is None:
        return 0.0

    state_items = parse_items(state)
    choice_items = parse_items(choice)

    # Source must exist in original state
    if src_slot not in state_items:
        return -1.0

    moved_name, src_prev = state_items[src_slot]
    dst_prev = state_items.get(dst_slot, (moved_name, 0))[1]

    # In choice, destination must exist (or be created) and have moved_name
    if dst_slot not in choice_items:
        return -1.0
    dst_new_name, dst_new = choice_items[dst_slot]
    if dst_new_name != moved_name:
        return -1.0
    if (dst_new - dst_prev) != qty:
        return -1.0

    # Source in choice may be missing (removed) or decreased by qty, and if present name must match moved_name
    src_new_name, src_new = choice_items.get(src_slot, (moved_name, 0))
    if src_slot in choice_items and src_new_name != moved_name:
        return -1.0
    if (src_prev - src_new) != qty:
        return -1.0

    # Check that no unrelated slots changed (allow src_slot, dst_slot, and [0])
    allowed_slots = {src_slot, dst_slot, '[0]'}
    # Compare each slot in union of keys except allowed_slots
    all_slots = set(state_items.keys()) | set(choice_items.keys())
    for slot in all_slots:
        if slot in allowed_slots:
            continue
        s_item = state_items.get(slot)
        c_item = choice_items.get(slot)
        # If either presence or name/qty differs for slots other than allowed, penalize
        if s_item != c_item:
            return -1.0

    # If we reach here, move was applied correctly and only allowed slot changes occurred
    # Give positive score proportional to meeting checks (perfect => 1.0)
    return 1.0

# Rule 11
def rule_reward(state, action, choice):
    """
    Returns a float in [-1,1]: positive if the choice likely implements the craft (move + other changes),
    negative if it only does the move or gets the move wrong.
    Applies only when state indicates a 'Craft an item of type:' and action is a move.
    """
    import re

    def parse_craft_intent(s):
        m = re.search(r'Craft an item of type:\s*([^\n\r]+)', s)
        return m.group(1).strip() if m else None

    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z0-9]+\])\s*to\s*(\[[A-Z0-9]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, None
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns dict slot -> (name, qty)
        items = {}
        for name, slot, qty in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            slot_key = f'[{slot}]'
            items[slot_key] = (name.strip(), int(qty))
        return items

    craft = parse_craft_intent(state)
    src_slot, dst_slot, qty = parse_move_action(action)
    if craft is None or src_slot is None:
        # rule not applicable; return neutral 0.0
        return 0.0

    state_items = parse_items(state)
    choice_items = parse_items(choice)

    # Ensure the source existed in the original state
    if src_slot not in state_items:
        return -1.0

    moved_name, src_prev = state_items[src_slot]
    # Some destinations may not exist in state_items; treat missing as quantity 0.
    dst_prev = choice_dst_prev = state_items.get(dst_slot, (moved_name, 0))[1]

    # Check that the candidate implements the move correctly:
    # destination in choice should have moved_name and increased by qty
    # source in choice should have decreased by qty (or be absent / zero)
    # Find in choice the dst and src entries
    dst_entry = choice_items.get(dst_slot)
    src_entry = choice_items.get(src_slot)

    move_correct = True
    # Check destination
    if dst_entry is None:
        # If destination absent in choice but it existed in state and the move should increase it -> incorrect
        # But if dst_prev was 0 in state and dst absent in choice, treat as incorrect
        move_correct = False
    else:
        dst_name, dst_new = dst_entry
        if dst_name != moved_name:
            move_correct = False
        else:
            # dst_prev from state might be missing; get it
            dst_prev_state = state_items.get(dst_slot, (moved_name, 0))[1]
            if (dst_new - dst_prev_state) != qty:
                move_correct = False

    # Check source
    if src_entry is None:
        # source must have decreased to zero (valid only if src_prev == qty)
        if src_prev != qty:
            move_correct = False
    else:
        src_name_c, src_new = src_entry
        if src_name_c != moved_name:
            move_correct = False
        else:
            if (src_prev - src_new) != qty:
                move_correct = False

    if not move_correct:
        return -1.0

    # Build expected inventory after the move only
    expected_after_move = dict(state_items)  # shallow copy
    # adjust source
    if src_prev == qty:
        # should disappear
        expected_after_move.pop(src_slot, None)
    else:
        expected_after_move[src_slot] = (moved_name, src_prev - qty)
    # adjust destination
    dst_prev_state = state_items.get(dst_slot, (moved_name, 0))[1]
    expected_after_move[dst_slot] = (moved_name, dst_prev_state + qty)

    # Now compare expected_after_move to choice_items
    # They might differ in ordering; compare slot-wise.
    def inventories_equal(inv_a, inv_b):
        # inv_* are dict slot->(name,qty)
        # They are equal if they have same slots and same (name,qty) per slot
        if set(inv_a.keys()) != set(inv_b.keys()):
            return False
        for k in inv_a:
            if inv_a[k] != inv_b[k]:
                return False
        return True

    if inventories_equal(expected_after_move, choice_items):
        # Candidate only performed the move and did not apply any craft-related changes
        return -0.8
    else:
        # Candidate implemented move and also made additional changes (likely crafting effects) -> reward
        return 1.0

# Rule 12
def rule_reward(state, action, choice):
    import re

    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, None
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns dict slot -> (name, qty)
        items = {}
        for name, slot, qty in re.findall(r'-\s+([^\[]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            items[f'[{slot}]'] = (name.strip(), int(qty))
        return items

    # Only apply this rule when the state is a Craft task and action is a move
    if 'Craft an item of type' not in state:
        return 0.0

    src_slot, dst_slot, qty = parse_move_action(action)
    if src_slot is None:
        return 0.0

    state_items = parse_items(state)
    choice_items = parse_items(choice)

    # Get moved item info from state
    moved = state_items.get(src_slot)
    if moved is None:
        # no item at source in the state -> can't validate move
        return -1.0

    moved_name, src_prev = moved
    dst_prev = state_items.get(dst_slot, (moved_name, 0))[1]

    # Get post-move values from the choice (if slot missing, treat as (None,0))
    src_choice = choice_items.get(src_slot, (None, 0))
    dst_choice = choice_items.get(dst_slot, (None, 0))
    src_choice_name, src_new = src_choice
    dst_choice_name, dst_new = dst_choice

    checks = 0
    max_checks = 3  # dst inc, src dec, totals unchanged

    # 1) Destination increased by qty with same item name
    dst_ok = (dst_choice_name == moved_name) and ((dst_new - dst_prev) == qty)
    if dst_ok:
        checks += 1

    # 2) Source decreased by qty for same item (or slot disappears / qty becomes 0)
    # Accept if name unchanged or slot removed; allow src_new == 0
    src_name_ok = (src_choice_name in (moved_name, None))
    src_qty_ok = ((src_prev - src_new) == qty)
    if src_name_ok and src_qty_ok:
        checks += 1

    # 3) Totals of unrelated items (exclude moved item and slot [0]) remain unchanged
    def totals(items):
        d = {}
        for slot, (name, cnt) in items.items():
            if name == moved_name:
                continue
            if slot == '[0]':
                continue
            d[name] = d.get(name, 0) + cnt
        return d

    t_state = totals(state_items)
    t_choice = totals(choice_items)
    if all(t_state.get(k, 0) == t_choice.get(k, 0) for k in set(t_state) | set(t_choice)):
        checks += 1

    # Additional important check: slot [0] must change for Craft tasks
    state_slot0 = state_items.get('[0]', (None, None))
    choice_slot0 = choice_items.get('[0]', (None, None))
    slot0_changed = (state_slot0 != choice_slot0)

    # Compute normalized score in [-1, 1]
    # If slot0 didn't change, treat as failure bias: reduce effective checks count by 1 (i.e., penalize)
    effective_checks = checks
    effective_max = max_checks
    if not slot0_changed:
        # penalize: pretend one check failed (but keep at least 0)
        effective_checks = max(0, effective_checks - 1)

    # Map to [-1,1]
    ratio = effective_checks / effective_max  # in [0,1]
    score = ratio * 2.0 - 1.0  # map to [-1,1]

    # Small guard: if dst_ok or src_qty_ok failed badly, strongly negative
    if not dst_ok and not src_qty_ok:
        score = -1.0

    # Clamp
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0

    return score

# Rule 13
def rule_reward(state, action, choice):
    import re

    # Known input -> (output_name, output_qty) conversions seen in examples
    conversion_map = {
        'sunflower': ('yellow_dye', 2),
        'pink_tulip': ('pink_dye', 1),
        # add more conversions here if known
    }

    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z0-9]+\])\s*to\s*(\[[A-Z0-9]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, None
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns dict slot -> (name, qty) and also a name->total_qty map (excluding [0] if needed)
        slot_map = {}
        name_totals = {}
        # match lines like: - sunflower [I1] quantity 1
        for name, slot, qty in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = name.strip()
            slot_label = f'[{slot}]'
            qty = int(qty)
            slot_map[slot_label] = (name, qty)
            name_totals[name] = name_totals.get(name, 0) + qty
        return slot_map, name_totals

    src_slot, dst_slot, qty = parse_move_action(action)
    if not src_slot:
        # Not a move action: this rule is not applicable; return 0 neutral
        return 0.0

    state_slots, state_totals = parse_items(state)
    choice_slots, choice_totals = parse_items(choice)

    # the moved item must exist in the state at src_slot
    if src_slot not in state_slots:
        return -1.0

    moved_name, src_prev = state_slots[src_slot]
    # destination previous quantity (may be absent)
    dst_prev = choice_prev = 0
    if dst_slot in state_slots:
        dst_prev = state_slots[dst_slot][1]
    else:
        dst_prev = 0

    # In choice, find new quantities (0 if absent)
    src_new = choice_slots.get(src_slot, (moved_name, 0))[1]
    dst_entry = choice_slots.get(dst_slot)
    if dst_entry:
        dst_new_name, dst_new = dst_entry
    else:
        dst_new_name, dst_new = moved_name, 0  # if slot absent, treat as zero

    # Basic existence checks
    score_parts = []

    # 1) Destination increased by exactly q for the same item name
    dest_ok = (dst_new_name == moved_name) and ((dst_new - dst_prev) == qty)
    score_parts.append(0.4 if dest_ok else 0.0)

    # 2) Source decreased by exactly q (or removed)
    source_ok = ((src_prev - src_new) == qty)
    score_parts.append(0.4 if source_ok else 0.0)

    # 3) Unrelated items unchanged (ignore moved_name and ignore output slot [0])
    def totals_excluding(state_totals_map, choice_totals_map, exclude_name):
        d_state = {}
        d_choice = {}
        for n, v in state_totals_map.items():
            if n == exclude_name:
                continue
            d_state[n] = v
        for n, v in choice_totals_map.items():
            if n == exclude_name:
                continue
            d_choice[n] = v
        return d_state, d_choice

    s_ex_state, s_ex_choice = totals_excluding(state_totals, choice_totals, moved_name)
    # allow output slot [0] changes, so if an item only appears/changes because it's in [0] we must ignore differences caused solely by [0]
    # We'll compute totals of non-[0] slots to be strict:
    def nonzero_slot_totals(slot_map):
        totals = {}
        for slot, (name, q) in slot_map.items():
            if slot == '[0]':
                continue
            totals[name] = totals.get(name, 0) + q
        return totals

    state_non0 = nonzero_slot_totals(state_slots)
    choice_non0 = nonzero_slot_totals(choice_slots)
    # Exclude moved_name since it is allowed to change
    state_non0.pop(moved_name, None)
    choice_non0.pop(moved_name, None)

    unrelated_ok = state_non0 == choice_non0
    score_parts.append(0.1 if unrelated_ok else 0.0)

    # 4) If moving into a crafting input (destination slot label starts with [A), check expected output in [0] if conversion known
    output_bonus = 0.0
    if dst_slot.startswith('[A'):
        conv = conversion_map.get(moved_name)
        if conv:
            expected_out_name, expected_out_qty = conv
            # previous output qty for that name
            prev_out_qty = 0
            prev_out_name = None
            if '[0]' in state_slots:
                prev_out_name, prev_out_qty = state_slots['[0]']
                # state [0] could be a different name - handle totals for [0]
                if prev_out_name != expected_out_name:
                    # but initial [0] could have different item; we need to count totals by name in [0]
                    # To be safe, compute choice [0] for expected name
                    prev_out_qty = 0
                    # If state has expected name elsewhere in [0], we can detect by totals. Simpler: use name_totals
                    prev_out_qty = state_totals.get(expected_out_name, 0) - sum(v for k, v in state_non0.items() if k == expected_out_name)
            # choice output qty for expected name (count only in [0])
            choice_out_qty = 0
            if '[0]' in choice_slots:
                out_name, out_qty = choice_slots['[0]']
                if out_name == expected_out_name:
                    choice_out_qty = out_qty
                else:
                    # If choice [0] holds different name, but expected name might appear elsewhere,
                    # fall back to name_totals difference for expected name (but that may include non-[0] slots)
                    choice_out_qty = choice_totals.get(expected_out_name, 0) - (state_totals.get(expected_out_name,0) - prev_out_qty)
            # compute delta on output for expected name
            delta_out = choice_out_qty - prev_out_qty
            if delta_out == expected_out_qty:
                output_bonus = 0.1
            else:
                output_bonus = 0.0
    score_parts.append(output_bonus)

    base_score = sum(score_parts)  # maximum ~1.0

    # Map base_score in [0,1] to [-1,1]
    final = base_score * 2.0 - 1.0

    # Heuristic guard: if the predicted move used a different item name at dst or source changed to different name, strongly penalize
    if dst_slot in choice_slots and choice_slots[dst_slot][0] != moved_name:
        final = min(final, -0.9)
    if src_slot in choice_slots and choice_slots[src_slot][0] != moved_name:
        final = min(final, -0.9)

    return float(max(-1.0, min(1.0, final)))

# Rule 14
def rule_reward(state, action, choice):
    """
    Return score in [-1, 1]: +1 = fully matches the rule; -1 = completely violates.
    For move actions "move: from [S] to [D] with quantity q": require
      - moved item name decreases at S by q and increases at D by q (D must have same item name)
      - all other slots except S, D, and [0] preserve both item name and quantity
    Partial credit given by number of checks passed (3 checks).
    """
    import re

    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z0-9]+\])\s*to\s*(\[[A-Z0-9]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, 0
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns dict slot -> (name, qty)
        items = {}
        for m in re.findall(r'-\s+([^\[]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = m[0].strip()
            slot = f'[{m[1]}]'
            qty = int(m[2])
            items[slot] = (name, qty)
        return items

    src, dst, qty = parse_move_action(action)
    if src is None:
        # Not a move action this rule applies to; return neutral (0.0)
        return 0.0

    s_items = parse_items(state)
    c_items = parse_items(choice)

    # Get moved item info in original state
    if src not in s_items:
        # Source slot absent in original -> invalid action context (can't move what isn't there)
        return -1.0

    moved_name, src_prev_q = s_items[src]

    # Check 1: destination increased by qty and has same item name
    dst_prev = s_items.get(dst, (None, 0))[1]
    dst_prev_name = s_items.get(dst, (None, 0))[0]
    dst_new = c_items.get(dst, (None, 0))[1]
    dst_new_name = c_items.get(dst, (None, 0))[0]

    check_dst = False
    # destination name must equal moved name, and quantity increase must be exactly qty
    if dst_new_name == moved_name and (dst_new - dst_prev) == qty:
        check_dst = True

    # Check 2: source decreased by qty (and name unchanged or slot removed)
    src_new_name, src_new_q = c_items.get(src, (None, 0))
    check_src = False
    # either slot removed
    if src not in c_items:
        if src_prev_q == qty:
            check_src = True
    else:
        # name must be same and decreased by qty
        if src_new_name == moved_name and (src_prev_q - src_new_q) == qty:
            check_src = True

    # Check 3: all other slots except src, dst, and [0] remain identical in name and qty
    check_others = True
    for slot, (name, qn) in s_items.items():
        if slot in (src, dst, '[0]'):
            continue
        cpair = c_items.get(slot)
        if cpair is None:
            # original slot disappeared -> violation
            check_others = False
            break
        cname, cqn = cpair
        if cname != name or cqn != qn:
            check_others = False
            break
    # Also ensure no new unrelated slots were added in the choice (except dst if it was new, and [0])
    for slot, (name, qn) in c_items.items():
        if slot in (src, dst, '[0]'):
            continue
        if slot not in s_items:
            # added a new non-allowed slot
            check_others = False
            break

    checks = sum([check_dst, check_src, check_others])
    # Map checks (0..3) to score in [-1,1]
    score = (checks / 3.0) * 2.0 - 1.0
    return float(score)

# Rule 15
def rule_reward(state, action, choice):
    import re

    def parse_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, 0
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
        # Not a move action — this rule only applies to moves
        return 0.0

    s_items = parse_items(state)
    c_items = parse_items(choice)

    # If the source slot didn't exist in the state, cannot validate reliably -> penalize
    if src_slot not in s_items:
        return -1.0

    moved_name, src_prev = s_items[src_slot]

    def get_slot(items, slot):
        return items.get(slot, (None, 0))

    dst_name_prev, dst_prev = get_slot(s_items, dst_slot)
    dst_name_new, dst_new = get_slot(c_items, dst_slot)
    src_name_new, src_new = get_slot(c_items, src_slot)

    checks = 0
    total_checks = 3  # src change, dst change, other-items unchanged (with conservative exception)

    # Check 1: source decreased by exactly qty and name consistent
    # If source removed entirely -> treated as decreased by src_prev (src_new may be 0)
    src_new_qty = 0 if src_name_new is None else src_new
    if (src_prev - src_new_qty) == qty and (src_name_new in (moved_name, None)):
        checks += 1

    # Check 2: destination increased by exactly qty and same item name
    dst_increased_ok = False
    if dst_name_new == moved_name and (dst_new - dst_prev) == qty:
        checks += 1
        dst_increased_ok = True

    # Check 3: other items unchanged, with a conservative allowed exception for slot [0]
    # Build list of differing slots excluding src and dst
    all_slots = set(s_items.keys()) | set(c_items.keys())
    diffs = []
    for slot in all_slots:
        if slot in (src_slot, dst_slot):
            continue
        s_name, s_qty = get_slot(s_items, slot)
        c_name, c_qty = get_slot(c_items, slot)
        if s_name != c_name or s_qty != c_qty:
            diffs.append((slot, s_name, s_qty, c_name, c_qty))

    other_unchanged_ok = False
    if not diffs:
        other_unchanged_ok = True
    else:
        # Allow at most one differing slot, and only if it is '[0]' and its quantity change is small
        if len(diffs) == 1:
            slot, s_name, s_qty, c_name, c_qty = diffs[0]
            if slot == '[0]':
                # Conservative threshold for plausible crafting output: allow absolute change <= 4
                qty_change = abs((c_qty or 0) - (s_qty or 0))
                # Only allow this exception if the primary move checks passed (to avoid masking unrelated changes)
                if qty_change <= 4 and checks >= 2:
                    other_unchanged_ok = True
    if other_unchanged_ok:
        checks += 1

    score = checks / total_checks  # in [0,1]
    return 2.0 * score - 1.0

# Rule 16
def rule_reward(state, action, choice):
    import re

    # Only consider properly formatted smelt actions
    def is_smelt_action(a):
        return re.search(r'smelt:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', a) is not None

    # Parse lines like: "- cactus [A1] quantity 3"
    def parse_items(txt):
        items = []
        for name, slot, qty in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', txt):
            items.append((name.strip(), '[' + str(slot) + ']', int(qty)))
        return items

    def totals_by_name(items):
        totals = {}
        for name, _slot, qty in items:
            totals[name] = totals.get(name, 0) + int(qty)
        return totals

    if not is_smelt_action(action):
        return 0.0

    # Conservative mapping observed in examples
    mappings = {
        'cactus': 'green_dye',
        'sand': 'glass',
        'red_sand': 'glass'
    }
    expected_delta_src = 1
    expected_delta_dst = 1

    state_totals = totals_by_name(parse_items(state))
    choice_totals = totals_by_name(parse_items(choice))

    scores = []
    # Evaluate each mapping for which the source exists in the pre-action state
    for src, dst in mappings.items():
        if state_totals.get(src, 0) <= 0:
            continue  # don't evaluate mappings not present in the state

        # Compute name-level deltas (ignore slots)
        delta_src = max(0, int(state_totals.get(src, 0)) - int(choice_totals.get(src, 0)))
        delta_dst = max(0, int(choice_totals.get(dst, 0)) - int(state_totals.get(dst, 0)))

        # If there is no evidence of either the source decreasing or the product increasing,
        # skip to avoid penalizing unrelated changes.
        if delta_src == 0 and delta_dst == 0:
            continue

        score = 0.0
        # Prefer increases in the expected product (heavier weight) and decreases in the source
        if delta_dst > 0:
            score += 0.6
        else:
            score -= 0.2  # mild penalty when product didn't increase

        if delta_src > 0:
            score += 0.3
        else:
            score -= 0.2  # mild penalty when source didn't decrease

        # Bonus if observed deltas meet or exceed expected minimal change
        if delta_dst >= expected_delta_dst and delta_src >= expected_delta_src:
            score += 0.3

        # Clamp per-mapping score
        if score > 1.0:
            score = 1.0
        if score < -1.0:
            score = -1.0

        scores.append(score)

    if not scores:
        return 0.0

    # Be conservative: return the best-matching mapping score (avoid aggregating unrelated signals)
    final = max(scores)
    # Ensure final is a float in [-1, 1]
    if final > 1.0:
        final = 1.0
    if final < -1.0:
        final = -1.0
    return float(final) * 100

# Rule 17
def rule_reward(state, action, choice):
    import re

    def is_smelt_action(a):
        # Strictly match the smelt action pattern used in the examples
        return re.search(r'smelt:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', a) is not None

    def parse_items(txt):
        # Parse lines like "- name [SLOT] quantity N"
        items = []
        for name, slot, qty in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', txt):
            items.append((name.strip(), '[' + str(slot) + ']', int(qty)))
        return items

    def totals_by_name(items):
        totals = {}
        for name, _slot, qty in items:
            totals[name] = totals.get(name, 0) + int(qty)
        return totals

    # Only apply to strictly parsed smelt actions
    if not is_smelt_action(action):
        return 0.0

    # Known conservative smelt mappings and their minimal observed deltas
    mappings = {
        'cobblestone': ('stone', 1, 1),  # src -> (dst, expected_delta_src, expected_delta_dst)
        'clay_ball': ('brick', 1, 1),
    }

    state_totals = totals_by_name(parse_items(state))
    choice_totals = totals_by_name(parse_items(choice))

    best_score = 0.0
    any_source_present = False

    for src_name, (dst_name, expected_delta_src, expected_delta_dst) in mappings.items():
        src_in_state = int(state_totals.get(src_name, 0))
        if src_in_state <= 0:
            continue  # no evidence this mapping is relevant in the prior state

        any_source_present = True

        # Compute slot-agnostic deltas
        delta_src = max(0, src_in_state - int(choice_totals.get(src_name, 0)))
        delta_dst = max(0, int(choice_totals.get(dst_name, 0)) - int(state_totals.get(dst_name, 0)))

        score = 0.0
        # Reward presence of expected product increase (conservative, no penalties for unrelated changes)
        if delta_dst > 0:
            score += 0.6

        # Additional reward when both source decreased and product increased by at least expected amounts
        if delta_dst >= expected_delta_dst and delta_src >= expected_delta_src:
            score += 0.4

        if score > 1.0:
            score = 1.0

        if score > best_score:
            best_score = score

    # If none of the known sources were present in state, abstain to avoid false positives
    if not any_source_present:
        return 0.0

    # Return final score (within [-1, 1] as required; this rule is conservative and returns [0,1])
    return float(best_score) * 100

# Rule 18
def rule_reward(state, action, choice):
    import re

    # Strict detection of the smelt action (matches examples)
    m = re.search(r'smelt:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', action)
    if not m:
        return 0.0
    try:
        action_qty = int(m.group(3))
    except Exception:
        action_qty = 1

    # Known, conservative smelt mappings
    mappings = {
        'cobblestone': 'stone',
        'chorus_fruit': 'popped_chorus_fruit'
    }

    # Parse item lists of the form "- name [SLOT] quantity N"
    def parse_items(txt):
        items = []
        for name, slot, qty in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', txt):
            items.append((name.strip(), slot, int(qty)))
        return items

    def totals_by_name(items):
        totals = {}
        for name, _slot, qty in items:
            totals[name] = totals.get(name, 0) + int(qty)
        return totals

    state_totals = totals_by_name(parse_items(state or ""))
    choice_totals = totals_by_name(parse_items(choice or ""))

    # Consider only mappings whose source is present in the state (conservative)
    candidate_scores = []
    for src_name, dst_name in mappings.items():
        if state_totals.get(src_name, 0) <= 0:
            continue

        # Compute deltas ignoring slots (only positive changes)
        delta_src = max(0, int(state_totals.get(src_name, 0)) - int(choice_totals.get(src_name, 0)))
        delta_dst = max(0, int(choice_totals.get(dst_name, 0)) - int(state_totals.get(dst_name, 0)))

        # If neither source decreased nor product increased, skip (avoid penalizing unrelated changes)
        if delta_src == 0 and delta_dst == 0:
            continue

        # Expected minimal change is the smelt action quantity (conservative fallback to 1)
        expected = max(1, action_qty)

        score = 0.0
        # Prefer choices where the expected product increases
        if delta_dst > 0:
            score += 0.6
        else:
            score -= 0.6

        # Bonus if both source decreased and product increased by at least expected amount
        if delta_dst >= expected and delta_src >= expected:
            score += 0.4
        else:
            score -= 0.4

        # Clamp per-candidate score to [-1, 1]
        if score > 1.0:
            score = 1.0
        if score < -1.0:
            score = -1.0

        candidate_scores.append(score)

    if not candidate_scores:
        return 0.0

    # Be conservative: pick the best matching candidate mapping (do not aggregate penalties)
    final_score = max(candidate_scores)

    # Ensure final bounding
    if final_score > 1.0:
        final_score = 1.0
    if final_score < -1.0:
        final_score = -1.0

    return float(final_score) * 100

# Rule 19
def rule_reward(state, action, choice):
    import re

    # Only consider actions that match the smelt action format used in the examples
    def is_smelt_action(a):
        return re.search(r'smelt:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', a) is not None

    # Parse items of the form:
    # - <name> [<slot>] quantity <qty>
    def parse_items(txt):
        items = []
        for name, slot, qty in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', txt):
            items.append((name.strip(), '[' + str(slot) + ']', int(qty)))
        return items

    def totals_by_name(items):
        totals = {}
        for name, _slot, qty in items:
            totals[name] = totals.get(name, 0) + int(qty)
        return totals

    if not is_smelt_action(action):
        return 0.0

    # Conservative mapping learned from examples
    mapping = {
        'nether_quartz_ore': 'quartz',
        'nether_gold_ore': 'gold_ingot'
    }

    state_totals = totals_by_name(parse_items(state))
    choice_totals = totals_by_name(parse_items(choice))

    best_score = None  # we will take the best (most supporting) evidence among candidates

    for src, dst in mapping.items():
        src_state = int(state_totals.get(src, 0))
        if src_state <= 0:
            # If the source wasn't present in the state, do not evaluate this mapping
            continue

        delta_src = max(0, src_state - int(choice_totals.get(src, 0)))  # how much source decreased
        dst_state = int(state_totals.get(dst, 0))
        dst_choice = int(choice_totals.get(dst, 0))
        delta_dst = max(0, dst_choice - dst_state)  # how much product increased

        # No clear evidence for this mapping: abstain
        if delta_src == 0 and delta_dst == 0:
            continue

        # Scoring rules (conservative):
        # - Strong positive (1.0) when both source decreased and product increased by at least 1.
        # - Moderate positive (0.6) when both change but not clearly >=1 each (rare here).
        # - Small positive (0.2) if product increases but source does not (may be unrelated).
        # - Moderate negative (-0.4) if source decreases but product does not increase.
        if delta_src >= 1 and delta_dst >= 1:
            score = 1.0
        elif delta_src > 0 and delta_dst > 0:
            score = 0.6
        elif delta_dst > 0 and delta_src == 0:
            score = 0.2
        elif delta_src > 0 and delta_dst == 0:
            score = -0.4
        else:
            score = 0.0

        if best_score is None or score > best_score:
            best_score = score

    if best_score is None:
        return 0.0

    # Clamp to [-1, 1] just in case
    if best_score > 1.0:
        best_score = 1.0
    if best_score < -1.0:
        best_score = -1.0

    return float(best_score) * 100

# Rule 20
def rule_reward(state, action, choice):
    import re

    def parse_move_action(a):
        # match: move: from [I2] to [A1] with quantity 1
        m = re.search(r'move:\s*from\s*(\[[^\]]+\])\s*to\s*(\[[^\]]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, 0
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns dict slot -> (name, qty)
        items = {}
        for m in re.finditer(r'-\s+([^\[]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = m.group(1).strip()
            slot = f'[{m.group(2)}]'
            qty = int(m.group(3))
            items[slot] = (name, qty)
        return items

    src_slot, dst_slot, qty = parse_move_action(action)
    if src_slot is None:
        # Not a move action: this rule does not apply
        return 0.0

    state_items = parse_items(state)
    choice_items = parse_items(choice)

    # 1) Source must have existed in prior state
    if src_slot not in state_items:
        return -1.0

    moved_name, src_prev_qty = state_items[src_slot]

    # Conservative sanity check: cannot move more than is available
    if qty > src_prev_qty:
        return -1.0

    # Destination prior name/qty (if present)
    if dst_slot in state_items:
        dst_prev_name, dst_prev_qty = state_items[dst_slot]
    else:
        dst_prev_name, dst_prev_qty = (None, 0)

    # Strong check A: Source decreased by qty (or disappeared if expected 0)
    expected_src_qty = src_prev_qty - qty
    c_src = choice_items.get(src_slot)
    src_ok = False
    if c_src is None:
        # allowed only if expected_src_qty == 0
        if expected_src_qty == 0:
            src_ok = True
    else:
        # source present in choice: same name and decreased by qty
        if c_src[0] == moved_name and (src_prev_qty - c_src[1]) == qty:
            src_ok = True

    if not src_ok:
        # Source did not change appropriately -> fail strongly
        return -1.0

    # Strong check B: Destination increased by qty for same item (unless dst == src -> no-op)
    dst_ok = False
    if dst_slot == src_slot:
        # move to same slot should be a no-op: after move, slot should reflect same item name
        # and no net quantity change relative to expected_src_qty (which should equal src_prev_qty if qty == 0,
        # or if moving within same slot we expect identical previous behavior: treat as okay if choice shows same name
        # and qty equals src_prev_qty or equals expected_src_qty depending on interpretation).
        # Conservative approach: require that choice has the same name at src_slot and quantity equals expected_src_qty
        if c_src is not None and c_src[0] == moved_name and c_src[1] == expected_src_qty:
            dst_ok = True
        else:
            # If we cannot verify destination in same-slot move, treat as failure of move semantics.
            return -1.0
    else:
        # dst != src
        c_dst = choice_items.get(dst_slot)
        if c_dst is None:
            # If qty > 0 and dst doesn't appear in choice, it's suspicious -> fail
            # Conservative decision: require destination to appear if something was moved to it.
            if qty > 0:
                return -1.0
            else:
                dst_ok = True
        else:
            # destination present in choice: its name must be moved_name and increase equals qty
            if c_dst[0] == moved_name:
                prev_dst_qty = dst_prev_qty if dst_prev_name == moved_name else 0
                if c_dst[1] - prev_dst_qty == qty:
                    dst_ok = True
            if not dst_ok:
                # Destination present but doesn't show correct increase -> fail
                return -1.0

    # At this point, both source and destination consistency checks passed.
    # Now do conservative checks on other slots but do NOT fail strongly for benign differences.
    #  - If choice introduces brand-new slots (not in state) other than dst_slot or '[0]', downgrade to neutral.
    #  - If slots present in both state and choice differ (name or qty) and are not src/dst, treat as suspicious -> neutral.
    suspicious_new_slots = []
    for slot in choice_items.keys():
        if slot in (dst_slot, src_slot, '[0]'):
            continue
        if slot not in state_items:
            suspicious_new_slots.append(slot)

    suspicious_changes = []
    for slot, (name, qtys) in state_items.items():
        if slot in (src_slot, dst_slot):
            continue
        c = choice_items.get(slot)
        if c is None:
            # removal of a state slot is allowed (do not count as suspicious)
            continue
        # If slot exists in both, but name or qty changed, it's suspicious (but not fatal)
        if c[0] != name or c[1] != qtys:
            suspicious_changes.append(slot)

    # If we have any new unexpected slots (beyond dst or '[0]'), be conservative: return neutral
    if suspicious_new_slots:
        return 0.0

    # If only suspicious changes (existing slots with changed name/qty), be conservative: return neutral
    if suspicious_changes:
        return 0.0

    # Otherwise, everything consistent and no suspicious extras -> strong positive
    return 1.0

# Rule 21
def rule_reward(state, action, choice):
    import re

    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Za-z0-9]+\])\s*to\s*(\[[A-Za-z0-9]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None
        src, dst, q = m.group(1), m.group(2), int(m.group(3))
        return src, dst, q

    def parse_items(s):
        # returns list of (name, slot, qty)
        items = []
        for m in re.findall(r'-\s+([^\[]+?)\s*\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = m[0].strip()
            slot = f'[{m[1]}]'
            qty = int(m[2])
            items.append((name, slot, qty))
        return items

    # Only apply this rule when state indicates crafting and action is a move
    if 'Craft an item' not in state and 'Craft an item' not in state.splitlines()[0]:
        return 0.0

    mv = parse_move_action(action)
    if not mv:
        return 0.0
    src_slot, dst_slot, qty = mv

    state_items = parse_items(state)
    choice_items = parse_items(choice)

    # Build maps (name,slot) -> qty and slot->(name,qty) for convenience
    def build_maps(items):
        by_pair = {}
        slot_map = {}
        for name, slot, q in items:
            by_pair[(name, slot)] = q
            slot_map.setdefault(slot, []).append((name, q))
        return by_pair, slot_map

    s_by_pair, s_slot = build_maps(state_items)
    c_by_pair, c_slot = build_maps(choice_items)

    # Find moved item name in state at src_slot (slot may contain multiple items,
    # but typical data has single item per slot). We'll try to find the single
    # item at the source slot; if multiple, pick the one whose quantity >= qty.
    moved_name = None
    # find candidate(s)
    src_candidates = [(name, q) for (name, slot), q in s_by_pair.items() if slot == src_slot]
    if not src_candidates:
        # source didn't exist in state => can't validate move
        return -1.0

    # Prefer exact candidate with q >= qty
    for name, q in src_candidates:
        if q >= qty:
            moved_name = name
            break
    if moved_name is None:
        # pick first candidate
        moved_name = src_candidates[0][0]

    # Check move applied in choice:
    s_src_qty = s_by_pair.get((moved_name, src_slot), 0)
    c_src_qty = c_by_pair.get((moved_name, src_slot), 0)

    s_dst_qty = s_by_pair.get((moved_name, dst_slot), 0)
    c_dst_qty = c_by_pair.get((moved_name, dst_slot), 0)

    move_applied = False
    # Source decreased by qty (or removed)
    if s_src_qty - c_src_qty == qty:
        # Destination increased by qty (or created)
        if c_dst_qty - s_dst_qty == qty:
            move_applied = True

    if not move_applied:
        return -1.0

    # Now check for crafting side-effects beyond the move.
    # 1) Increase at slot [0] for any item (excluding the moved item at [0])
    produced_at_zero = False
    for (name, slot), c_q in c_by_pair.items():
        if slot == '[0]':
            s_q = s_by_pair.get((name, slot), 0)
            if c_q - s_q > 0:
                produced_at_zero = True
                break

    # 2) Any non-moved item whose quantity decreased relative to state (consumed ingredient)
    consumed_other = False
    for (name, slot), s_q in s_by_pair.items():
        # skip the moved item instances (same name and either src_slot or dst_slot)
        if name == moved_name and slot in (src_slot, dst_slot):
            continue
        c_q = c_by_pair.get((name, slot), 0)
        if c_q < s_q:
            consumed_other = True
            break

    # Decide score
    if produced_at_zero or consumed_other:
        return 1.0
    else:
        # Move was applied but no crafting side-effect -> likely incorrect
        return -0.8

# Rule 22
def rule_reward(state, action, choice):
    """
    Refined rule_reward:
    - Returns float in [-1, 1]. Positive -> likely correct, negative -> likely wrong.
    - Applies only when state contains "Craft an item of type:" and action is a move into grid.
    - Recognized recipes (kept from observed cases):
        - hay_block: exactly 9 wheat in grid -> +1 hay_block at [0]
        - ladder: exactly 7 stick in grid -> +3 ladder at [0]
        - <color>_banner: exactly 6 same-color *_wool + at least 1 stick in grid -> +1 <color>_banner at [0]
    - Refinement: If the recipe is NOT recognized, do NOT penalize changes at [0]. Only require that the move was applied and that unrelated inventory (excluding the moved item and slot [0]) is unchanged.
    """
    import re
    from collections import defaultdict, Counter

    # helper: parse items into dict slot -> (name, qty), and list of (name, slot, qty)
    def parse_items(inv_text):
        items = []
        # pattern captures "- name [SLOT] quantity N"
        for m in re.finditer(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', inv_text):
            name = m.group(1).strip()
            slot = f'[{m.group(2)}]'
            qty = int(m.group(3))
            items.append((name, slot, qty))
        slot_map = {slot: (name, qty) for (name, slot, qty) in items}
        # also produce totals by name (all slots)
        totals = defaultdict(int)
        for name, slot, qty in items:
            totals[name] += qty
        return slot_map, dict(totals), items

    # parse action: move: from [S] to [D] with quantity Q
    m = re.search(r'move:\s*from\s*(\[[A-Z0-9]+\])\s*to\s*(\[[A-Z0-9]+\])\s*with\s*quantity\s*(\d+)', action)
    if not m:
        # rule not applicable; neutral small score so we don't give misleading positives
        return 0.0
    src_slot, dst_slot, q = m.group(1), m.group(2), int(m.group(3))

    # only apply when state mentions craft
    if 'Craft an item of type:' not in state:
        return 0.0

    # parse state and choice inventories
    state_slots, state_totals, state_items = parse_items(state)
    choice_slots, choice_totals, choice_items = parse_items(choice)

    # moved item must exist at source in state
    if src_slot not in state_slots:
        return -1.0
    moved_name, src_prev_q = state_slots[src_slot]

    # ensure there was at least q to move
    if src_prev_q < q:
        return -1.0

    # determine destination previous
    dst_prev_name, dst_prev_q = state_slots.get(dst_slot, (None, 0))

    # compute expected quantities for source/destination after move (simulation)
    src_expected_q = src_prev_q - q
    # Determine expected dest name and quantity
    if dst_slot in state_slots and state_slots[dst_slot][0] == moved_name:
        dst_expected_name = moved_name
        dst_expected_q = dst_prev_q + q
    else:
        # Either destination empty or different name: realistic result is moved_name at dst with qty = q
        dst_expected_name = moved_name
        dst_expected_q = q

    # Build simulated grid counts (A/B/C slots) after move
    def grid_counts_from_slotmap(slotmap):
        cnt = Counter()
        for slot, (name, qty) in slotmap.items():
            inner = slot[1:-1]
            if inner and inner[0] in ('A', 'B', 'C'):
                cnt[name] += qty
        return cnt

    simulated_state_slots = dict(state_slots)  # shallow copy
    # update source
    if src_expected_q > 0:
        simulated_state_slots[src_slot] = (moved_name, src_expected_q)
    else:
        if src_slot in simulated_state_slots:
            simulated_state_slots.pop(src_slot)
    # update destination
    if dst_slot in state_slots and state_slots[dst_slot][0] == moved_name:
        simulated_state_slots[dst_slot] = (moved_name, dst_expected_q)
    else:
        simulated_state_slots[dst_slot] = (moved_name, dst_expected_q)

    simulated_grid = grid_counts_from_slotmap(simulated_state_slots)

    # detect recipe from simulated grid (limited recipe table)
    expected_output = None
    expected_output_qty = 0

    # hay_block: requires exactly 9 wheat in grid -> 1 hay_block
    if simulated_grid.get('wheat', 0) == 9:
        expected_output = 'hay_block'
        expected_output_qty = 1

    # ladder: requires exactly 7 sticks in grid -> 3 ladder
    if expected_output is None and simulated_grid.get('stick', 0) == 7:
        expected_output = 'ladder'
        expected_output_qty = 3

    # color banner: need exactly 6 same-color wool + at least 1 stick -> 1 <color>_banner
    if expected_output is None:
        wool_counts = {}
        for name, cnt in simulated_grid.items():
            if name.endswith('_wool'):
                wool_counts[name] = cnt
        for wool_name, cnt in wool_counts.items():
            if cnt == 6 and simulated_grid.get('stick', 0) >= 1:
                color = wool_name[:-5]  # strip "_wool"
                expected_output = f'{color}_banner'
                expected_output_qty = 1
                break

    # Helper to get slot (name, qty) from a slotmap; missing -> (None,0)
    def slot_get(slotmap, slot):
        return slotmap.get(slot, (None, 0))

    # Check move correctness in the choice
    choice_src_name, choice_src_q = slot_get(choice_slots, src_slot)
    choice_dst_name, choice_dst_q = slot_get(choice_slots, dst_slot)

    move_ok = True
    # source should have decreased by q and keep same name (or disappear)
    if src_expected_q == 0:
        # source should be absent in choice OR present with quantity 0 (we treat absent as OK)
        if src_slot in choice_slots:
            # if present, its quantity must equal src_expected_q (but src_expected_q is 0 so presence with nonzero is wrong)
            if choice_src_q != 0:
                move_ok = False
            else:
                # present with zero is unusual but accept it
                move_ok = True
    else:
        if choice_src_name != moved_name or choice_src_q != src_expected_q:
            move_ok = False

    # destination should have moved_name and expected quantity
    if choice_dst_name != dst_expected_name or choice_dst_q != dst_expected_q:
        move_ok = False

    if not move_ok:
        # If the move itself wasn't applied correctly, this is likely incorrect
        return -1.0

    # Now evaluate output expectations
    state_out_name, state_out_q = slot_get(state_slots, '[0]')
    choice_out_name, choice_out_q = slot_get(choice_slots, '[0]')

    # Check unrelated changes: totals for items excluding moved_name and excluding slot [0] should be equal
    def totals_excluding(slotmap, exclude_name):
        totals = defaultdict(int)
        for slot, (name, qty) in slotmap.items():
            if slot == '[0]':
                continue
            if name == exclude_name:
                continue
            totals[name] += qty
        return dict(totals)

    state_tot_exc = totals_excluding(state_slots, moved_name)
    choice_tot_exc = totals_excluding(choice_slots, moved_name)

    unrelated_ok = (state_tot_exc == choice_tot_exc)

    # Scoring logic

    if expected_output is None:
        # Conservative behavior for unrecognized recipes:
        # - Move must be applied (we already enforced move_ok)
        # - Unrelated inventory (excluding moved_name and [0]) should be unchanged
        # - Do NOT penalize arbitrary changes at [0] because the rule cannot identify the actual recipe
        if not unrelated_ok:
            return -0.5
        # Move applied and no unrelated changes: consider this likely correct
        return 1.0
    else:
        # Recognized recipe: require precise behavior for [0] and no unrelated changes
        # state may or may not have an existing output; compute delta
        prev_out_qty = state_out_q if state_out_name == expected_output else 0
        new_out_qty = choice_out_q if choice_out_name == expected_output else 0

        # The choice must have expected_output at [0]
        if choice_out_name != expected_output:
            return -0.9
        # The produced quantity must match expected_output_qty
        if (new_out_qty - prev_out_qty) != expected_output_qty:
            return -0.7
        # ensure no unrelated changes
        if not unrelated_ok:
            return -0.5
        # all checks passed
        return 1.0

# Rule 23
def rule_reward(state, action, choice):
    import re

    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, 0
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns dict slot->(name, qty) and name->total_qty
        slot_map = {}
        name_totals = {}
        # pattern matches "- name [SLOT] quantity N"
        for m in re.finditer(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = m.group(1).strip()
            slot = f'[{m.group(2)}]'
            qty = int(m.group(3))
            slot_map[slot] = (name, qty)
            name_totals[name] = name_totals.get(name, 0) + qty
        return slot_map, name_totals

    # parse action
    src_slot, dst_slot, qty = parse_move_action(action)
    if not src_slot:
        # this rule only applies to move actions; be neutral otherwise
        return 0.0

    s_slots, s_totals = parse_items(state)
    c_slots, c_totals = parse_items(choice)

    # If source slot doesn't exist in state, this is invalid move -> negative
    if src_slot not in s_slots:
        return -1.0
    moved_name, src_prev_qty = s_slots[src_slot]

    # Destination previous in state (may be absent)
    dst_prev_name, dst_prev_qty = s_slots.get(dst_slot, (None, 0))

    # Candidate values for src/dst (default to None,0 if absent)
    c_src_name, c_src_qty = c_slots.get(src_slot, (None, 0))
    c_dst_name, c_dst_qty = c_slots.get(dst_slot, (None, 0))

    # Check source decreased by qty for same item (source may be removed)
    # Be a bit stricter: if candidate still has a differing item at src, fail the src check.
    src_decreased_ok = False
    if c_src_name in (moved_name, None):
        # if candidate removed the slot (None) or same name remains with decreased qty
        if (src_prev_qty - c_src_qty) == qty:
            src_decreased_ok = True

    # Check destination increased by qty for same item (destination may be new)
    dst_increased_ok = False
    if c_dst_name == moved_name:
        # determine previous quantity (0 if absent)
        prev_q = dst_prev_qty
        if (c_dst_qty - prev_q) == qty:
            dst_increased_ok = True

    # If candidate places a different named item at src/dst, explicitly fail those parts
    if c_dst_name is not None and c_dst_name != moved_name:
        dst_increased_ok = False
    if c_src_name is not None and c_src_name != moved_name:
        src_decreased_ok = False

    # Check unrelated slots unchanged (excluding src, dst, and [0])
    unrelated_changed = False
    for slot, (name, qty0) in s_slots.items():
        if slot in (src_slot, dst_slot, '[0]'):
            continue
        c_name, c_qty = c_slots.get(slot, (None, 0))
        if c_name != name or c_qty != qty0:
            unrelated_changed = True
            break
    if not unrelated_changed:
        for slot, (c_name, c_qty) in c_slots.items():
            if slot in (src_slot, dst_slot, '[0]'):
                continue
            s_name, s_qty = s_slots.get(slot, (None, 0))
            if c_name != s_name or c_qty != s_qty:
                unrelated_changed = True
                break

    # Output slot handling
    target_match = re.search(r'Craft an item of type:\s*([^\n]+)', state)
    target = target_match.group(1).strip() if target_match else None

    s_out_name, s_out_qty = s_slots.get('[0]', (None, 0))
    c_out_name, c_out_qty = c_slots.get('[0]', (None, 0))

    # Detect whether output increased to the target in candidate
    out_increased_to_target = False
    if target is not None and c_out_name == target:
        # only consider it an "increase to target" if the output name is target and quantity actually grew
        if s_out_name != target or (c_out_qty > s_out_qty):
            out_increased_to_target = True

    # Conservative consumption check: only enforce when out_increased_to_target is True,
    # and only require that some inventory total decreased (non-[0]) OR the moved item total decreased.
    consumption_ok = True
    strict_consumption_checked = False
    if out_increased_to_target:
        strict_consumption_checked = True
        # require that some inventory name (excluding the output slot's name) decreased in total
        consumption_ok = False
        for name, s_tot in s_totals.items():
            if name == c_out_name:
                # skip if this name is actually the output; we need consumption from other inventory
                continue
            c_tot = c_totals.get(name, 0)
            if c_tot < s_tot:
                consumption_ok = True
                break
        # allow cases where the moved item itself was consumed (i.e., its total decreased)
        moved_tot_s = s_totals.get(moved_name, 0)
        moved_tot_c = c_totals.get(moved_name, 0)
        if moved_tot_c < moved_tot_s:
            consumption_ok = True

        # If consumption_ok is still False, be conservative: if there are many unrelated changes,
        # treat as ambiguous (do not fail hard here). We'll mark consumption failure, but with
        # lower weight and only make a hard negative if unrelated changes are also present.
    else:
        # Not an output increase to target: do not enforce consumption
        consumption_ok = True

    # Assemble weighted score
    # Make the scoring more conservative: reduce the output weight so failing that check does not flip
    # otherwise-correct moves too easily. Also soften the hard -1.0 behavior.
    w_src = 2.0
    w_dst = 2.0
    w_unrel = 1.0
    # If we performed strict consumption check, reduce weight to 2 (was 3), otherwise 0.
    w_out = 2.0 if strict_consumption_checked else 0.0
    total_w = w_src + w_dst + w_unrel + w_out

    passed = 0.0
    if src_decreased_ok:
        passed += w_src
    if dst_increased_ok:
        passed += w_dst
    if not unrelated_changed:
        passed += w_unrel
    if consumption_ok:
        passed += w_out

    # Soft negative handling:
    # If both source and destination checks fail:
    # - If there are multiple unrelated changes or a strict consumption check failed, return a stronger negative.
    # - Otherwise return a mild negative score (avoid harsh -1.0 for small/ambiguous differences).
    if not src_decreased_ok and not dst_increased_ok:
        if unrelated_changed or (strict_consumption_checked and not consumption_ok):
            return -1.0  # strong negative when there are substantial other changes or failed strict consumption
        else:
            # mild negative: produce a score slightly below 0 but not extreme
            # Use a fixed mild negative value to avoid flipping otherwise-correct examples harshly.
            return -0.4

    # Normalize to [ -1, 1 ] as before
    if total_w <= 0:
        # No meaningful checks applied (shouldn't happen for move actions) -> neutral
        return 0.0

    score_norm = passed / total_w  # in [0,1]
    return round(-1.0 + 2.0 * score_norm, 3)

# Rule 24
def rule_reward(state, action, choice):
    """
    Return a float in [-1, 1] estimating correctness of `choice` given `state` and `action`.
    More conservative than the original: only apply checks when they can be reliably evaluated,
    do not assume missing slots carry the moved name, and only compare totals for names
    present in both state and choice (intersection).
    """
    import re

    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[^\]]+\])\s*to\s*(\[[^\]]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, None
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns list of (name, slot, qty)
        items = []
        # Accept lines like: - name [SLOT] quantity N
        for name, slot, qty in re.findall(r'-\s+(.+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            items.append((name.strip(), f'[{slot}]', int(qty)))
        return items

    src_slot, dst_slot, qty = parse_move_action(action)
    if src_slot is None:
        # not a move action we can judge here
        return 0.0

    s_items = parse_items(state)
    c_items = parse_items(choice)

    state_map = {slot: (name, q) for (name, slot, q) in s_items}
    choice_map = {slot: (name, q) for (name, slot, q) in c_items}

    # SRC must exist in state
    if src_slot not in state_map:
        return -1.0

    moved_name, src_prev = state_map[src_slot]

    # previous dst in state
    dst_prev_name_state, dst_prev_q_state = state_map.get(dst_slot, (None, 0))

    # get new src and dst values in choice (don't assume moved_name as default)
    src_new_name, src_new_q = choice_map.get(src_slot, (None, 0))
    dst_new_name, dst_new_q = choice_map.get(dst_slot, (None, 0))

    checks_passed = 0
    checks_total = 0

    # Check A: destination increased by qty and name equals moved_name
    # Only apply if destination slot appears in the choice (we have reliable new dst data)
    if dst_slot in choice_map:
        checks_total += 1
        if dst_new_name == moved_name and (dst_new_q - dst_prev_q_state) == qty:
            checks_passed += 1

    # Check B: source decreased by qty (and name remains moved_name or slot removed)
    # If source appears in the choice, check quantity difference; if absent, accept only if emptied exactly.
    checks_total += 1
    if src_slot in choice_map:
        # source present in choice: verify decrease by qty and name not changed to some other non-moved name
        if (src_new_name in (moved_name, None)) and (src_prev - src_new_q) == qty:
            checks_passed += 1
    else:
        # source absent in choice: accept as valid only if it was exactly emptied by this move
        if src_prev == qty:
            checks_passed += 1

    # Check C: do not change output slot [0] unless SRC or DST is [0]
    # Only enforce if neither src nor dst is [0] and [0] appears in both state and choice
    if src_slot != '[0]' and dst_slot != '[0]' and ('[0]' in state_map) and ('[0]' in choice_map):
        checks_total += 1
        if state_map.get('[0]') == choice_map.get('[0]'):
            checks_passed += 1

    # Check D: totals of unrelated items (excluding moved_name and excluding slot [0]) unchanged
    # Conservative: only compare names that appear in both state and choice (intersection).
    def totals_excluding(items, exclude_name):
        d = {}
        for name, slot, q in items:
            if slot == '[0]':
                continue
            if name == exclude_name:
                continue
            d[name] = d.get(name, 0) + q
        return d

    s_tot_all = totals_excluding(s_items, moved_name)
    c_tot_all = totals_excluding(c_items, moved_name)

    # Limit comparison to names present in both
    s_names = set(s_tot_all.keys())
    c_names = set(c_tot_all.keys())
    common_names = s_names & c_names

    if len(common_names) > 0:
        checks_total += 1
        s_common = {name: s_tot_all[name] for name in common_names}
        c_common = {name: c_tot_all[name] for name in common_names}
        if s_common == c_common:
            checks_passed += 1
    else:
        # No reliable overlapping names to compare -- skip this check (conservative)
        pass

    if checks_total == 0:
        return 0.0
    frac = checks_passed / checks_total
    # map fraction [0..1] to [-1..1] conservatively as before
    return max(-1.0, min(1.0, 2.0 * frac - 1.0))

# Rule 25
def rule_reward(state, action, choice):
    import re

    def parse_action(a):
        # expects: move: from [X#] to [Y#] with quantity N
        m = re.search(r'move:\s*from\s*(\[[A-Z0-9]+\])\s*to\s*(\[[A-Z0-9]+\])\s*with\s*quantity\s*(\d+)', a, re.IGNORECASE)
        if not m:
            return None, None, 0
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns list of (name, slot, qty) with slot like [I17], [A1], [0]
        items = []
        # match lines like: - name [SLOT] quantity N
        for m in re.findall(r'-\s+(.+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = m[0].strip()
            slot = f'[{m[1]}]'
            qty = int(m[2])
            items.append((name, slot, qty))
        return items

    # Parse action - only apply rule if it's a move
    src_slot, dst_slot, qty = parse_action(action)
    if src_slot is None:
        # Not a move action: don't judge here
        return 0.0

    # Parse items in state and choice
    state_items = parse_items(state)
    choice_items = parse_items(choice)

    # Build slot lookups and name totals
    s_slot = {slot: (name, q) for (name, slot, q) in state_items}
    c_slot = {slot: (name, q) for (name, slot, q) in choice_items}

    def totals(items):
        d = {}
        for name, slot, q in items:
            d[name] = d.get(name, 0) + q
        return d

    s_tot = totals(state_items)
    c_tot = totals(choice_items)

    # Check 1: source present in original state
    if src_slot not in s_slot:
        return -1.0  # clearly invalid: moving from non-existent slot

    moved_name, src_prev = s_slot[src_slot]

    # Check 2: destination in choice must have moved_name and increase by qty
    dst_prev_qty = s_slot.get(dst_slot, (moved_name, 0))[1]
    dst_choice = c_slot.get(dst_slot)
    if dst_choice is None:
        dst_new_name, dst_new_qty = None, 0
    else:
        dst_new_name, dst_new_qty = dst_choice

    check_dst = (dst_new_name == moved_name) and ((dst_new_qty - dst_prev_qty) == qty)

    # Check 3: source decreased by qty (or removed); if present, name must match or be absent
    src_choice = c_slot.get(src_slot)
    if src_choice is None:
        src_new_name, src_new_qty = None, 0
    else:
        src_new_name, src_new_qty = src_choice
    check_src = ((src_prev - src_new_qty) == qty) and (src_new_name in (moved_name, None))

    # Resource-balance check (refined): allow other names to change only if net creation
    # (excluding the moved_name) is explainable by net consumption of other names.
    # Compute diffs for all names except moved_name
    names_union = set(s_tot.keys()) | set(c_tot.keys())
    diffs = {}
    for name in names_union:
        if name == moved_name:
            continue
        diffs[name] = c_tot.get(name, 0) - s_tot.get(name, 0)
    total_created = sum(d for d in diffs.values() if d > 0)
    total_consumed = sum(-d for d in diffs.values() if d < 0)

    # Conservative rule: allow net creation only if it's covered by net consumption.
    # If there is no net creation (total_created == 0) that's obviously fine.
    check_resource_balance = (total_created <= total_consumed)

    # Name consistency bonus: if names in src/dst slots in the choice (when present)
    # match moved_name, mark consistent. Be conservative: if a slot exists but name changed,
    # treat as inconsistency.
    name_consistent = True
    if src_slot in c_slot and c_slot[src_slot][0] != moved_name:
        name_consistent = False
    if dst_slot in c_slot and c_slot[dst_slot][0] != moved_name:
        name_consistent = False

    # Compose checks conservatively: require src and dst correctness first.
    checks = 0
    checks += 1 if check_dst else 0
    checks += 1 if check_src else 0
    checks += 1 if check_resource_balance else 0
    checks += 1 if name_consistent else 0

    # If parsing produced ambiguous or missing pieces (e.g., couldn't find dst_name),
    # avoid harsh penalties: return neutral 0.0 if both src/dst checks are missing but parsing succeeded.
    if not check_dst and not check_src:
        # ambiguous move result -> neutral
        return 0.0

    # Map checks (0..4) to reward in [-1, 1]
    reward = (checks / 4.0) * 2.0 - 1.0
    return float(reward)

# Rule 26
def rule_reward(state, action, choice):
    import re

    # Parse craft target from state's first line "State: Craft an item of type: <target>"
    m_target = re.search(r'Craft an item of type:\s*([^\n\r]+)', state)
    craft_target = m_target.group(1).strip() if m_target else None

    # Parse move action of the expected form
    m = re.search(r'move:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', action)
    if not m:
        # Rule only applies to this move action format; return neutral
        return 0.0
    src_slot, dst_slot, qty = m.group(1), m.group(2), int(m.group(3))

    # Helper to parse inventory items: returns list of (name, slot, qty)
    def parse_items(text):
        items = []
        for name, slot, q in re.findall(r'-\s+([^\[]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', text):
            items.append((name.strip(), f'[{slot}]', int(q)))
        return items

    state_items = parse_items(state)
    choice_items = parse_items(choice)

    # Convert to dict slot -> (name, qty)
    def to_dict(items):
        d = {}
        for name, slot, q in items:
            d[slot] = (name, q)
        return d

    sdict = to_dict(state_items)
    cdict = to_dict(choice_items)

    # Source must exist in state
    if src_slot not in sdict:
        # Clear invalid action (source does not exist)
        return -1.0

    moved_name, src_prev_qty = sdict[src_slot]
    # Destination previous quantity (if not present, treat as 0); keep previous name if present
    dst_prev_name, dst_prev_qty = sdict.get(dst_slot, (None, 0))

    # Destination in choice must exist (destination may be present)
    dst_entry = cdict.get(dst_slot)
    if not dst_entry:
        return -1.0
    dst_new_name, dst_new_qty = dst_entry

    # Source entry in choice (may be removed entirely => treat qty 0)
    src_entry_choice = cdict.get(src_slot, (moved_name, 0))
    src_new_name, src_new_qty = src_entry_choice

    # Basic validations
    # 1) Source had enough quantity
    if src_prev_qty < qty:
        # impossible move
        return -1.0

    # 2) Destination must contain the moved item name after the move
    if dst_new_name != moved_name:
        return -1.0

    # 3) Quantities: destination increased by qty, source decreased by qty
    move_dst_ok = (dst_new_qty - dst_prev_qty == qty)
    move_src_ok = (src_prev_qty - src_new_qty == qty)
    if not (move_dst_ok and move_src_ok):
        return -1.0

    # Now check for changes outside allowed set [src_slot, dst_slot, '[0]']
    allowed_slots = {src_slot, dst_slot, '[0]'}

    def non_allowed_changes(sd, cd):
        # returns True if any slot outside allowed_slots changed (name or qty),
        # or if any new slot outside allowed_slots was introduced
        # Check for changed or removed slots from state
        for slot, (sname, sq) in sd.items():
            if slot in allowed_slots:
                continue
            centry = cd.get(slot)
            if centry is None:
                # removed a non-allowed slot -> change
                return True
            cname, cq = centry
            if cname != sname or cq != sq:
                return True
        # check for new slots in choice outside allowed_slots
        for slot, (cname, cq) in cd.items():
            if slot in allowed_slots:
                continue
            if slot not in sd:
                return True
        return False

    if non_allowed_changes(sdict, cdict):
        # Be conservative: penalize but less harshly than before to avoid false positives.
        # This indicates other inventory slots changed; return a moderate negative to flag
        # likely incorrect broader mutations while not strictly flipping correct moves in ambiguous cases.
        return -0.5

    # If output [0] exists in choice, determine if it changed relative to state
    if '[0]' in cdict:
        out_name, out_qty = cdict['[0]']
        state_out = sdict.get('[0]')  # may be None
        state_out_qty = state_out[1] if state_out else 0
        out_qty_changed = (out_qty != state_out_qty)

        # If output equals craft target and increased, give full reward
        if craft_target and out_name == craft_target and out_qty > state_out_qty:
            return 1.0
        else:
            # Be conservative: do not penalize legitimate outputs that are not exactly the craft_target.
            # Treat the move as successful (neutral/high reward) rather than downgrade it.
            return 1.0

    # If we reach here, move was correct, no other slots changed, and there is no output [0]
    return 1.0

# Rule 27
def rule_reward(state, action, choice):
    import re

    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, 0
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns list of (name, slot, qty) with slot like [I17], [A1], [0]
        items = []
        for m in re.finditer(r'-\s+(.+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = m.group(1).strip()
            slot = f'[{m.group(2)}]'
            qty = int(m.group(3))
            items.append((name, slot, qty))
        return items

    def build_slot_map(items):
        # slot -> (name, qty)
        d = {}
        for name, slot, qty in items:
            d[slot] = (name, qty)
        return d

    src_slot, dst_slot, qty = parse_move_action(action)
    if not src_slot:
        # rule applies only to move actions
        return 0.0

    state_items = parse_items(state)
    choice_items = parse_items(choice)
    s_map = build_slot_map(state_items)
    c_map = build_slot_map(choice_items)

    # Source must exist in original state
    if src_slot not in s_map:
        return -1.0

    moved_name, src_prev = s_map[src_slot]
    # destination previous quantity for same name (0 if not present or different name)
    dst_prev = 0
    if dst_slot in s_map and s_map[dst_slot][0] == moved_name:
        dst_prev = s_map[dst_slot][1]

    # In choice, check source and destination entries
    src_new_name, src_new = c_map.get(src_slot, (None, 0))
    dst_new_name, dst_new = c_map.get(dst_slot, (None, 0))

    checks = 0.0
    # 1) Destination increased by qty and name matches moved_name
    if dst_new_name == moved_name and (dst_new - dst_prev) == qty:
        checks += 1.0

    # 2) Source decreased by qty and name matches (or absent)
    src_name_ok = (src_new_name == moved_name) or (src_slot not in c_map)
    if src_slot in c_map:
        src_qty_ok = (src_prev - src_new) == qty
    else:
        # source slot absent in choice: allow if it was exactly consumed
        src_qty_ok = (src_prev == qty)
    if src_name_ok and src_qty_ok:
        checks += 1.0

    # 3) No other items changed names or quantities (ignore src, dst, and special '[0]' slot)
    def totals_except(dmap, ignore_slots):
        # return dict of (slot, name) -> qty excluding ignored slots and excluding slot '[0]'
        res = {}
        for slot, (name, qty_val) in dmap.items():
            if slot in ignore_slots:
                continue
            if slot == '[0]':
                # treat [0] as special: do not include it in this comparison
                continue
            res[(slot, name)] = qty_val
        return res

    s_except = totals_except(s_map, {src_slot, dst_slot})
    c_except = totals_except(c_map, {src_slot, dst_slot})

    if s_except == c_except:
        checks += 1.0

    # 4) Conservative handling of a newly introduced '[0]' slot:
    # Only penalize if '[0]' is newly introduced AND the move checks largely failed.
    new_zero_slot = ('[0]' in c_map) and ('[0]' not in s_map)

    # Compose final score in [-1, 1]
    # checks ranges 0..3. Map to base_score in [ -0.2 .. 1.0 ] (so failing all gives -0.2)
    base_score = (checks / 3.0) * 1.2 - 0.2  # 0->-0.2, 3->1.0

    if new_zero_slot:
        # Be conservative: only apply a modest penalty when checks < 2 (i.e., move is suspicious).
        # This avoids punishing legitimate craft-output '[0]' entries when the move itself is correct.
        if checks < 2.0:
            base_score -= 0.6  # moderate penalty when the move mostly fails and a new [0] appears

    # clamp
    if base_score > 1.0:
        base_score = 1.0
    if base_score < -1.0:
        base_score = -1.0
    return float(base_score)

# Rule 28
def rule_reward(state, action, choice):
    import re

    def parse_move_action(a):
        # match "move: from [I13] to [A2] with quantity 1"
        m = re.search(r'move:\s*from\s*(\[[A-Za-z0-9]+\])\s*to\s*(\[[A-Za-z0-9]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, None
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns list of (name, slot, qty) with slot like [I17], [A1], [0]
        items = []
        for name, slot, qty in re.findall(r'-\s+([^\[]+?)\s*\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            items.append((name.strip(), f'[{slot}]', int(qty)))
        return items

    # Parse action
    src_slot, dst_slot, qty = parse_move_action(action)
    if src_slot is None:
        # Not a move action: this rule does not apply
        return 0.0

    q_items = parse_items(state)
    c_items = parse_items(choice)

    # Build slot -> (name, count) maps
    q_slot_map = {slot: (name, count) for (name, slot, count) in q_items}
    c_slot_map = {slot: (name, count) for (name, slot, count) in c_items}

    # Source must exist in original
    if src_slot not in q_slot_map:
        # invalid action (moving from non-existent slot) -> strong negative
        return -1.0

    moved_name, src_prev = q_slot_map[src_slot]
    dst_prev = q_slot_map.get(dst_slot, (None, 0))[1]

    # In choice, get new counts (if slot absent, default (None,0))
    c_src_name, c_src = c_slot_map.get(src_slot, (None, 0))
    c_dst_name, c_dst = c_slot_map.get(dst_slot, (None, 0))

    # Basic move checks (conservative)
    checks = 0

    # Destination: acceptable if it now contains the moved_name and increased by qty
    if (c_dst_name == moved_name) and ((c_dst - dst_prev) == qty):
        checks += 1

    # Source: acceptable if decreased by qty or disappeared
    # If missing in choice, treat as zero
    if (c_src_name in (moved_name, None)) and ((src_prev - c_src) == qty):
        checks += 1

    # Totals for "other items" excluding the moved item
    def totals_excluding(items, exclude_name=None, exclude_slot_none=True):
        d = {}
        for name, slot, count in items:
            if name == exclude_name:
                continue
            # if exclude_slot_none True, we will exclude slot '[0]' (craft output) from aggregates where needed
            d[name] = d.get(name, 0) + count
        return d

    # Totals excluding moved_name (but we'll compute both including and excluding slot [0])
    totals_q = totals_excluding(q_items, exclude_name=moved_name)
    totals_c = totals_excluding(c_items, exclude_name=moved_name)

    # Totals excluding moved_name and excluding occurrences at slot [0] in choice and state
    def totals_excluding_zero(items, exclude_name=None):
        d = {}
        for name, slot, count in items:
            if name == exclude_name:
                continue
            if slot == '[0]':  # ignore crafting output slot for conservative checks
                continue
            d[name] = d.get(name, 0) + count
        return d

    totals_q_nonzero = totals_excluding_zero(q_items, exclude_name=moved_name)
    totals_c_nonzero = totals_excluding_zero(c_items, exclude_name=moved_name)

    # If no increases in totals of other items (outside slot [0]), pass this conservative check
    increases_nonzero = [name for name in set(totals_q_nonzero) | set(totals_c_nonzero)
                         if totals_c_nonzero.get(name, 0) > totals_q_nonzero.get(name, 0)]
    if not increases_nonzero:
        checks += 1

    base_score = checks / 3.0  # 0.0..1.0

    penalty = 0.0

    # New item names handling: allow new names if they only appear at slot [0] (craft output).
    names_q = set(name for name, _, _ in q_items)
    names_c = set(name for name, _, _ in c_items)
    raw_new_names = (names_c - names_q) - {moved_name}

    # Determine if any new name appears outside slot [0] in the choice
    new_names_elsewhere = set()
    new_names_at_zero = set()
    for name in raw_new_names:
        appeared_outside_zero = False
        appeared_at_zero = False
        for n, slot, _ in c_items:
            if n != name:
                continue
            if slot == '[0]':
                appeared_at_zero = True
            else:
                appeared_outside_zero = True
            if appeared_outside_zero and appeared_at_zero:
                break
        if appeared_outside_zero:
            new_names_elsewhere.add(name)
        if appeared_at_zero and not appeared_outside_zero:
            new_names_at_zero.add(name)

    # Penalty for new names that actually appear outside the output slot (suspicious)
    if new_names_elsewhere:
        # Be conservative: apply a moderate penalty (not maximal).
        penalty += 0.6

    # Penalty for increases in non-zero-slot totals of other items (unexpected creation)
    if increases_nonzero:
        # If increases are small and move checks passed, be lenient; otherwise moderate penalty.
        penalty += 0.5

    # If totals differ outside slot [0] and there is no crafting output at [0], penalize softly.
    any_output_at_zero = any(slot == '[0]' for _, slot, _ in c_items)
    if not any_output_at_zero:
        # Compare totals excluding moved_name and excluding slot [0]
        if any(totals_q_nonzero.get(k, 0) != totals_c_nonzero.get(k, 0) for k in set(totals_q_nonzero) | set(totals_c_nonzero)):
            penalty += 0.4

    # Destination slot mismatch penalties (reduced)
    if c_dst_name is None:
        # Missing destination entry: moderate penalty only if move checks didn't clearly pass
        # If destination increase check passed, do not penalize missing slot listing (some representations omit it).
        if not ((c_dst_name == moved_name) and ((c_dst - dst_prev) == qty)):
            penalty += 0.2
    elif c_dst_name != moved_name:
        # Destination now holds a different item name -> suspicious, but be conservative
        penalty += 0.7

    # Safety guard: if both source and destination checks passed and there are no strong suspicions,
    # avoid returning a negative score. This prevents flipping correct predictions to wrong.
    strong_suspicion = (new_names_elsewhere or bool(increases_nonzero) or (c_dst_name is not None and c_dst_name != moved_name))
    if base_score >= 0.66 and not strong_suspicion:
        # Return a non-negative score reflecting good move (prefer positive)
        score = base_score - min(penalty, 0.5)  # allow only a small reduction
        if score < 0.0:
            score = 0.0
        return float(max(-1.0, min(1.0, score)))

    # Compose final score: base positive score minus penalties, clamp to [-1, 1]
    score = base_score - penalty
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return float(score)

# Rule 29
def rule_reward(state, action, choice):
    import re
    from collections import defaultdict

    # parse items: return slot_map: slot -> (name, qty) and name_totals: name -> total_qty
    def parse_items(s):
        slot_map = {}
        name_totals = defaultdict(int)
        # pattern: - <name> [<slot>] quantity <qty>
        for m in re.finditer(r'-\s*([^\[\n]+?)\s*\[([^\]]+)\]\s*quantity\s*(\d+)', s):
            name = m.group(1).strip()
            slot = f'[{m.group(2)}]'
            qty = int(m.group(3))
            slot_map[slot] = (name, qty)
            name_totals[name] += qty
        return slot_map, dict(name_totals)

    # parse action (move or smelt)
    def parse_action(a):
        m = re.search(r'^(move|smelt):\s*from\s*(\[[^\]]+\])\s*to\s*(\[[^\]]+\])\s*with\s*quantity\s*(\d+)', a.strip())
        if not m:
            return None
        typ = m.group(1)
        src = m.group(2)
        dst = m.group(3)
        q = int(m.group(4))
        return typ, src, dst, q

    parsed = parse_action(action)
    if not parsed:
        # only apply this rule for move/smelt; neutral otherwise
        return 0.0
    typ, src_slot, dst_slot, qty = parsed

    state_slots, state_totals = parse_items(state)
    choice_slots, choice_totals = parse_items(choice)

    # helper to get slot info
    def get_slot_info(slots, slot):
        return slots.get(slot, (None, 0))

    # source slot must exist in the state to validate the requested move/smelt.
    src_name, src_qty = get_slot_info(state_slots, src_slot)
    if src_name is None:
        # cannot validate (no source in state) -> be conservative: neutral
        return 0.0

    # If attempt to move/smelt more than present in the source slot, that's suspicious.
    # But only penalize if there's no evidence anywhere in the choice that those items appeared.
    if src_qty < qty:
        # check if any slot in the choice shows an increase of this same item name by qty
        # compute net change for this name across all slots
        state_total_for_name = state_totals.get(src_name, 0)
        choice_total_for_name = choice_totals.get(src_name, 0)
        net_change = choice_total_for_name - state_total_for_name
        # If there's no increase anywhere and the source doesn't have enough, it's a clear contradiction.
        if net_change <= 0:
            return -1.0
        else:
            # If there's evidence items increased somewhere, don't penalize; be conservative.
            return 0.0

    # If source and destination are the same slot, treat as neutral (no meaningful move)
    if src_slot == dst_slot:
        return 0.0

    # slot-level checks
    def src_decreased_by_required():
        _, src_qty_new = get_slot_info(choice_slots, src_slot)
        return (src_qty - src_qty_new) == qty

    def dst_increased_by_required_and_name_matches(expected_name):
        dst_name_prev, dst_qty_prev = get_slot_info(state_slots, dst_slot)
        dst_name_new, dst_qty_new = get_slot_info(choice_slots, dst_slot)
        if (dst_qty_new - dst_qty_prev) != qty:
            return False
        if dst_name_new is None:
            return False
        return dst_name_new == expected_name

    if typ == 'move':
        # Require:
        #  - slot-level source decreased by qty
        #  - slot-level destination increased by qty with same name
        #  - total of the item name unchanged across state and choice (pure move)
        if not src_decreased_by_required():
            return 0.0
        if not dst_increased_by_required_and_name_matches(src_name):
            return 0.0

        # Totals must be consistent for a true move: total of that item name should be unchanged.
        state_total = state_totals.get(src_name, 0)
        choice_total = choice_totals.get(src_name, 0)
        if state_total != choice_total:
            # Totals changed: this may indicate other operations (crafting/smelting) happened.
            # For conservatism, do not reward unless totals are consistent.
            return 0.0

        # All clear: reward positively.
        return 1.0

    elif typ == 'smelt':
        # Require:
        #  - slot-level source decreased by qty
        #  - slot-level destination increased by qty
        #  - destination name exists and is different from source name (transformation)
        #  - totals: source total decreased by qty and destination total increased by qty
        if not src_decreased_by_required():
            return 0.0
        dst_name_prev, dst_qty_prev = get_slot_info(state_slots, dst_slot)
        dst_name_new, dst_qty_new = get_slot_info(choice_slots, dst_slot)
        if (dst_qty_new - dst_qty_prev) != qty:
            return 0.0
        if dst_name_new is None:
            return 0.0
        if dst_name_new == src_name:
            return 0.0

        # Totals check for transformation
        src_state_total = state_totals.get(src_name, 0)
        src_choice_total = choice_totals.get(src_name, 0)
        dst_state_total = state_totals.get(dst_name_new, 0)
        dst_choice_total = choice_totals.get(dst_name_new, 0)

        if (src_state_total - src_choice_total) != qty:
            return 0.0
        if (dst_choice_total - dst_state_total) != qty:
            return 0.0

        # All clear: reward positively.
        return 1.0

    else:
        return 0.0

# Rule 30
def rule_reward(state, action, choice):
    import re

    # parse move action: return src_slot (like '[I6]'), dst_slot, qty (int)
    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[^\]]+\])\s*to\s*(\[[^\]]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, None
        return m.group(1), m.group(2), int(m.group(3))

    # parse inventory text into list of tuples (name, slot, qty)
    def parse_items(s):
        items = []
        # Matches lines like: - item name [SLOT] quantity N
        for name, slot, qty in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            items.append((name.strip(), f'[{slot}]', int(qty)))
        return items

    src_slot, dst_slot, qty = parse_move_action(action)
    # Rule applies only for move actions from non-[0] sources
    if src_slot is None:
        return 0.0
    if src_slot == '[0]':
        # Not applicable for moves from [0]
        return 0.0

    state_items = parse_items(state)
    choice_items = parse_items(choice)

    # Build mappings: slot -> (name, qty)
    slot_to_item_state = {slot: (name, q) for (name, slot, q) in state_items}
    slot_to_item_choice = {slot: (name, q) for (name, slot, q) in choice_items}

    # The moved item must exist at src in the original state
    if src_slot not in slot_to_item_state:
        return -1.0  # invalid: nothing to move

    moved_name, src_prev_qty = slot_to_item_state[src_slot]

    # Sanity: cannot move more than available
    if qty > src_prev_qty:
        return -1.0  # invalid move quantity

    # Destination previous qty (0 if absent or different name)
    dst_prev_name, dst_prev_qty = slot_to_item_state.get(dst_slot, (moved_name, 0))

    # In choice, determine new quantities at src and dst for the moved_name
    src_choice = slot_to_item_choice.get(src_slot, (None, 0))
    dst_choice = slot_to_item_choice.get(dst_slot, (None, 0))

    src_new_name, src_new_qty = src_choice
    dst_new_name, dst_new_qty = dst_choice

    checks = 0.0
    total_checks = 3.0

    # Check 1: destination increased by exactly qty for the moved item, and uses same item name
    if dst_new_name == moved_name and (dst_new_qty - dst_prev_qty) == qty:
        checks += 1.0

    # Check 2: source decreased by exactly qty for the moved item (or removed if reaches zero)
    # Accept if the slot is absent in choice (treated as qty 0) or present with the same name and decreased qty
    src_new_effective_qty = src_new_qty if src_new_name == moved_name else 0
    if (src_prev_qty - src_new_effective_qty) == qty:
        checks += 1.0

    # Check 3: totals for all other item names (except moved_name) must be identical BETWEEN state and choice,
    # but ignore items that are in slot '[0]' (treat these as ephemeral outputs and do not require equality)
    def totals_excluding_slot_zero(items):
        d = {}
        for name, slot, q in items:
            if slot == '[0]':
                continue  # ignore ephemeral/output slot items for totals comparison
            d[name] = d.get(name, 0) + q
        return d

    tot_state = totals_excluding_slot_zero(state_items)
    tot_choice = totals_excluding_slot_zero(choice_items)

    # allow that the moved_name totals will obviously remain the same (source - q + dst + q -> unchanged)
    # so compare totals for all other names
    other_names = set(tot_state.keys()) | set(tot_choice.keys())
    other_names.discard(moved_name)

    others_ok = all(tot_state.get(n, 0) == tot_choice.get(n, 0) for n in other_names)
    if others_ok:
        checks += 1.0

    # Map checks/3 in [0,1] to reward in [-1,1]
    reward = (checks / total_checks) * 2.0 - 1.0
    return float(reward)

# Rule 31
def rule_reward(state, action, choice):
    import re

    # parse craft target from state header e.g. "State: Craft an item of type: tripwire_hook"
    m = re.search(r'State:\s*Craft an item of type:\s*([^\n\r]+)', state, re.IGNORECASE)
    craft_target = m.group(1).strip() if m else None

    # parse move action "move: from [I4] to [A2] with quantity 1"
    m2 = re.search(r'move:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', action, re.IGNORECASE)
    if not m2:
        # This rule only applies for explicit move actions; neutral score
        return 0.0
    src_slot, dst_slot, qty = m2.group(1), m2.group(2), int(m2.group(3))

    # helper to parse inventory lists into slot->(name,count) and name totals
    def parse_items(text):
        # Expect lines like "- item_name [SLOT] quantity N"
        slot_map = {}   # slot -> (name, count)
        name_tot = {}   # name -> total count across slots
        for name, slot, qtys in re.findall(r'-\s+([^\[]+?)\s*\[([^\]]+)\]\s+quantity\s+(\d+)', text):
            nm = name.strip()
            sl = f'[{slot}]'
            qn = int(qtys)
            slot_map[sl] = (nm, qn)
            name_tot[nm] = name_tot.get(nm, 0) + qn
        return slot_map, name_tot

    state_slots, state_totals = parse_items(state)
    choice_slots, choice_totals = parse_items(choice)

    checks = 0.0
    max_checks = 4.0

    # Track which checks passed to decide whether to enforce check 4
    check1_ok = False
    check2_ok = False
    check3_ok = False

    # 1) Move: src must exist in state and be decreased by qty; allow missing in choice if reduced to 0
    if src_slot in state_slots:
        src_name, src_prev = state_slots[src_slot]
        c_src = choice_slots.get(src_slot, (src_name, 0))
        c_src_name, src_new = c_src
        # Allow the source slot to be missing in choice if quantity is 0; otherwise, names must match
        name_ok_src = (c_src_name == src_name) or (src_new == 0)
        qty_decr_ok = (src_prev - src_new) == qty
        if name_ok_src and qty_decr_ok:
            checks += 1.0
            check1_ok = True
    # else: cannot validate source -> do not award check1

    # 2) Move: destination increased by qty and name matches moved name (use source name from prior state)
    moved_name = state_slots.get(src_slot, (None, None))[0]
    st_dst = state_slots.get(dst_slot, (None, 0))
    st_dst_name, st_dst_qty = st_dst
    c_dst = choice_slots.get(dst_slot)
    if moved_name is not None and c_dst is not None:
        c_dst_name, c_dst_qty = c_dst
        name_ok_dst = (c_dst_name == moved_name)
        qty_incr_ok = (c_dst_qty - st_dst_qty) == qty
        if name_ok_dst and qty_incr_ok:
            checks += 1.0
            check2_ok = True
    # else: do not award check2

    # 3) Output slot [0] present and contains craft_target with positive produced quantity
    if craft_target:
        state_out_qty = 0
        if '[0]' in state_slots and state_slots['[0]'][0] == craft_target:
            state_out_qty = state_slots['[0]'][1]
        if '[0]' in choice_slots:
            c_out_name, c_out_qty = choice_slots['[0]']
            if c_out_name == craft_target and (c_out_qty - state_out_qty) > 0:
                checks += 1.0
                check3_ok = True
        # If craft_target not found in output, do not penalize here; simply do not award this check.
    # If craft_target not parseable, skip this check

    # 4) No unrelated changes: for all item names other than moved_name and the craft target,
    #    the total counts must be identical between state and choice.
    # Refined behavior:
    #  - Exclude moved_name and craft_target as before.
    #  - Also exclude any item names present in slot [0] in either the state or choice,
    #    because the output slot is allowed to legitimately differ (pre-existing or produced items).
    #  - Only enforce this check if the move was at least partially validated (check1 or check2).
    if (check1_ok or check2_ok):
        names_in_output = set()
        if '[0]' in state_slots:
            names_in_output.add(state_slots['[0]'][0])
        if '[0]' in choice_slots:
            names_in_output.add(choice_slots['[0]'][0])

        exclude = set()
        if moved_name:
            exclude.add(moved_name)
        if craft_target:
            exclude.add(craft_target)
        # Also exclude any names that appear in slot [0] in either state or choice
        exclude.update(names_in_output)

        def totals_excluding(totals, exclude_names):
            return {k: v for k, v in totals.items() if k not in exclude_names}

        st_others = totals_excluding(state_totals, exclude)
        ch_others = totals_excluding(choice_totals, exclude)
        if st_others == ch_others:
            checks += 1.0
    else:
        # If the move wasn't validated at all, be conservative and skip the global totals check.
        pass

    # Map checks (0..max_checks) to [-1, 1]
    score = (checks / max_checks) * 2.0 - 1.0
    # clamp
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return score

# Rule 32
def rule_reward(state, action, choice):
    import re
    from collections import defaultdict

    # parse move action "move: from [I6] to [B1] with quantity 1"
    m = re.search(r'move:\s*from\s*(\[[A-Za-z0-9]+\])\s*to\s*(\[[A-Za-z0-9]+\])\s*with\s*quantity\s*(\d+)', action)
    if not m:
        return 0.0  # rule not applicable

    src_slot, dst_slot, qty = m.group(1), m.group(2), int(m.group(3))

    # helper to parse inventory block into list of (name, slot, qty)
    def parse_items(blob):
        # Matches lines like "- granite [A1] quantity 1" or "- birch_button [0] quantity 1"
        items = []
        for name, slot, q in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', blob):
            items.append((name.strip(), f'[{slot}]', int(q)))
        return items

    state_items = parse_items(state)
    choice_items = parse_items(choice)

    # maps slot -> (name, qty)
    s_slot = {slot: (name, qty) for (name, slot, qty) in state_items}
    c_slot = {slot: (name, qty) for (name, slot, qty) in choice_items}

    # quick existence check
    if src_slot not in s_slot:
        return -1.0  # source didn't exist in state (invalid)

    moved_name, src_prev = s_slot[src_slot]
    src_new_name, src_new_qty = c_slot.get(src_slot, (None, 0))
    dst_prev_name, dst_prev_qty = s_slot.get(dst_slot, (moved_name, 0))
    dst_new_name, dst_new_qty = c_slot.get(dst_slot, (None, 0))

    score = -1.0  # start pessimistic

    # Check source decreased by qty
    if src_prev - src_new_qty != qty:
        return -1.0  # must decrease by q exactly

    # Check destination increased by qty
    if dst_new_qty - dst_prev_qty != qty:
        return -1.0  # destination must increase by q exactly

    # Destination must have same item name as moved item
    if dst_new_name != moved_name:
        return -1.0

    # Now check that no unrelated item totals changed (ignore source, destination, and slot "[0]")
    def totals(items):
        d = defaultdict(int)
        for name, slot, q in items:
            if slot == src_slot or slot == dst_slot or slot == '[0]':
                continue
            d[name] += q
        return dict(d)

    tot_state = totals(state_items)
    tot_choice = totals(choice_items)

    if tot_state != tot_choice:
        # unrelated item counts changed -> bad
        return -1.0

    # At this point src/dst are correct and unrelated totals preserved.
    # Handle output slot [0] changes permissively but check prefix correspondence.
    s_out = s_slot.get('[0]', (None, 0))
    c_out = c_slot.get('[0]', (None, 0))
    out_name_s, out_qty_s = s_out
    out_name_c, out_qty_c = c_out

    # If output slot unchanged, good
    if out_name_s == out_name_c and out_qty_s == out_qty_c:
        return 1.0

    # If output changed, allow only if output name shares prefix with moved item
    def prefix(name):
        if not name:
            return ''
        return name.split('_', 1)[0]

    if out_name_c is None:
        # output removed entirely - penalize (examples indicate output changes often matter)
        return 0.0

    if prefix(out_name_c) == prefix(moved_name):
        # plausible derived output; give positive but not full if quantities differ wildly
        return 0.8 if out_qty_c >= 0 else 0.0

    # output changed to something unrelated -> penalize
    return -1.0

# Rule 33
def rule_reward(state, action, choice):
    import re
    def parse_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns dict slot -> (name, qty) where slot includes brackets like "[I22]" or "[0]"
        items = {}
        # Match lines like: - item_name [SLOT] quantity N
        for m in re.finditer(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = m.group(1).strip()
            slot = f'[{m.group(2)}]'
            qty = int(m.group(3))
            items[slot] = (name, qty)
        return items

    # only apply this rule for move actions
    parsed = parse_action(action)
    if not parsed:
        return 0.0
    src_slot, dst_slot, move_q = parsed

    s_items = parse_items(state)
    c_items = parse_items(choice)

    # Source must exist in state
    if src_slot not in s_items:
        # invalid move reference
        return -1.0

    moved_name, src_prev_q = s_items[src_slot]
    # Check destination in state (may or may not exist)
    dst_prev_name, dst_prev_q = c_prev = s_items.get(dst_slot, (None, 0))

    # Check the move in the choice:
    # Source in choice should be decreased by move_q (or removed)
    c_src = c_items.get(src_slot)
    if c_src:
        c_src_name, c_src_q = c_src
        # name at source in choice must match moved_name (if present), and quantity must equal prev - move_q
        if c_src_name != moved_name:
            return -1.0
        if c_src_q != src_prev_q - move_q:
            return -1.0
    else:
        # absent in choice -> treated as quantity 0; require src_prev_q == move_q
        if src_prev_q != move_q:
            return -1.0

    # Destination in choice must contain the moved item and quantity must be increased by move_q
    c_dst = c_items.get(dst_slot)
    if not c_dst:
        return -1.0
    c_dst_name, c_dst_q = c_dst
    # Destination name must equal moved item
    if c_dst_name != moved_name:
        return -1.0
    # Compute original dst quantity (if dst had different name in state, we still require the dst slot in choice to be the moved item increased by move_q
    dst_state_q = s_items.get(dst_slot, (None, 0))[1]
    if c_dst_q != dst_state_q + move_q:
        return -1.0

    # Allowed slots to be modified: source slot, destination slot, any crafting-grid slots (A/B/C...), and output slot [0]
    def slot_is_crafting(slot):
        # slot string format: "[A1]" -> inside is "A1"
        inner = slot.strip('[]')
        return len(inner) >= 1 and inner[0] in ('A','B','C')

    allowed_slots = set([src_slot, dst_slot, '[0]'])
    # include all crafting-grid slots present in the original state (and choice), as they may be consumed/changed
    for slot in set(list(s_items.keys()) + list(c_items.keys())):
        if slot_is_crafting(slot):
            allowed_slots.add(slot)

    # Check that no other slot outside allowed_slots changed (name or qty)
    def slot_same(smap, cmap, slot):
        if slot in smap and slot in cmap:
            return smap[slot] == cmap[slot]
        if slot not in smap and slot not in cmap:
            return True
        # one present and other absent -> not same
        return False

    # collect all slots to check
    all_slots = set(list(s_items.keys()) + list(c_items.keys()))
    for slot in all_slots:
        if slot in allowed_slots:
            continue
        if not slot_same(s_items, c_items, slot):
            # Unallowed modification detected
            return -0.9

    # At this point the move is correct and only allowed slots were modified.
    # Base positive score
    score = 0.6

    # Bonus if output [0] was produced or changed (local craft happened)
    s_out = s_items.get('[0]')
    c_out = c_items.get('[0]')
    if c_out and (s_out != c_out):
        score += 0.35

    # Slight extra bonus if no other allowed slot except the move and output changed
    changed_allowed = []
    for slot in allowed_slots:
        if not slot_same(s_items, c_items, slot):
            changed_allowed.append(slot)
    # If only src, dst and possibly [0] changed, give small bonus
    if all(slot in (src_slot, dst_slot, '[0]') for slot in changed_allowed):
        score += 0.05

    # Clip to [-1, 1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return float(score)

# Rule 34
def rule_reward(state, action, choice):
    import re
    # Helper to parse inventories into list of (name, slot, qty)
    def parse_items(s):
        items = []
        # match lines like "- magma_cream [A1] quantity 1"
        for m in re.findall(r'-\s+([^\[]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = m[0].strip()
            slot = f'[{m[1]}]'
            qty = int(m[2])
            items.append((name, slot, qty))
        return items

    # parse move or smelt
    def parse_move(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None
        return ('move', m.group(1), m.group(2), int(m.group(3)))

    def parse_smelt(a):
        m = re.search(r'smelt:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None
        return ('smelt', m.group(1), m.group(2), int(m.group(3)))

    move_parsed = parse_move(action)
    smelt_parsed = parse_smelt(action)

    q_items = parse_items(state)
    c_items = parse_items(choice)

    # build maps: slot -> (name, qty) and name -> total qty (excluding [0])
    def build_maps(items):
        slot_map = {}
        name_tot = {}
        for name, slot, qty in items:
            slot_map[slot] = (name, qty)
            if slot != '[0]':  # track totals for non-output
                name_tot[name] = name_tot.get(name, 0) + qty
        return slot_map, name_tot

    q_slot_map, q_tot = build_maps(q_items)
    c_slot_map, c_tot = build_maps(c_items)

    # minimal flower->dye mapping to allow legitimate dye outputs for flowers
    flower_to_dye = {
        'cornflower': 'blue_dye',
        # add more mappings if known (example: 'oxeye_daisy': 'white_dye', etc.)
    }

    # Utility to check "no-other-changes" except allowed set of names
    def no_other_changes_allowed(q_totals, c_totals, allowed_names):
        # allowed_names: set of names that may change (like moved item and optionally dye)
        all_names = set(q_totals.keys()) | set(c_totals.keys())
        for name in all_names:
            if name in allowed_names:
                continue
            if q_totals.get(name, 0) != c_totals.get(name, 0):
                return False
        return True

    # Score scale: compute a score in [0,1] then map to [-1,1]
    score = 0.0
    max_checks = 3.0
    checks = 0.0

    if move_parsed:
        _, src_slot, dst_slot, qty = move_parsed
        # must have source in original
        if src_slot not in q_slot_map:
            return -1.0
        src_name, src_prev = q_slot_map[src_slot]
        dst_prev_name, dst_prev = q_slot_map.get(dst_slot, (None, 0))

        # in choice, source slot quantity decreased by qty (or absent -> 0)
        c_src_name, c_src_qty = c_slot_map.get(src_slot, (src_name, 0))
        if (src_prev - c_src_qty) == qty and (c_src_name == src_name or c_src_qty == 0):
            checks += 1.0

        # destination must have same item name as source and increased by qty
        c_dst_name, c_dst_qty = c_slot_map.get(dst_slot, (dst_prev_name, 0))
        if c_dst_name == src_name and (c_dst_qty - dst_prev) == qty:
            checks += 1.0

        # Allowed extra change: if source is a known flower, permit one output at [0] matching mapping
        allowed_names = {src_name}
        dye_allowed = False
        if src_name in flower_to_dye:
            expected_dye = flower_to_dye[src_name]
            # check slot [0] in choice was created/updated by qty and matches expected dye
            if '[0]' in c_slot_map:
                o_name, o_qty = c_slot_map['[0]']
                # in original state there might or might not be [0]; treat absent as qty 0
                q_o_qty = q_slot_map.get('[0]', (None, 0))[1]
                if o_name == expected_dye and (o_qty - q_o_qty) == qty:
                    checks += 1.0
                    dye_allowed = True
                    allowed_names.add(expected_dye)

        # disallow any other changes in non-output totals except the allowed names
        if no_other_changes_allowed(q_tot, c_tot, allowed_names):
            # small bonus if we already had the two main checks
            checks += 0.0
        else:
            # heavy penalty: changed unrelated items
            pass

        # Normalize checks: best-case checks = 3 (src decrease, dst increase+name, dye optional)
        score = checks / max_checks

    elif smelt_parsed:
        _, src_slot, dst_slot, qty = smelt_parsed
        # require source present
        if src_slot not in q_slot_map:
            return -1.0
        src_name, src_prev = q_slot_map[src_slot]
        dst_prev_name, dst_prev = q_slot_map.get(dst_slot, (None, 0))

        # source reduced by qty
        c_src_name, c_src_qty = c_slot_map.get(src_slot, (src_name, 0))
        if (src_prev - c_src_qty) == qty:
            checks += 1.0

        # destination increased by qty
        c_dst_name, c_dst_qty = c_slot_map.get(dst_slot, (dst_prev_name, 0))
        if (c_dst_qty - dst_prev) == qty:
            checks += 1.0

        # ensure these are the only changes in totals (destination name may differ)
        allowed_names = {src_name, c_dst_name}
        if no_other_changes_allowed(q_tot, c_tot, allowed_names):
            checks += 1.0

        score = checks / 3.0

    else:
        # not a move or smelt action -> no rule applied; neutral score 0
        return 0.0

    # Map [0,1] -> [-1,1]
    return max(-1.0, min(1.0, 2.0 * score - 1.0))

# Rule 35
def rule_reward(state, action, choice):
    import re

    # parse craft target from state header: "State: Craft an item of type: <target>"
    m_target = re.search(r'State:\s*Craft an item of type:\s*([^\n\r]+)', state)
    target = m_target.group(1).strip() if m_target else None

    # parse move action: "move: from [S] to [D] with quantity q"
    m_act = re.search(r'move:\s*from\s*(\[[A-Za-z0-9]+\])\s*to\s*(\[[A-Za-z0-9]+\])\s*with\s*quantity\s*(\d+)', action)
    if not m_act:
        return 0.0  # rule only applies to the observed pattern

    src_slot, dst_slot, qty = m_act.group(1), m_act.group(2), int(m_act.group(3))

    # parse inventory lists into list of tuples (name, slot, qty)
    def parse_items(s):
        items = []
        # Matches lines like: - name [I6] quantity 3
        for name, slot, num in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            items.append((name.strip(), f'[{slot}]', int(num)))
        return items

    state_items = parse_items(state)
    choice_items = parse_items(choice)

    # helper maps
    def slot_map(items):
        return {slot: (name, qty) for (name, slot, qty) in items}

    s_map = slot_map(state_items)
    c_map = slot_map(choice_items)

    # 1) moved item must exist in source slot in original state
    if src_slot not in s_map:
        return -1.0
    moved_name, src_prev = s_map[src_slot]

    # 2) source in choice: same name (or absent) and decreased by qty
    c_src = c_map.get(src_slot)
    if c_src is None:
        src_new_qty = 0
        src_new_name = None
    else:
        src_new_name, src_new_qty = c_src

    # name should be same if present
    if src_new_name is not None and src_new_name != moved_name:
        return -1.0
    if src_prev - src_new_qty != qty:
        return -1.0

    # 3) destination in original state and choice: name must be moved_name
    dst_prev = 0
    dst_prev_name = None
    if dst_slot in s_map:
        dst_prev_name, dst_prev = s_map[dst_slot]
    # In many tasks destination before could be empty; allow that (treat as 0).
    c_dst = c_map.get(dst_slot)
    if c_dst is None:
        return -1.0  # destination must appear after move
    dst_new_name, dst_new_qty = c_dst
    # destination name must equal moved_name
    if dst_new_name != moved_name:
        return -1.0
    if dst_new_qty - dst_prev != qty:
        return -1.0

    # 4) For all non-[0], non-src, non-dst slots, totals by item must be unchanged.
    def totals_excluding_zero_src_dst(items):
        d = {}
        for name, slot, qty_ in items:
            if slot in (src_slot, dst_slot, '[0]'):
                continue
            d[name] = d.get(name, 0) + qty_
        return d

    t_state = totals_excluding_zero_src_dst(state_items)
    t_choice = totals_excluding_zero_src_dst(choice_items)
    if t_state != t_choice:
        return -1.0

    # 5) Check for new item names in non-[0] slots: there should be none.
    state_names_non0 = set(name for name, slot, _ in state_items if slot != '[0]')
    choice_names_non0 = set(name for name, slot, _ in choice_items if slot != '[0]')
    # allow moved_name presence; both sets should be equal (except moved_name may move slots)
    # but we've already enforced totals; a straightforward check:
    # any name in choice non-[0] not in state non-[0] -> bad
    new_non0 = choice_names_non0 - state_names_non0
    if new_non0:
        return -1.0

    # 6) Disallow introducing the craft target name as a new item anywhere unless it was already present in state
    if target:
        state_has_target = any(name == target for name, _, _ in state_items)
        choice_has_target = any(name == target for name, _, _ in choice_items)
        if choice_has_target and not state_has_target:
            # introduced target name -> penalize
            return -1.0

    # If all checks pass, give a positive reward
    return 1.0

# Rule 36
def rule_reward(state, action, choice):
    import re

    # parse move action "move: from [I9] to [A1] with quantity 1"
    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, 0
        return m.group(1), m.group(2), int(m.group(3))

    # parse inventory lines like "- beetroot [I9] quantity 1"
    def parse_items(s):
        items = []
        # allow names that can include hyphens/underscores and spaces; slot inside [...]
        for m in re.finditer(r'-\s+(.+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = m.group(1).strip()
            slot = f'[{m.group(2)}]'
            qty = int(m.group(3))
            items.append((name, slot, qty))
        return items

    src_slot, dst_slot, qty = parse_move_action(action)
    if not src_slot:
        # rule applies only to move actions; return neutral small negative
        return -0.2

    s_items = parse_items(state)
    c_items = parse_items(choice)

    s_slot_map = {slot: (name, q) for (name, slot, q) in s_items}
    c_slot_map = {slot: (name, q) for (name, slot, q) in c_items}

    # The moved item must exist at source in the state
    if src_slot not in s_slot_map:
        return -1.0

    moved_name, src_prev_qty = s_slot_map[src_slot]
    # previous dest quantity and name (if any)
    dst_prev_name, dst_prev_qty = c_slot_map.get(dst_slot, (None, 0))
    # In state, destination might exist too
    dst_prev_in_state = s_slot_map.get(dst_slot, (None, 0))[1] if dst_slot in s_slot_map else 0
    dst_prev_name_in_state = s_slot_map.get(dst_slot, (None, 0))[0] if dst_slot in s_slot_map else None

    # Checks to perform (4 checks)
    checks_passed = 0
    total_checks = 4

    # 1) Destination increased by qty and has the same item name
    # Determine destination quantities in state and choice
    state_dst_qty = s_slot_map.get(dst_slot, (None, 0))[1]
    state_dst_name = s_slot_map.get(dst_slot, (None, None))[0]
    choice_dst_qty = c_slot_map.get(dst_slot, (None, 0))[1]
    choice_dst_name = c_slot_map.get(dst_slot, (None, None))[0]

    if choice_dst_name == moved_name and (choice_dst_qty - state_dst_qty) == qty:
        checks_passed += 1

    # 2) Source decreased by qty (and name either same or slot removed)
    choice_src_entry = c_slot_map.get(src_slot)
    if choice_src_entry is None:
        # slot removed -> acceptable if src_prev_qty == qty
        if src_prev_qty == qty:
            checks_passed += 1
    else:
        choice_src_name, choice_src_qty = choice_src_entry
        if choice_src_name == moved_name and (src_prev_qty - choice_src_qty) == qty:
            checks_passed += 1

    # 3) No unrelated item total-count changes (ignore moved_name and slot '[0]' changes)
    def totals(items):
        d = {}
        for name, slot, count in items:
            if name == moved_name:
                continue
            if slot == '[0]':
                continue
            d[name] = d.get(name, 0) + count
        return d

    tot_state = totals(s_items)
    tot_choice = totals(c_items)
    # If all names and counts equal, pass
    if all(tot_state.get(k, 0) == tot_choice.get(k, 0) for k in set(tot_state) | set(tot_choice)):
        checks_passed += 1

    # 4) Validate changes to slot [0]: allowed conversions only
    # Allowed conversion map derived from observed cases
    conversion_map = {
        'beetroot': 'red_dye',
        'lily_of_the_valley': 'white_dye'
    }
    # previous and new [0]
    state_zero = s_slot_map.get('[0]', (None, 0))
    choice_zero = c_slot_map.get('[0]', (None, 0))
    zero_prev_name, zero_prev_qty = state_zero
    zero_new_name, zero_new_qty = choice_zero

    zero_changed = (zero_prev_name != zero_new_name) or (zero_prev_qty != zero_new_qty)

    if not zero_changed:
        # no change to [0] is acceptable
        checks_passed += 1
    else:
        # if changed, must be allowed conversion for moved_name and quantity change must match qty
        expected = conversion_map.get(moved_name)
        if expected is not None:
            # either the [0] was absent and now equals qty of expected, or increased by qty
            prev_q = zero_prev_qty if zero_prev_name == expected else 0
            if zero_new_name == expected and (zero_new_qty - prev_q) == qty:
                checks_passed += 1
            else:
                # fail this check
                pass
        else:
            # moved item has no permitted conversion, so any change is invalid
            pass

    # Map fraction of passed checks to [-1, 1]
    score_frac = checks_passed / total_checks
    return score_frac * 2.0 - 1.0

# Rule 37
def rule_reward(state, action, choice):
    import re

    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z0-9]+\])\s*to\s*(\[[A-Z0-9]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns dict slot -> (name, qty)
        items = {}
        # match lines like: - item_name [SLOT] quantity N
        for m in re.finditer(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = m.group(1).strip()
            slot = '[' + m.group(2) + ']'
            qty = int(m.group(3))
            items[slot] = (name, qty)
        return items

    # parse action
    parsed = parse_move_action(action)
    if not parsed:
        # This rule only applies to move actions
        return 0.0
    src_slot, dst_slot, qty = parsed

    # parse states
    src_items = parse_items(state)
    ch_items = parse_items(choice)

    # source must exist in prior state
    if src_slot not in src_items:
        # invalid action reference
        return -1.0
    moved_name, src_prev_qty = src_items[src_slot]

    # destination previous qty and name (may be absent)
    dst_prev = src_items.get(dst_slot, (None, 0))[1]

    # destination in choice must exist and be the moved item
    if dst_slot not in ch_items:
        return -1.0
    dst_name_ch, dst_qty_ch = ch_items[dst_slot]
    if dst_name_ch != moved_name:
        return -1.0
    if (dst_qty_ch - dst_prev) != qty:
        return -1.0

    # source in choice: allowed to be absent (interpreted as zero) or present with same name
    if src_slot in ch_items:
        src_name_ch, src_qty_ch = ch_items[src_slot]
        if src_name_ch != moved_name:
            return -1.0
        if (src_prev_qty - src_qty_ch) != qty:
            return -1.0
        if src_qty_ch < 0:
            return -1.0
    else:
        # source absent: must have gone to zero
        if src_prev_qty - qty != 0:
            return -1.0

    # Now ensure no other slot/item changed
    for slot, (name_prev, qty_prev) in src_items.items():
        if slot == src_slot or slot == dst_slot:
            continue
        if slot not in ch_items:
            return -1.0
        name_ch, qty_ch = ch_items[slot]
        if name_ch != name_prev or qty_ch != qty_prev:
            return -1.0

    # Also ensure choice has no extra slots beyond allowed ones
    for slot, (name_ch, qty_ch) in ch_items.items():
        if slot == src_slot or slot == dst_slot:
            continue
        if slot not in src_items:
            # new slot introduced -> disallow
            return -1.0

    # All checks passed
    return 1.0

# Rule 38
def rule_reward(state, action, choice):
    """
    Returns a float in [-1, 1]. Positive means likely correct, negative means likely incorrect.
    Heavily rewards exact move-only changes; penalizes creation/changes of unrelated items (including [0]).
    """
    import re
    from collections import defaultdict

    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Za-z0-9]+\])\s*to\s*(\[[A-Za-z0-9]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None
        return m.group(1), m.group(2), int(m.group(3))

    def parse_items(s):
        # returns dict slot -> (name, qty)
        items = {}
        # pattern handles names that may include spaces and slot in square brackets
        for m in re.finditer(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = m.group(1).strip()
            slot = f'[{m.group(2)}]'
            qty = int(m.group(3))
            items[slot] = (name, qty)
        return items

    mv = parse_move_action(action)
    if mv is None:
        # not a move action: rule not applicable -> neutral 0.0
        return 0.0
    src_slot, dst_slot, q = mv

    state_items = parse_items(state)
    choice_items = parse_items(choice)

    # Source must exist in state
    if src_slot not in state_items:
        # can't validate move if source doesn't exist; negative signal
        return -1.0

    moved_name, src_prev_qty = state_items[src_slot]

    # In choice, source slot may be missing or present
    src_choice = choice_items.get(src_slot, (None, 0))
    src_choice_name, src_new_qty = src_choice

    # Destination previous and new
    dst_prev_name, dst_prev_qty = choice_items.get(dst_slot, (None, 0))
    # But we should get dst_prev from original state (destination may be new)
    dst_prev_name_state, dst_prev_qty_state = state_items.get(dst_slot, (None, 0))
    dst_choice = choice_items.get(dst_slot, (None, 0))
    dst_choice_name, dst_new_qty = dst_choice

    # Basic checks: names must match moved_name
    # Source in choice should either have same name or be absent (treated as name moved_name)
    # Destination in choice must have the moved_name
    # If names mismatch -> strongly penalize
    # Determine actual source/destination names in choice
    # Check name consistency
    if src_choice_name is not None and src_choice_name != moved_name:
        return -1.0  # source slot holds a different item in choice -> wrong
    if dst_choice_name is not None and dst_choice_name != moved_name:
        return -1.0  # destination slot holds a different item in choice -> wrong

    # Check exact quantities for move
    expected_src_new = src_prev_qty - q
    expected_dst_new = dst_prev_qty_state + q

    move_ok = (src_new_qty == expected_src_new) and (dst_new_qty == expected_dst_new)

    # Compute unrelated changes: for all slots other than src_slot and dst_slot,
    # sum absolute differences in quantities. Also treat name changes as differences.
    unrelated_diff = 0
    total_qty_state = 0
    for slot, (name_s, qty_s) in state_items.items():
        total_qty_state += qty_s
    # include choice-only slots in total for normalization
    for slot, (name_c, qty_c) in choice_items.items():
        total_qty_state += 0  # leave unchanged; we only need a normalizer >0

    # Sum differences
    for slot in set(list(state_items.keys()) + list(choice_items.keys())):
        if slot in (src_slot, dst_slot):
            continue
        s_entry = state_items.get(slot)
        c_entry = choice_items.get(slot)
        if s_entry is None and c_entry is None:
            continue
        if s_entry is None:
            # new slot created in choice -> count as difference (full qty)
            unrelated_diff += c_entry[1]
            continue
        if c_entry is None:
            # slot removed in choice -> count as difference (full qty)
            unrelated_diff += s_entry[1]
            continue
        # both exist; if name changed, count both amounts as difference
        if s_entry[0] != c_entry[0]:
            unrelated_diff += s_entry[1] + c_entry[1]
        else:
            unrelated_diff += abs(s_entry[1] - c_entry[1])

    # Special penalty: any change in slot '[0]' (creation or qty change) is suspect for move action.
    zero_slot_change = 0
    zero_slot = '[0]'
    if zero_slot in state_items or zero_slot in choice_items:
        s0 = state_items.get(zero_slot, (None, 0))[1]
        c0 = choice_items.get(zero_slot, (None, 0))[1]
        if s0 != c0:
            zero_slot_change = abs(s0 - c0)

    # Scoring policy:
    # - If move is exact and no unrelated changes and no [0] changes -> +1.0
    # - If move is exact but unrelated changes or [0] changes -> 0.0 (allowed but suspicious)
    # - If move partially correct (quantities off) -> -0.5
    # - If name mismatches or other severe violations -> -1.0

    if move_ok and unrelated_diff == 0 and zero_slot_change == 0:
        return 1.0
    if move_ok and (unrelated_diff > 0 or zero_slot_change > 0):
        # move performed correctly but extra unrelated changes -> penalize but not as severely as wrong move
        # map unrelated_diff and zero_slot_change to [0.0 .. -1.0) linearly; here return 0.0 as neutral/slightly negative
        return 0.0
    # move not exact: give negative score
    # compute how far off the move is in magnitude relative to expected q
    move_src_diff = abs(src_new_qty - expected_src_new)
    move_dst_diff = abs(dst_new_qty - expected_dst_new)
    move_error = move_src_diff + move_dst_diff
    if move_error > 0:
        # partially wrong move
        return -0.5
    return -1.0

# Rule 39
def rule_reward(state, action, choice):
    import re

    # parse a move action: return (src_slot, dst_slot, qty) or (None,None,0) if not a move
    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[^\]]+\])\s*to\s*(\[[^\]]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, 0
        return m.group(1), m.group(2), int(m.group(3))

    # parse inventory text into dict slot -> (name, qty)
    def parse_items(text):
        # find lines like "- item_name [SLOT] quantity N"
        items = {}
        for name, slot, qty in re.findall(r'-\s+([^\[\n]+?)\s*\[([^\]]+)\]\s*quantity\s*(\d+)', text):
            slot_br = f'[{slot}]'
            items[slot_br] = (name.strip(), int(qty))
        return items

    src, dst, q = parse_move_action(action)
    # if not a move action, this rule does not apply strongly; return neutral 0.0
    if src is None:
        return 0.0

    s_items = parse_items(state)
    c_items = parse_items(choice)

    # If source missing in original state, invalid move
    if src not in s_items:
        return -1.0

    moved_name, src_prev = s_items[src]
    # get previous dst quantity if existed
    dst_prev_name, dst_prev_qty = c_prev = (None, 0)
    if dst in s_items:
        dst_prev_name, dst_prev_qty = s_items[dst]

    # get choice values (if slot absent in choice then treat as removed)
    src_new_name, src_new_qty = c_items.get(src, (None, 0))
    dst_new_name, dst_new_qty = c_items.get(dst, (None, 0))

    score = 0.0
    checks = 0.0

    # 1) Source decreased by exactly q and name unchanged (or removed)
    checks += 1.0
    if (src_new_name in (None, moved_name)) and (src_prev - src_new_qty == q):
        score += 1.0

    # 2) Destination increased by exactly q and matches moved_name
    checks += 1.0
    # If destination existed previously with a different name, that's invalid.
    if dst in s_items:
        if dst_prev_name != moved_name:
            # If dst had different name before, moving into dst would overwrite — disallow
            pass
        else:
            if dst_new_name == moved_name and (dst_new_qty - dst_prev_qty == q):
                score += 1.0
    else:
        # dst didn't exist before — allowed only if dst now contains moved_name with qty == q
        if dst_new_name == moved_name and dst_new_qty == q:
            score += 1.0

    # 3) No other item changes (except source and destination). Also disallow any modification of slot [0].
    checks += 1.0
    ok = True
    # check slot [0] presence: must not be newly added or changed
    zero_slot = '[0]'
    if zero_slot in s_items:
        # If existed before, choice must have same name and qty for [0]
        if zero_slot not in c_items or c_items[zero_slot] != s_items[zero_slot]:
            ok = False
    else:
        # if it did not exist before, it must not be added now
        if zero_slot in c_items:
            ok = False

    # Check all other slots unchanged
    for slot, (name, qty) in s_items.items():
        if slot in (src, dst, zero_slot):
            continue
        centry = c_items.get(slot)
        if centry is None:
            # removal of unrelated slot is not allowed
            ok = False
            break
        if centry[0] != name or centry[1] != qty:
            ok = False
            break

    # Also ensure no extra new slots (other than dst) appear in choice
    for slot, (name, qty) in c_items.items():
        if slot in (src, dst, zero_slot):
            continue
        if slot not in s_items:
            # new slot other than dst not allowed
            ok = False
            break

    if ok:
        score += 1.0

    # map score/checks to [-1, 1]
    if checks == 0:
        return 0.0
    fraction = score / checks  # in [0,1]
    # convert to [-1,1]: fraction 1 -> +1, fraction 0 -> -1, linear mapping
    return fraction * 2.0 - 1.0

# Rule 40
def rule_reward(state, action, choice):
    import re
    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z0-9]+\])\s*to\s*(\[[A-Z0-9]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, 0
        return m.group(1), m.group(2), int(m.group(3))

    def parse_goal(s):
        m = re.search(r'Craft an item of type:\s*([^\n\r]+)', s)
        return m.group(1).strip() if m else None

    def parse_items(s):
        # returns dict slot -> (name, qty)
        items = {}
        for name, slot, qty in re.findall(r'-\s+([^\[]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            items[f'[{slot}]'] = (name.strip(), int(qty))
        return items

    # parse inputs
    src_slot, dst_slot, qty = parse_move_action(action)
    if not src_slot:
        # rule only applies to move actions in these examples
        return 0.0

    goal = parse_goal(state)

    state_items = parse_items(state)
    choice_items = parse_items(choice)

    # ensure source existed in the state
    if src_slot not in state_items:
        return -1.0

    moved_name, src_prev = state_items[src_slot]
    # get choice counts for src/dst (default to 0 and possibly no name)
    dst_prev = state_items.get(dst_slot, (None, 0))[1]
    src_choice = choice_items.get(src_slot, (None, 0))
    dst_choice = choice_items.get(dst_slot, (None, 0))

    src_choice_name, src_new = src_choice
    dst_choice_name, dst_new = dst_choice

    # Condition A: destination increased by qty and holds the same item name
    cond_a = (dst_choice_name == moved_name) and ((dst_new - dst_prev) == qty)

    # Condition B: source decreased by qty and either retains same name or removed (qty 0)
    # If slot removed entirely in choice, treat as 0
    src_prev_qty = src_prev
    src_new_qty = src_new if src_choice_name is not None else 0
    # If name at source in choice is present, ensure it matches moved_name (or allow missing when fully consumed)
    name_ok = (src_choice_name == moved_name) or (src_choice_name is None)
    cond_b = name_ok and ((src_prev_qty - src_new_qty) == qty)

    # Condition C: no unrelated item total-count changes (ignore moved_name and ignore slot [0])
    def totals(items):
        d = {}
        for slot, (name, count) in items.items():
            if slot == '[0]':
                continue
            if name == moved_name:
                continue
            d[name] = d.get(name, 0) + count
        return d
    totals_state = totals(state_items)
    totals_choice = totals(choice_items)
    cond_c = all(totals_state.get(k, 0) == totals_choice.get(k, 0) for k in set(totals_state) | set(totals_choice))

    # Condition D (penalty): goal item must not be newly introduced/increased at slot [0]
    goal_penalty = 0.0
    if goal:
        state_zero = state_items.get('[0]', (None, 0))
        choice_zero = choice_items.get('[0]', (None, 0))
        state_zero_name, state_zero_qty = state_zero
        choice_zero_name, choice_zero_qty = choice_zero
        # If choice has [0] equal to goal and either [0] was absent before or its qty increased, penalize
        if choice_zero_name == goal:
            prev_qty = state_zero_qty if state_zero_name == goal else 0
            if choice_zero_qty > prev_qty:
                goal_penalty = 1.0

    # compute base score from A,B,C
    checks = 0
    checks += 1 if cond_a else 0
    checks += 1 if cond_b else 0
    checks += 1 if cond_c else 0
    base = checks / 3.0  # in [0,1]

    # subtract penalty if goal was introduced/increased at [0]
    score = base - goal_penalty

    # map to [-1,1] and clamp
    # base in [0,1], penalty can make it [-1,1]; scale: (score * 1.0) then clamp
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return float(score)

# Rule 41
def rule_reward(state, action, choice):
    import re
    def parse_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None
        return m.group(1), m.group(2), int(m.group(3))
    def parse_items(s):
        # returns dict slot_id -> (name, qty), and name->list of (slot, qty)
        slot_map = {}
        name_map = {}
        for name, slot, qty in re.findall(r'-\s+([^\[]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            nm = name.strip()
            sid = slot.strip()
            q = int(qty)
            slot_map[sid] = (nm, q)
            name_map.setdefault(nm, []).append((sid, q))
        return slot_map, name_map

    # parse
    act = parse_action(action)
    if not act:
        return 0.0
    src_slot_br, dst_slot_br, qty = act
    # strip brackets for keys used in parse_items
    src_slot = src_slot_br.strip('[]')
    dst_slot = dst_slot_br.strip('[]')

    s_slot_map, s_name_map = parse_items(state)
    c_slot_map, c_name_map = parse_items(choice)

    # Helper: get slot name and quantity (0 if missing)
    def slot_item(slot_map, sid):
        return slot_map.get(sid, (None, 0))

    # Check move consistency: source decreased by qty and destination increased by qty for same item name
    src_name, src_prev = slot_item(s_slot_map, src_slot)
    dst_name, dst_prev = slot_item(s_slot_map, dst_slot)
    # After move in choice:
    src_new_name, src_new = slot_item(c_slot_map, src_slot)
    dst_new_name, dst_new = slot_item(c_slot_map, dst_slot)

    # If source didn't have the moved item originally, cannot validate; return modest negative
    if src_name is None:
        return -0.6

    move_ok = False
    # Expect destination to now contain the same item as src_name, and amounts to change by qty
    if dst_new_name == src_name and dst_new - dst_prev == qty:
        # source decreased by qty (or became zero/missing)
        # allow src_new_name to be same or missing; compute src_prev - src_new == qty
        if (src_prev - src_new) == qty:
            move_ok = True

    # If move is incorrect, heavy penalty
    if not move_ok:
        return -0.9

    # Now detect recipes that should trigger output at slot [0]
    # Check for log -> planks rule
    moved_name = src_name
    reward = 0.0
    applied_recipe = False

    if moved_name.endswith('_log'):
        applied_recipe = True
        expected_planks = moved_name[:-4] + '_planks'  # replace suffix
        expected_qty = 4 * qty
        out_name, out_qty = slot_item(c_slot_map, '0')
        # Good if output slot contains expected planks with expected quantity
        if out_name == expected_planks and out_qty == expected_qty:
            reward = 1.0
        else:
            # If output contains some other item, penalize; if missing, penalize
            reward = -0.9
        # return clamped
        return max(-1.0, min(1.0, reward))

    # Check for bowl + beetroot -> beetroot_soup among crafting slots (A/B/C)
    # Build set of item names present in crafting slots after move
    craft_items = set()
    for sid, (nm, q) in c_slot_map.items():
        if len(sid) >= 1 and sid[0] in ('A', 'B', 'C'):
            if q > 0 and nm:
                craft_items.add(nm)
    # Also consider the moved destination if somehow not parsed under A/B/C (defensive)
    if len(dst_slot) >= 1 and dst_slot[0] in ('A', 'B', 'C'):
        # already included
        pass

    if 'bowl' in craft_items and 'beetroot' in craft_items:
        applied_recipe = True
        out_name, out_qty = slot_item(c_slot_map, '0')
        if out_name == 'beetroot_soup' and out_qty >= 1:
            reward = 1.0
        else:
            reward = -0.9
        return max(-1.0, min(1.0, reward))

    # If no recipe applies, reward positive for correct move representation
    if not applied_recipe:
        return 0.6  # move_ok already validated, so modest positive

    return 0.0

# Rule 42
def rule_reward(state, action, choice):
    import re

    # parse move action: move: from [I?] to [A?] with quantity q
    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z]\d+\])\s*to\s*(\[[A-Z]\d+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, None
        return m.group(1), m.group(2), int(m.group(3))

    # parse items: returns list of (name, slot, qty)
    def parse_items(s):
        items = []
        # match "- name [SLOT] quantity N" where name can include hyphens/spaces
        for m in re.finditer(r'-\s+(.+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = m.group(1).strip()
            slot = f'[{m.group(2)}]'
            qty = int(m.group(3))
            items.append((name, slot, qty))
        return items

    src_slot, dst_slot, qty = parse_move_action(action)
    # only apply this rule for relevant move-to-A craft steps
    if not src_slot or not dst_slot or not dst_slot.startswith('[A'):
        return 0.0

    q_items = parse_items(state)
    c_items = parse_items(choice)

    # build dictionaries slot -> (name, qty) and totals by name (excluding slot [0])
    def by_slot(items):
        d = {}
        for name, slot, qty in items:
            d[slot] = (name, qty)
        return d

    def totals_excluding(items, exclude_names=set(), exclude_slots=set()):
        t = {}
        for name, slot, qty in items:
            if slot in exclude_slots:
                continue
            if name in exclude_names:
                continue
            t[name] = t.get(name, 0) + qty
        return t

    q_slot = by_slot(q_items)
    c_slot = by_slot(c_items)

    # source must exist in the original state
    if src_slot not in q_slot:
        return -1.0

    moved_name, src_prev = q_slot[src_slot]
    # destination previous quantity (may be 0 if not present)
    dst_prev = q_slot.get(dst_slot, (moved_name, 0))[1]

    # destination in choice
    dst_new_name, dst_new = c_slot.get(dst_slot, (None, 0))
    # source in choice (may be absent)
    src_new_name, src_new = c_slot.get(src_slot, (moved_name, 0))

    # Check 1: move reflected correctly
    move_ok = False
    if dst_new_name == moved_name and (dst_new - dst_prev) == qty:
        # source decreased by qty (allow source to disappear)
        if (src_prev - src_new) == qty:
            move_ok = True

    # Check 2: output [0] exists with positive quantity
    out_name_qty = c_slot.get('[0]', None)
    output_ok = out_name_qty is not None and out_name_qty[1] > 0

    # Check 3: unrelated item totals unchanged
    # exclude moved_name and slot [0] from totals
    q_tot = totals_excluding(q_items, exclude_names={moved_name}, exclude_slots={'[0]'})
    c_tot = totals_excluding(c_items, exclude_names={moved_name}, exclude_slots={'[0]'})
    unrelated_ok = (q_tot == c_tot)

    # Aggregate a points score in [0,1], emphasizing presence of output and correct move.
    move_w = 0.3
    output_w = 0.6
    unrelated_w = 0.1

    p = 0.0
    if move_ok:
        p += move_w
    if output_ok:
        p += output_w
    if unrelated_ok:
        p += unrelated_w

    # clamp p
    if p < 0:
        p = 0.0
    if p > 1.0:
        p = 1.0

    # map to [-1, 1]
    score = 2.0 * p - 1.0
    return float(score)

# Rule 43
def rule_reward(state, action, choice):
    import re
    from collections import defaultdict

    def parse_items(s):
        # returns dict slot -> (name, qty) and name->total for inventory slots
        slot_map = {}
        for m in re.findall(r'-\s+([^\[\n]+?)\s+\[([^\]]+)\]\s+quantity\s+(\d+)', s):
            name = m[0].strip()
            slot = f'[{m[1]}]'
            qty = int(m[2])
            slot_map[slot] = (name, qty)
        return slot_map

    def parse_move_action(a):
        m = re.search(r'move:\s*from\s*(\[[A-Z0-9]+\])\s*to\s*(\[[A-Z0-9]+\])\s*with\s*quantity\s*(\d+)', a)
        if not m:
            return None, None, None
        return m.group(1), m.group(2), int(m.group(3))

    # Only apply this rule for craft operations that move from [0] to an inventory slot
    if "Craft an item of type:" not in state:
        return 0.0

    src_slot, dst_slot, qty = parse_move_action(action)
    if not src_slot or not dst_slot:
        return 0.0

    # We focus on moves from [0] to an inventory slot [I...]
    if src_slot != '[0]' or not re.match(r'\[I\d+\]', dst_slot):
        return 0.0

    state_items = parse_items(state)
    choice_items = parse_items(choice)

    # If there was no item at source in state, cannot verify
    if src_slot not in state_items:
        return -1.0

    moved_name, src_prev = state_items[src_slot]
    dst_prev = state_items.get(dst_slot, (moved_name, 0))[1]
    # In choice, get new source/dest counts (0 if absent)
    src_new = choice_items.get(src_slot, (moved_name, 0))[1]
    dst_new = choice_items.get(dst_slot, (moved_name, 0))[1]

    # 1) Move correctness checks
    move_points = 0.0
    # destination increased by exactly qty and name matches
    dst_name_ok = (choice_items.get(dst_slot, (None, 0))[0] == moved_name)
    if dst_name_ok and (dst_new - dst_prev) == qty:
        move_points += 0.6  # strong reward for correct destination change
    # source decreased by qty (or disappeared)
    if src_prev - src_new == qty:
        move_points += 0.4

    # Partial credit if only one of the above held
    # move_points is in [0,1] now.

    # 2) Ingredient consumption: slots like [A1], [B1], [C1], etc. (capital letter prefix not 'I')
    ing_slots = [s for s in state_items.keys() if re.match(r'\[[A-HJ-Z]\d+\]', s)]
    ing_points = 1.0
    if ing_slots:
        successes = 0
        for s in ing_slots:
            name, prev_q = state_items[s]
            new_q = choice_items.get(s, (name, 0))[1]
            # success if decreased or removed
            if new_q < prev_q:
                successes += 1
        ing_points = successes / len(ing_slots)  # fraction in [0,1]

    # 3) Unrelated inventory stability: inventory slots [I#] except dst_slot and excluding moved_name should not change totals
    def inventory_totals(items):
        d = defaultdict(int)
        for slot, (name, q) in items.items():
            if re.match(r'\[I\d+\]', slot):
                d[name] += q
        return dict(d)

    st_inv = inventory_totals(state_items)
    ch_inv = inventory_totals(choice_items)

    # compute differences for inventory items excluding the moved item and ignoring destination's expected change
    penalty_flag = 0
    for name in set(st_inv.keys()) | set(ch_inv.keys()):
        if name == moved_name:
            # allow the moved_name to change by exactly qty at destination; compute net expected
            expected = st_inv.get(name, 0)
            # The moved item total across inventory may legitimately increase by qty (if moved into inventory),
            # but we don't have exact recipe outputs; we treat arbitrary changes for other names only.
            continue
        # any change in counts for other inventory items is penalized
        if st_inv.get(name, 0) != ch_inv.get(name, 0):
            penalty_flag = 1
            break

    # Compose final score: weight move higher, require ingredient consumption if present, penalize unrelated changes
    move_score = move_points  # in [0,1]
    ingredient_score = ing_points  # in [0,1]

    base = 0.6 * move_score + 0.4 * ingredient_score  # in [0,1]
    if penalty_flag:
        base -= 0.7  # strong penalty for changing unrelated inventory items

    # Clamp to [-1,1]
    if base > 1.0:
        base = 1.0
    if base < -1.0:
        base = -1.0

    return float(base)

# Rule 44
def rule_reward(state, action, choice):
    import re

    def parse_action(a):
        # expects: move: from [X] to [Y] with quantity N
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

    # parse inputs
    src_slot, dst_slot, qty = parse_action(action)
    prior = parse_items(state)
    cand = parse_items(choice)

    # Only apply this rule when action is a move from [0] (cursor) to an inventory slot [I...]
    if src_slot is None:
        return 0.0
    if src_slot != '[0]':
        return 0.0
    if not re.match(r'\[I\d+\]', dst_slot):
        return 0.0

    # Identify the moved item in the prior state
    if src_slot not in prior:
        # nothing to move -> strongly negative
        return -1.0
    moved_name, src_prev_qty = prior[src_slot]

    # destination previous quantity (0 if absent)
    dst_prev_qty = prior.get(dst_slot, (moved_name, 0))[1]

    # Checks
    score = 0.0
    max_score = 1.0

    # 1) Destination correctness: name matches and quantity increased exactly by qty
    dst_in_cand = cand.get(dst_slot)
    if dst_in_cand is not None:
        dst_name_c, dst_qty_c = dst_in_cand
        if dst_name_c == moved_name and (dst_qty_c - dst_prev_qty) == qty:
            score += 0.5
        else:
            # partial credit if name matches but qty mismatch
            if dst_name_c == moved_name:
                score += 0.2
            else:
                score -= 0.6
    else:
        # destination missing -> bad
        score -= 0.8

    # 2) Source removal: [0] must be absent in candidate
    if '[0]' not in cand:
        score += 0.2
    else:
        # if present but quantity decreased by qty exactly and not leftover, still penalize (should be removed)
        cand_name0, cand_qty0 = cand['[0]']
        if cand_name0 == moved_name and (src_prev_qty - cand_qty0) == qty and cand_qty0 == 0:
            score += 0.05
        else:
            score -= 0.4

    # 3) Clear A-slots: any [A\d+] present in prior should be absent in candidate
    a_slots_in_prior = [s for s in prior.keys() if re.match(r'\[A\d+\]', s)]
    a_cleared = True
    for a in a_slots_in_prior:
        if a in cand:
            a_cleared = False
            break
    if a_slots_in_prior:
        if a_cleared:
            score += 0.2
        else:
            score -= 0.6  # significant penalty when inputs are not consumed

    # 4) No unrelated changes: other slots must keep same (name and qty)
    # allowed exceptions: src_slot '[0]' (handled), dst_slot (handled), any A slots (allowed removed)
    unrelated_penalty = 0.0
    for slot, (name, qty_prior) in prior.items():
        if slot in (src_slot, dst_slot):
            continue
        if re.match(r'\[A\d+\]', slot):
            continue
        # must exist in candidate with same name and qty
        if slot not in cand:
            unrelated_penalty += 0.6  # missing unrelated slot
        else:
            name_c, qty_c = cand[slot]
            if name_c != name or qty_c != qty_prior:
                unrelated_penalty += 0.8

    # also penalize any new unrelated slots introduced in candidate (except dst)
    for slot in cand.keys():
        if slot in (dst_slot, src_slot):
            continue
        if re.match(r'\[A\d+\]', slot):
            # A slots should be absent, but if candidate introduces new A slot, penalize
            if slot not in prior:
                unrelated_penalty += 0.6
            continue
        if slot not in prior and slot != dst_slot:
            # new unrelated slot introduced
            unrelated_penalty += 0.6

    score -= unrelated_penalty
    # clamp result to [-1, 1]
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0

    return float(score)

