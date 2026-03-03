# WMQA Improved Rules
# Improved from (2 files):
#   - webshop_result/rules_webshop_transition_qa_qwen3-4b.py
#   - webshop_result/task_rules_webshop_transition_qa_qwen3-4b.py
# Dev unit-weight improvement vs original: +27.59%
# Dev unit-weight accuracy (improved rules): 91.22%
# Dev weighted accuracy (learned on dev): 92.32%
# Test baseline accuracy: 46.65%
# Test weighted accuracy: 92.60%
# Test weighted improvement: +45.95%

# Rule 1
def rule_reward(state, action, choice):
    import re
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = (action or "").strip()

    # We only handle click[...] actions
    m = re.match(r"^\s*click\[(.+)\]\s*$", act, flags=re.I)
    if not m:
        return 0.0
    label = m.group(1).strip()
    if not label:
        return 0.0

    st_norm = st
    # Check that the label appears in the state as a button (clicked or not)
    # Accept common variants like "[button] LABEL [button_]" or "[clicked button] LABEL [clicked button_]"
    label_esc = re.escape(label)
    btn_pattern = re.compile(
        r"(\[button\]\s*" + label_esc + r"\s*\[button_\])|(\[clicked button\]\s*" + label_esc + r"\s*\[clicked button_\])",
        flags=re.I
    )
    if not btn_pattern.search(st_norm):
        # If the exact bracketed form isn't found, be conservative and require the label to appear near a "size" header
        # or near a "color" or "size" marker — otherwise we don't apply the rule.
        if not re.search(r"\bsize\b", st_norm, flags=re.I) and not re.search(r"\bcolor\b", st_norm, flags=re.I) and not re.search(r"\bdescription\b", st_norm, flags=re.I):
            return 0.0
        # also check label appears somewhere in state text
        if re.search(label_esc, st_norm, flags=re.I) is None:
            return 0.0

    ch = choice or ""
    ch_norm = norm(ch)

    # Case 1: Buy Now -> expect terminal Success
    if label.lower() == "buy now" or label.lower() == "buy now]":
        # Accept choice being exactly "Success" (case-insensitive)
        return 1.0 if ch_norm == "success" else -1.0

    # Case 2: Size option clicked -> expect "You have clicked <label>" or a "[clicked button]" marker with label
    # Detect size-like labels by presence of numbers/units or the word "pack"
    if re.search(r"\b(fl oz|oz|ounce|pound|lb|pack|pack of)\b", label, flags=re.I) or re.search(r"\d", label):
        # look for explicit clicked indication in choice
        clicked_phrase = re.search(r"you have clicked\s+" + re.escape(label), ch, flags=re.I) is not None
        clicked_marker = re.search(r"\[clicked button\]\s*" + re.escape(label) + r"\s*\[clicked button_\]", ch, flags=re.I) is not None
        if clicked_phrase or clicked_marker:
            return 1.0
        else:
            return -1.0

    # Case 3: Content tab (Description / Features / Reviews) -> expect descriptive text (non-trivial)
    if label.lower() in ("description", "features", "reviews"):
        # Heuristic: choice should contain a reasonably long paragraph or multiple sentences.
        # Accept if choice has >80 chars of non-whitespace text OR contains at least two sentence-ending periods.
        stripped = re.sub(r"\[.*?\]", "", ch)  # remove bracket tags for length check
        long_enough = len(re.sub(r"\s+", "", stripped)) > 80
        multi_sent = len(re.findall(r"\.", stripped)) >= 2 or re.search(r"\bproviding\b|\bcontains\b|\bdescript", stripped, flags=re.I) is not None
        if long_enough or multi_sent:
            return 1.0
        else:
            return -1.0

    # For other buttons that appear as buttons but are not covered above, don't apply the rule.
    return 0.0

# Rule 2
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = norm(action)
    if act != "click[buy now]":
        return 0.0

    # Accept typical formatting of the Buy Now button.
    has_buy_now = re.search(r"(?i)\[button\]\s*buy\s*now\s*\[button_\]", st) is not None
    if not has_buy_now:
        return 0.0

    ch = norm(choice)
    return 1.0 if ch == "success" else -1.0

# Rule 3
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = norm(action or "")

    # Only consider buy-now clicks
    if act != "click[buy now]":
        return 0.0

    # Require a visible Buy Now button on the page and get its position
    buy_match = re.search(r"(?i)\[button\]\s*buy now\s*\[button_\]", st)
    if not buy_match:
        return 0.0
    buy_pos = buy_match.start()

    # Helper: find a threshold match that occurs before the buy button (likely from instruction)
    thr = None
    thr_patterns = [
        # patterns like "price < 50" or "price < $50"
        r"(?i)price[^a-z0-9]{0,6}<\s*\$?\s*([0-9]+(?:\.[0-9]+)?)",
        # patterns like "price lower than 50" or "price less than $50"
        r"(?i)price\s+(?:lower|less)\s+than\s+\$?\s*([0-9]+(?:\.[0-9]+)?)"
    ]
    thr_match = None
    for pat in thr_patterns:
        for m in re.finditer(pat, st):
            # require the matched threshold to appear before the Buy Now occurrence
            if m.start() < buy_pos:
                try:
                    thr_candidate = float(m.group(1))
                except Exception:
                    continue
                thr = thr_candidate
                thr_match = m
                break
        if thr is not None:
            break

    if thr is None:
        # No clear instruction threshold found before this Buy Now -> do not apply
        return 0.0

    # Find a product price near the buy button (price is usually shown near the Buy Now)
    # Search for a Price: $X that occurs before the buy button but not too far away.
    nearest_price = None
    nearest_price_pos = -1
    for m in re.finditer(r"(?i)price\s*[:\-]?\s*\$?\s*([0-9]+(?:\.[0-9]+)?)", st):
        ppos = m.start()
        # only consider prices that are before the buy button and reasonably close (e.g., within 600 chars)
        if ppos < buy_pos and (buy_pos - ppos) <= 600:
            # choose the closest one (largest start that is still < buy_pos)
            if ppos > nearest_price_pos:
                nearest_price_pos = ppos
                try:
                    nearest_price = float(m.group(1))
                except Exception:
                    nearest_price = None

    if nearest_price is None:
        # couldn't robustly find a product price near this Buy Now -> conservative: do not apply
        return 0.0

    product_price = nearest_price

    # Conservative guard: if instruction or nearby product block mentions quantities/units,
    # the price may be per-unit or per-pack; avoid applying the rule in those ambiguous cases.
    # Extract instruction text up to the buy button and a product-snippet around the price.
    instr_text = st[:buy_pos]
    product_snippet_start = max(0, nearest_price_pos - 200)
    product_snippet_end = min(len(st), buy_pos + 200)
    product_snippet = st[product_snippet_start:product_snippet_end]

    unit_ambiguity_pattern = re.compile(
        r"(?i)\b(pack(?:\s+of)?|\bper\b|\beach\b|\bunit\b|\bpackaged\b|\bpackage\b|\bpk\b|\bpcs\b|"
        r"\boz\b|\bounces?\b|\blb\b|\blbs\b|\blitre\b|\bliter\b)\b"
    )
    # If instruction mentions pack/each/per or the product snippet mentions weight/pack indicators, skip rule
    if unit_ambiguity_pattern.search(instr_text) or unit_ambiguity_pattern.search(product_snippet):
        return 0.0

    # Treat equality as acceptable (<=). But require a meaningful margin before penalizing:
    # If product_price > thr but within a small margin, treat as borderline and do not apply.
    # margin = max($1.00, 5% of threshold)
    margin = max(1.0, 0.05 * thr)

    if product_price <= thr:
        expected_success = True
    else:
        # product_price > thr
        if (product_price - thr) < margin:
            # borderline difference; be conservative and do not apply penalty/reward
            return 0.0
        expected_success = False

    ch = norm(choice or "")
    if expected_success:
        return 1.0 if ch == "success" else -1.0
    else:
        return 1.0 if ch == "fail" else -1.0

# Rule 4
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = action or ""
    ch = choice or ""

    # Only apply for click[...] actions
    m = re.search(r"click\[(.+?)\]", act, flags=re.IGNORECASE)
    if not m:
        return 0.0

    label = m.group(1).strip()  # raw label text, preserve punctuation/case for matching
    if not label:
        return 0.0

    # Check that the label exists as a button on the current page.
    # Accept either regular or already-clicked button representations.
    label_escaped = re.escape(label)
    button_pattern = re.compile(
        r"(?i)(\[button\]\s*" + label_escaped + r"\s*\[button_\]|\[clicked button\]\s*" + label_escaped + r"\s*\[clicked button_\])"
    )
    if not button_pattern.search(st):
        # The click targets a button not present on page -> rule not applicable
        return 0.0

    # Special-case "buy now" -> terminal Success expected
    if label.strip().lower() == "buy now":
        return 1.0 if norm(ch) == "success" else -1.0

    # For other buttons, the next state should reflect the click.
    # Accept either "You have clicked <label>." or a clicked-button marking.
    clicked_sentence_re = re.compile(r"(?i)you have clicked\s+" + label_escaped + r"\b")
    clicked_button_re = re.compile(r"(?i)\[clicked button\]\s*" + label_escaped + r"\s*\[clicked button_\]")

    if clicked_sentence_re.search(ch) or clicked_button_re.search(ch):
        return 1.0
    else:
        return -1.0

# Rule 5
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = norm(action)

    # This rule only applies for click[buy now]
    if act != "click[buy now]":
        return 0.0

    # Require that the page actually shows a Buy Now button
    has_buy_now = re.search(r"(?i)\[button\]\s*buy now\s*\[button_\]", st) is not None
    if not has_buy_now:
        return 0.0

    # Try to extract the instruction text (if present)
    instr = ""
    m_instr = re.search(r"(?is)instruction:\s*(.*?)(?:\n\[button\]|\nsize\n|\ncolor\n|\npage \d|\Z)", st)
    if m_instr:
        instr = m_instr.group(1).strip()
    else:
        # fallback: look for a single-line "Instruction:" occurrence
        m2 = re.search(r"(?i)instruction:\s*(.*)", st)
        if m2:
            instr = m2.group(1).strip()

    instr_n = norm(instr)

    # Extract price threshold from instruction (e.g., "price lower than 50.00 dollars")
    thr = None
    m_thr = re.search(r"price\s*(?:lower than|under|less than|<)?\s*\$?\s*([0-9]+(?:\.[0-9]+)?)", instr_n)
    if m_thr:
        try:
            thr = float(m_thr.group(1))
        except:
            thr = None

    # If no explicit price constraint in instruction, do not apply this rule
    if thr is None:
        return 0.0

    # Extract product price shown on page
    m_price = re.search(r"(?i)price:\s*\$([0-9]+(?:\.[0-9]+)?)", st)
    if not m_price:
        return 0.0
    try:
        prod_price = float(m_price.group(1))
    except:
        return 0.0

    price_ok = (prod_price <= thr)

    # If instruction contains an explicit color requirement, try to extract it
    color_req = None
    m_color = re.search(r"(?i)color\s*(?:was|is|in|:)?\s*([^,;\.\n]+)", instr_n)
    if m_color:
        color_req = m_color.group(1).strip()

    # If color requirement exists, check that it appears somewhere on the page text
    color_ok = True
    if color_req:
        # normalize and check presence (allow substrings like "white" matching "white | off white")
        page_norm = norm(st)
        col_norm = norm(color_req)
        if col_norm == "":
            color_ok = True
        else:
            color_ok = (col_norm in page_norm)

    # Apply the rule only if product satisfies the price constraint and (if specified) the color
    if price_ok and color_ok:
        return 1.0 if norm(choice) == "success" else -1.0

    # Conditions not met: rule does not apply
    return 0.0

# Rule 6
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = norm(action)
    ch = norm(choice)

    # Only trigger for the buy now click action
    if act != "click[buy now]":
        return 0.0

    # Detect presence of a Buy Now button (accept common formatting variants)
    has_buy_now = re.search(r"(?i)\[button\]\s*buy\s*now\s*\[button_\]", st) is not None
    if not has_buy_now:
        return 0.0

    # If buy now exists, expected terminal token is "Success"
    return 1.0 if ch == "success" else -1.0

# Rule 7
def rule_reward(state, action, choice):
    """
    Rule: Clicking an option under 'size' or 'color' should produce either
    a confirmation "You have clicked <label>" or the option shown as a
    clicked button "[clicked button] <label> [clicked button_]".
    Returns:
      1.0 if choice matches expected outcome,
     -1.0 if rule applies but choice does not match,
      0.0 if rule does not apply.
    """
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip())

    st = state or ""
    act = action or ""
    ch = choice or ""

    # Only apply to click[...] actions
    m = re.match(r'^\s*click\[(.+)\]\s*$', act, flags=re.I)
    if not m:
        return 0.0
    label = m.group(1).strip()

    # Only apply when state contains a 'size' or 'color' section (option-type click)
    if re.search(r'(?i)^\s*size\b', st, flags=re.M) is None and re.search(r'(?i)^\s*color\b', st, flags=re.M) is None:
        return 0.0

    # Confirm the label exists as a button option in the state
    pattern_button = r'\[button\]\s*' + re.escape(label) + r'\s*\[button_\]'
    if re.search(pattern_button, st, flags=re.I) is None:
        return 0.0

    # Expected outcomes:
    # 1) Confirmation line "You have clicked <label>"
    # 2) The option rendered as clicked: "[clicked button] <label> [clicked button_]"
    pat_conf = r'(?i)\byou have clicked\s*' + re.escape(label) + r'\b'
    pat_clicked = r'\[clicked button\]\s*' + re.escape(label) + r'\s*\[clicked button_\]'

    if re.search(pat_conf, ch, flags=re.I) or re.search(pat_clicked, ch, flags=re.I):
        return 1.0
    else:
        return -1.0

# Rule 8
def rule_reward(state, action, choice):
    import re
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = action or ""
    ch = choice or ""

    nact = norm(act)

    # Only consider click[...] actions
    m = re.match(r"click\[(.*)\]\s*$", nact)
    if not m:
        return 0.0

    # Extract clicked label (preserve original casing from action string for matching)
    # Use raw action string to capture label exactly as user clicked
    raw_label_match = re.match(r"click\[(.*)\]\s*$", (action or ""))
    if not raw_label_match:
        return 0.0
    label = raw_label_match.group(1).strip()
    if not label:
        return 0.0

    # Check that the label exists in the state as an unclicked button.
    # Accept patterns like "[button] <label> [button_]" (case-insensitive, allow extra spaces)
    esc_label = re.escape(label)
    btn_pattern = re.compile(r"\[button\]\s*" + esc_label + r"\s*\[button_\]", re.IGNORECASE)
    clicked_pattern = re.compile(r"\[clicked button\]\s*" + esc_label + r"\s*\[clicked button_\]", re.IGNORECASE)

    if not btn_pattern.search(st):
        # The clicked label is not present as an unclicked button in state -> rule does not apply
        return 0.0

    # Now the rule applies: check choice for expected clicked-state or product detail
    # 1) clicked-state: "[clicked button] <label" or "you have clicked <label>"
    if clicked_pattern.search(ch):
        return 1.0
    if re.search(r"you have clicked\s+" + esc_label, ch, re.IGNORECASE):
        return 1.0

    # 2) product detail view: presence of product fields commonly shown on item click
    # Accept "price:", "buy now", or "rating:" as indicators of a product details page.
    if re.search(r"\bprice\s*:", ch, re.IGNORECASE) or re.search(r"\bbuy now\b", ch, re.IGNORECASE) or re.search(r"\brating\s*:", ch, re.IGNORECASE):
        return 1.0

    # If none of the expected indicators appear, penalize (page likely unchanged)
    # Also penalize trivial navigation-only responses (e.g., only "Back to Search")
    # Detect trivial nav-only by absence of meaningful content beyond navigation buttons.
    # If choice contains only "[button] back to search [button_]" or lacks product/selection change, treat as failure.
    # (Simpler fallback: if none of the positive indicators above, return -1.0)
    return -1.0

# Rule 9
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    ch = choice or ""
    act = (action or "").strip()

    # Only apply when action is click[...] exactly
    m = re.match(r"(?i)^\s*click\[(.+)\]\s*$", act)
    if not m:
        return 0.0

    label = m.group(1).strip()
    if not label:
        return 0.0

    # Prepare regex-escaped label for searches
    esc_label = re.escape(label)

    # Check that the label exists in the current state as a selectable button
    has_button = (
        re.search(r"(?i)\[button\]\s*" + esc_label + r"\s*\[button_\]", st) is not None
        or re.search(r"(?i)\[clicked button\]\s*" + esc_label + r"\s*\[clicked button_\]", st) is not None
    )
    if not has_button:
        # Preconditions don't match: do not apply rule
        return 0.0

    # Normalize choice for checks
    n_ch = norm(ch)

    # 1) Accept explicit acknowledgement: "You have clicked <label>"
    ack_pattern = r"(?i)\byou have clicked\s+" + esc_label + r"\b"
    if re.search(ack_pattern, ch):
        return 1.0

    # 2) Accept button rendered as clicked: "[clicked button] <label> [clicked button_]"
    clicked_btn_pattern = r"(?i)\[clicked button\]\s*" + esc_label + r"\s*\[clicked button_\]"
    if re.search(clicked_btn_pattern, ch):
        return 1.0

    # 3) For content buttons (features/description/reviews), accept substantial content expansion
    content_buttons = {"features", "description", "reviews"}
    if label.lower() in content_buttons:
        # If choice is considerably longer than state and contains the label/title, accept.
        if len(ch) > len(st) + 40 and re.search(r"(?i)\b" + esc_label + r"\b", ch):
            return 1.0

    # If none of the expected reflections of the click are present, penalize.
    return -1.0

# Rule 10
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = action or ""
    ch = choice or ""

    # Extract label from action if it's a click[...] action.
    m = re.match(r'^\s*click\[(.*)\]\s*$', act, re.IGNORECASE)
    if not m:
        return 0.0

    raw_label = m.group(1)
    label = norm(raw_label)

    # Normalize state and choice for simple substring checks.
    st_norm = norm(st)
    ch_norm = norm(ch)

    # Compose the expected button tokens in normalized form.
    unclicked_token = f"[button] {label} [button_]"
    clicked_token = f"[clicked button] {label} [clicked button_]"

    # Only apply the rule if the target label exists as a button (either unclicked or already clicked)
    # on the current page. If not present, rule does not apply.
    if (unclicked_token not in st_norm) and (clicked_token not in st_norm):
        return 0.0

    # Now evaluate the choice: it should show the label as clicked after the click action.
    if clicked_token in ch_norm:
        return 1.0
    else:
        # If the choice still shows the unclicked button (or doesn't show the expected clicked marker),
        # treat as incorrect.
        return -1.0

# Rule 11
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip())

    st = state or ""
    act = (action or "").strip()

    # Only apply for click[...] actions
    m = re.match(r"^\s*click\[(.+)\]\s*$", act, flags=re.IGNORECASE)
    if not m:
        return 0.0

    label = m.group(1).strip()
    if label == "":
        return 0.0

    # Build regex-safe label and allow optional surrounding whitespace
    esc = re.escape(label)
    # Check that the initial state contains an unclicked button with this exact label
    unclicked_pattern = re.compile(r"(?i)\[button\]\s*" + esc + r"\s*\[button_\]")
    if not unclicked_pattern.search(st):
        # The clicked label isn't present as an unclicked button in the state -> rule doesn't apply
        return 0.0

    # Check whether the choice marks this button as clicked
    clicked_pattern = re.compile(r"(?i)\[clicked button\]\s*" + esc + r"\s*\[clicked button_\]")
    if clicked_pattern.search(choice or ""):
        return 1.0

    # The state had the button, action clicked it, but the choice does NOT show it clicked -> negative signal
    return -1.0

# Rule 12
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = action or ""
    ch = choice or ""

    # Extract label from action if it's click[...]
    m = re.match(r"(?i)\s*click\[(.*?)\]\s*$", act)
    if not m:
        return 0.0
    label = m.group(1).strip()
    if not label:
        return 0.0

    # Check that the state contains the unclicked button with that label
    # Allow minor whitespace and case insensitivity.
    label_esc = re.escape(label)
    has_button_in_state = re.search(r"(?i)\[button\]\s*%s\s*\[button_\]" % label_esc, st) is not None
    if not has_button_in_state:
        return 0.0

    # Patterns that indicate the label was reflected in the next page:
    # 1) clicked button marker
    clicked_pat = re.compile(r"(?i)\[clicked button\].*?\b%s\b.*?\[clicked button_\]" % label_esc, re.DOTALL)
    # 2) explicit "You have clicked X"
    you_clicked_pat = re.compile(r"(?i)you have clicked\s+%s\b" % label_esc)

    if clicked_pat.search(ch) or you_clicked_pat.search(ch):
        return 1.0

    # For tab-like labels, accept content indicative of the tab instead of clicked markers.
    tab = label.strip().lower()
    tab_keywords = []
    if tab in ("features",):
        tab_keywords = ["materials", "occasion", "features"]
    elif tab in ("description",):
        tab_keywords = ["description", "materials", "model", "product", "features"]
    elif tab in ("reviews",):
        tab_keywords = ["reviews", "review", "rating"]

    if tab_keywords:
        ch_norm = norm(ch)
        for kw in tab_keywords:
            if re.search(r"(?i)\b%s\b" % re.escape(kw), ch_norm):
                return 1.0

    # If none of the acceptable indicators are present, penalize.
    return -1.0

# Rule 13
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = action or ""
    ch = choice or ""

    # Only apply to click[...] actions
    m = re.match(r"^\s*click\[(.*)\]\s*$", act, flags=re.IGNORECASE)
    if not m:
        return 0.0
    action_label = norm(m.group(1))

    # Extract all button labels from the state (both normal and already-clicked variants)
    btn_labels = []
    for pat in [r"\[button\](.*?)\[button_\]", r"\[clicked button\](.*?)\[clicked button_\]"]:
        for g in re.findall(pat, st, flags=re.IGNORECASE | re.DOTALL):
            btn_labels.append(norm(g))

    # If no button in state matches the action label (as substring match either way), do not apply rule
    def label_matches(btn, lbl):
        return (btn and lbl) and (lbl in btn or btn in lbl)

    matching_buttons = [b for b in btn_labels if label_matches(b, action_label)]
    if not matching_buttons:
        return 0.0

    # Check whether the choice shows the clicked/selected state for the action label.
    # Accept either a [clicked button] wrapper or a sentence like "You have clicked X"
    clicked_labels = []
    for g in re.findall(r"\[clicked button\](.*?)\[clicked button_\]", ch, flags=re.IGNORECASE | re.DOTALL):
        clicked_labels.append(norm(g))

    you_clicked_present = re.search(r"(?i)you have clicked\s+(.+?)(?:[.\n]|$)", ch) is not None
    you_clicked_labels = []
    if you_clicked_present:
        # capture the label(s) mentioned after "You have clicked"
        for gm in re.finditer(r"(?i)you have clicked\s+(.+?)(?:[.\n]|$)", ch):
            you_clicked_labels.append(norm(gm.group(1)))

    # Determine if any clicked label matches the action_label
    clicked_match = any(label_matches(lbl, action_label) for lbl in clicked_labels + you_clicked_labels)

    return 1.0 if clicked_match else -1.0

# Rule 14
# Task group: click_buy_now
def rule_reward(state, action, choice):
    """
    Conservative merged rule for the click_buy_now group.

    Triggers only when:
      - action exactly equals "click[buy now]" (after stripping whitespace)
      - state contains the explicit buy-button token "[button] Buy Now [button_]"
        (matched case-insensitively)

    Returns:
      +1.0 if choice == "Success"
      -1.0 if choice == "Fail"
       0.0 otherwise (including when rule does not apply)
    """
    if action is None:
        return 0.0
    if action.strip() != "click[buy now]":
        return 0.0

    s = (state or "").lower()
    if "[button] buy now [button_]" not in s:
        return 0.0

    if choice == "Success":
        return 1.0
    if choice == "Fail":
        return -1.0
    return 0.0

# Rule 15
# Task group: click_change_page
import re

def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    # Conservative: only apply when action is exactly the pagination clicks and
    # the state clearly indicates a Page number and a matching Next/Prev affordance.
    if not isinstance(state, str) or not isinstance(action, str):
        return 0.0

    if action not in ("click[next >]", "click[< prev]"):
        return 0.0

    # Require a clear page header like "Page N"
    m = re.search(r"Page\s+(\d+)", state)
    if not m:
        return 0.0
    try:
        current_n = int(m.group(1))
    except Exception:
        return 0.0

    # Require the navigation affordance to be present in the state for extra confidence.
    if action == "click[next >]":
        if not ("Next >" in state or "[button] Next" in state or "[button] Next >" in state):
            return 0.0
        expected_n = current_n + 1
    else:  # action == "click[< prev]"
        if not ("< Prev" in state or "[button] < Prev" in state or "[button] < Prev [button_]" in state):
            return 0.0
        expected_n = current_n - 1
        if expected_n < 1:
            return 0.0

    # If the state contains an exact total-results suffix like "(Total results: 50)",
    # require the same substring to appear in the candidate.
    tot_match = re.search(r"\(Total results:\s*\d+\)", state)
    tot_substr = tot_match.group(0) if tot_match else None

    if not isinstance(choice, str):
        # candidate not a textual page -> definite mismatch when preconditions hold
        return -1.0

    # Check that the candidate includes the expected Page header and, if present, the same total suffix.
    if ("Page {}".format(expected_n) in choice) and (tot_substr is None or tot_substr in choice):
        return 1.0
    else:
        return -1.0

# Rule 16
# Task group: click_item
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    # Map known click actions to the expected markers on the product detail page.
    expected = {
        ("click[item - extra soft toothbrush deep clean toothbrush with 10000 "
         "bristles for sensitive toothbrushes sensitive teeth manual protection care]"):
            {
                "title_substr": "Extra Soft Toothbrush Deep Clean Toothbrush",
                "price_substr": "Price: $4.99",
                "require_buy": True,
                "buy_substr": "[button] Buy Now [button_]"
            },
        ("click[item - extra soft toothbrush for sensitive gums/ tongue cleaner set 4pcs - "
         "nano manual toothbrush for sensitive teeth/ soft bristles toothbrush for adult /children "
         "(3 toothbrush+1tongue scraper)]"):
            {
                "title_substr": "Extra Soft Toothbrush for Sensitive Gums/ Tongue Cleaner Set 4PCS",
                "price_substr": "Price: $15.99",
                "require_buy": False,
                "buy_substr": None
            }
    }

    # Only apply when the action exactly matches one of the known click strings.
    if action not in expected:
        return 0.0

    # Must be a string candidate to inspect.
    if not isinstance(choice, str):
        return -1.0

    spec = expected[action]
    has_title = spec["title_substr"] in choice
    has_price = spec["price_substr"] in choice

    if not (has_title and has_price):
        return -1.0

    if spec["require_buy"]:
        if spec["buy_substr"] in choice:
            return 1.0
        else:
            return -1.0

    return 1.0

# Rule 17
# Task group: click_nav
def rule_reward(state, action, choice):
    """
    Returns float in [-1.0, 1.0].

    Applies only for exact actions "click[description]" or "click[features]" and only when the
    corresponding button is present in the state. Conservative: if signals are ambiguous, return 0.0.
    """
    if not isinstance(action, str):
        return 0.0

    # Normalize inputs
    s = state or ""
    c = choice or ""
    lower_c = c.lower()
    lower_s = s.lower()

    # Helper: detect navigation-only appearance
    nav_indicators = ["[button] back to search", "[button] < prev", "[button] <prev", "[button] back"]
    has_nav = any(nb in lower_c for nb in nav_indicators)

    if action == "click[description]":
        # Require the page had a Description button
        if "[button] description [button_]" not in lower_s:
            return 0.0

        # description indicators (case-insensitive)
        desc_indicators = [
            "description:", "product features", "plug-&-play", "ultra series",
            "specifications", "installation", "compatibility", "supports 3d",
            "instruction:", "official hdmi adopter"
        ]

        if any(kw in lower_c for kw in desc_indicators):
            return 1.0

        # If we see only navigation buttons and no description indicators, penalize
        if has_nav and not any(kw in lower_c for kw in desc_indicators):
            return -1.0

        # Ambiguous / cannot decide
        return 0.0

    elif action == "click[features]":
        # Require the page had a Features button
        if "[button] features [button_]" not in lower_s:
            return 0.0

        # Strong feature phrases (case-insensitive)
        strong_feature_phrases = ["embedded glitter", "three layers protection", "raised bezel"]
        # We also accept a combination of structural markers as signal of a features block
        has_contains = "contains:" in lower_c
        has_cert = "certification" in lower_c
        has_any_feature_keyword = any(k in lower_c for k in strong_feature_phrases) or "features:" in lower_c

        # Reward if clear feature content is present
        if any(k in lower_c for k in strong_feature_phrases):
            return 1.0
        if has_contains and has_cert:
            return 1.0
        if has_any_feature_keyword:
            # weaker signal but still indicative -> accept
            return 1.0

        # If we see only navigation buttons and no feature indicators, penalize
        if has_nav and not (has_contains or has_cert or has_any_feature_keyword):
            return -1.0

        # Ambiguous / cannot decide
        return 0.0

    # Action not relevant to this rule
    return 0.0

# Rule 18
# Task group: click_other
def rule_reward(state, action, choice):
    """
    Returns a float in [-1, 1].
    Triggers only for actions of the form "click[LABEL]" when the state contains
    the unclicked button substring "[button] LABEL [button_]".
    Rewards +1.0 if the choice contains an explicit clicked indication for LABEL
    ("You have clicked LABEL." or "[clicked button] LABEL [clicked button_]"),
    -1.0 if the action/state match but the choice lacks those indicators,
    and 0.0 otherwise (conservative).
    """
    import re

    # Basic type checks
    if not isinstance(action, str) or not isinstance(state, str):
        return 0.0

    # Match actions of the form click[LABEL]
    m = re.fullmatch(r"click\[(.+)\]", action)
    if not m:
        return 0.0
    label = m.group(1)

    # Only apply when the current state clearly shows the unclicked button
    unclicked_marker = f"[button] {label} [button_]"
    if unclicked_marker not in state:
        return 0.0

    # choice must be a string; if not, treat as incorrect transition
    if not isinstance(choice, str):
        return -1.0

    # Indicators that the next state correctly reflects the click
    verbal = f"You have clicked {label}."
    clicked_markup = f"[clicked button] {label} [clicked button_]"

    if verbal in choice or clicked_markup in choice:
        return 1.0
    else:
        return -1.0

# Rule 19
# Task group: search
def rule_reward(state, action, choice):
    """
    Returns a float in [-1, 1].
    - Only triggers for actions that look like search[...] queries.
    - +1.0 if the choice contains multiple strong search-results indicators
      (pagination + product/ASIN or pagination + price or Back-to-Search + product).
    - -1.0 if the choice contains clear error/redirect/payment/category signals.
    - 0.0 (abstain) otherwise (conservative).
    """
    import re

    # Only consider explicit search actions
    if not isinstance(action, str) or not action.startswith("search["):
        return 0.0

    if not isinstance(choice, str):
        return 0.0

    low = choice.lower()

    # Strong negative/incorrect patterns observed in numerous cases -> penalize
    negative_indicators = [
        "you have been redirected",         # redirected to product page wording
        "you've searched for",              # short summary / wrong "you've searched" reply
        "you've been redirected",           # variant
        "please choose a payment method",   # payment selection page
        "select a payment option",          # payment selection page
        "amazon home & kitchen",            # unrelated category landing page
        "filter by price",                  # category/filter landing page
        "filter by brand",                  # category/filter landing page
        "checklist available for:",         # model's generic checklist output
        "here is the list of available items:", # another generic non-results format
    ]
    for neg in negative_indicators:
        if neg in low:
            return -1.0

    # Detect pagination / search-results markers (several formats observed)
    has_pagination = (
        "page 1 (total results:" in choice  # observed exact substring
        or "page 1 (total results:" in low
        or "total results: 50" in low
        or "total results:" in low and "page 1" in low
    )

    # Detect a product/ASIN marker: either an ASIN like BXXXXXXXXX or the button prefix used in pages
    asin_match = re.search(r"\bB[0-9A-Z]{9}\b", choice.upper() or "")
    has_button_b = "[button] b" in low or "[button] B" in choice  # catches "[button] B09..." patterns
    has_product_marker = bool(asin_match) or has_button_b

    # Detect price presence (simple dollar-sign heuristic)
    has_price = "$" in choice

    # Detect "Back to Search" control which many correct pages include
    has_back_to_search = ("[button] back to search" in low) or ("[button] Back to Search" in choice)

    # Conservative positive decision: require at least two different strong indicators
    # (pagination + product OR pagination + price OR back-to-search + product)
    if (has_pagination and has_product_marker) or (has_pagination and has_price) or (has_back_to_search and has_product_marker):
        return 1.0

    # If the action is a search but we don't see clear positives or negatives, abstain
    return 0.0

# Rule 20
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = norm(action)
    ch = norm(choice)

    # Only apply to explicit buy attempts
    if act != "click[buy now]":
        return 0.0

    # Try to extract a user-specified max price from the instruction line in the state.
    # Look for phrases like "price lower than 240.00 dollars", "price less than $50", etc.
    # We search the entire state for a "price ... than" pattern.
    thr = None
    m = re.search(r"price\s*(?:lower|less)?\s*(?:than|<)\s*\$?\s*([0-9]{1,3}(?:[0-9,]*)(?:\.[0-9]+)?)", st, flags=re.I)
    if not m:
        # Accept variations like "price lower than 240.00 dollars" or "price < 50.00"
        m = re.search(r"price.*?([0-9]{1,3}(?:[0-9,]*)(?:\.[0-9]+)?)\s*(?:dollars|\$)?", st, flags=re.I)
        # The fallback above may capture other numbers; ensure "price" is near it
        if m:
            # confirm the substring near match contains comparison words
            window = st[max(0, m.start()-30):m.end()+30]
            if not re.search(r"(lower|less|<|maximum|max|<=)", window, flags=re.I):
                m = None

    if not m:
        # Rule only applies when the user's instruction contains an explicit max price
        return 0.0

    try:
        thr = float(m.group(1).replace(",", ""))
    except Exception:
        return 0.0

    # Check for presence of Buy Now button on the page
    has_buy_now = re.search(r"(?i)\[button\]\s*buy now\s*\[button_\]", st) is not None

    # If there's no buy button, the expected result is Fail (can't buy)
    if not has_buy_now:
        expected = "fail"
        return 1.0 if ch == expected else -1.0

    # If Buy Now exists, attempt to parse the displayed product price(s).
    # Prefer "Price: $X" or "Price: $X to $Y" patterns.
    price_match = re.search(r"price\s*[:]\s*\$?\s*([0-9]{1,3}(?:[0-9,]*)(?:\.[0-9]+)?)(?:\s*(?:to|-|–)\s*\$?\s*([0-9]{1,3}(?:[0-9,]*)(?:\.[0-9]+)?))?", st, flags=re.I)
    if not price_match:
        # As a fallback, try to find dollar amounts nearby top of page or first occurrence after product title.
        all_dollars = re.findall(r"\$\s*([0-9]{1,3}(?:[0-9,]*)(?:\.[0-9]+)?)", st)
        if all_dollars:
            # Use the first dollar amount found (conservative)
            try:
                price_vals = [float(x.replace(",", "")) for x in all_dollars]
                prod_max_price = max(price_vals)
            except Exception:
                return 0.0
        else:
            # Cannot reliably parse product price; do not apply the rule
            return 0.0
    else:
        try:
            low = float(price_match.group(1).replace(",", ""))
            if price_match.group(2):
                high = float(price_match.group(2).replace(",", ""))
                prod_max_price = max(low, high)
            else:
                prod_max_price = low
        except Exception:
            return 0.0

    # Compare product max price to user's threshold: user asked for price lower than thr.
    # If product_max >= thr => does NOT meet user's "lower than" constraint => expected Fail
    if prod_max_price >= thr:
        expected = "fail"
    else:
        expected = "success"

    if ch == expected:
        return 1.0
    elif ch in ("success", "fail"):
        return -1.0
    else:
        return 0.0

# Rule 21
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = norm(action)
    ch = norm(choice)

    # Apply only for actions of the form: click[item - <item-name>]
    m = re.match(r"click\[\s*item\s*-\s*(.*?)\s*\]$", act)
    if not m:
        return 0.0

    item_name = m.group(1)
    if not item_name:
        return 0.0

    item_norm = norm(item_name)

    # If the choice is exactly a terminal token, treat it as not matching a product page.
    if ch in ("success", "fail", ""):
        return -1.0

    # Exact substring match is the strongest signal.
    if item_norm in ch:
        return 1.0

    # Otherwise, check token overlap on significant tokens (length > 2).
    def tokens(s):
        return [t for t in re.findall(r"[a-z0-9']+", s) if len(t) > 2]

    item_tokens = tokens(item_norm)
    if not item_tokens:
        # fallback to substring check already done; penalize if nothing matched
        return -1.0

    choice_tokens = set(tokens(ch))
    matched = sum(1 for t in item_tokens if t in choice_tokens)
    # require majority of significant tokens to be present
    if matched / len(item_tokens) >= 0.6:
        return 1.0
    else:
        return -1.0

# Rule 22
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = norm(action)
    # Only apply when action is exactly "click[buy now]"
    if act != "click[buy now]":
        return 0.0

    # Look for a Buy Now button in the page text (common format: "[button] Buy Now [button_]")
    has_buy_now = re.search(r"(?i)\[button\]\s*buy\s*now\s*\[button_\]", st) is not None

    # If the page actually contains a Buy Now button, this rule does not apply.
    if has_buy_now:
        return 0.0

    ch = norm(choice)
    # If no Buy Now button but the agent clicked buy now, the correct terminal result should be Fail.
    return 1.0 if ch == "fail" else -1.0

# Rule 23
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = norm(action or "")
    ch = choice or ""

    # Apply only for clicks on product-detail tabs
    if act not in ("click[description]", "click[features]", "click[reviews]"):
        return 0.0

    # Ensure the current page actually has a description/features/reviews button
    has_tab = re.search(r"(?i)\[button\]\s*(description|features|reviews)\s*\[button_\]", st) is not None
    if not has_tab:
        return 0.0

    # Terminal tokens are inappropriate for a detail-tab click
    if norm(ch) in ("success", "fail"):
        return -1.0

    # Heuristics to detect an options/selection page (lots of buttons, size/color keywords)
    button_count = len(re.findall(r"\[button\]", ch))
    has_selection_keywords = re.search(r"(?i)\b(size|color|price|rating|buy now|back to search|<\s*prev|next\s*>)\b", ch) is not None
    options_page = (button_count >= 3) and has_selection_keywords

    if options_page:
        return -1.0

    # Heuristics to detect a product-description/detail page:
    # - Contains at least one reasonably long sentence (periods) and overall length is nontrivial.
    desc_sentences = re.findall(r"[A-Z][^\.]{10,}\.", ch)
    long_enough = len(ch) > 80
    has_prose = (len(desc_sentences) >= 1) or (long_enough and ch.count('.') >= 1)

    if has_prose:
        return 1.0

    # If we can't confidently recognize a description, penalize slightly (treated as incorrect)
    return -1.0

# Rule 24
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = (action or "").strip()

    # Only apply when action is a search[...] call.
    if not re.match(r"(?i)^\s*search\[\s*.*\s*\]\s*$", act):
        return 0.0

    # Check that current state actually contains a Search control (we expect searches to be issued from a page with a Search button).
    has_search_button = re.search(r"(?i)\[button\]\s*search\s*\[button_?\]", st) is not None
    if not has_search_button:
        # If the page doesn't show a Search button, this rule is not applicable.
        return 0.0

    ch = choice or ""

    # Detect result-page markers in the choice:
    # 1) "total results" phrase
    has_total_results = re.search(r"(?i)total results", ch) is not None
    # 2) product ASIN-like entries: lines like "[button] B0..." possibly followed by a price like "$12.99"
    has_asin_entry = re.search(r"(?i)\[button\]\s*b0[0-9a-z]{5,}\b", ch) is not None
    # 3) price occurrences and Next > pagination button
    has_price = re.search(r"\$\s*\d+(\.\d{2})?", ch) is not None
    has_next = re.search(r"(?i)\[button\]\s*next\s*>\s*\[button_?\]", ch) is not None

    looks_like_results = has_total_results or (has_asin_entry and has_price) or has_next

    return 1.0 if looks_like_results else -1.0

# Rule 25
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = (action or "").strip()

    # Recognize pagination click actions robustly
    is_click_prev = re.fullmatch(r"\s*click\[\s*<\s*prev\s*\]\s*", act, flags=re.IGNORECASE) is not None
    is_click_next = re.fullmatch(r"\s*click\[\s*next\s*>\s*\]\s*", act, flags=re.IGNORECASE) is not None
    if not (is_click_prev or is_click_next):
        return 0.0

    # Verify the clicked button actually exists on the current state
    has_prev_btn = re.search(r"(?i)\[button\]\s*<\s*prev\s*\[button_\]", st) is not None
    has_next_btn = re.search(r"(?i)\[button\]\s*next\s*>\s*\[button_\]", st) is not None

    if is_click_prev and not has_prev_btn:
        return 0.0
    if is_click_next and not has_next_btn:
        return 0.0

    ch = choice or ""
    ch_norm = norm(ch)

    # Check that choice looks like a search-results page:
    # - contains "page <number>"
    # - contains "total results"
    # - contains a "Next >" pagination button
    has_page_num = re.search(r"(?i)\bpage\s*\d+", ch) is not None
    has_total_results = re.search(r"(?i)total\s+results", ch) is not None
    has_next_in_choice = re.search(r"(?i)\[button\]\s*next\s*>\s*\[button_\]", ch) is not None

    if has_page_num and has_total_results and has_next_in_choice:
        return 1.0
    else:
        return -1.0

# Rule 26
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = (action or "").strip()
    ch = choice or ""

    # Normalize checks for presence of UI elements in the current state
    has_prev_in_state = re.search(r"(?i)\[button\]\s*<\s*prev\s*\[button_\]", st) is not None
    has_search_in_state = re.search(r"(?i)\[button\]\s*search\s*\[button_\]", st) is not None

    # Determine action type
    is_click_prev = re.search(r"(?i)^click\[\s*<\s*prev\s*\]\s*$", act) is not None
    is_search_action = re.search(r"(?i)^search\[(.*)\]\s*$", act) is not None

    # Only apply this rule when the action matches and the current page contains the expected button
    if is_click_prev and not has_prev_in_state:
        return 0.0
    if is_search_action and not has_search_in_state:
        return 0.0
    if not (is_click_prev or is_search_action):
        return 0.0

    # Heuristics to detect a results/listing page in the choice text
    has_next = re.search(r"(?i)\[button\]\s*next\s*>\s*\[button_\]", ch) is not None
    has_page_num = re.search(r"(?i)\bpage\s*\d+", ch) is not None
    has_total_results = re.search(r"(?i)total\s+results?", ch) is not None

    is_results_page = has_next and (has_total_results or has_page_num)

    # Heuristics to detect a single product detail page
    has_buy_now = re.search(r"(?i)\[button\]\s*buy\s*now\s*\[button_\]", ch) is not None
    has_description_buttons = re.search(r"(?i)\[button\]\s*description\s*\[button_\]", ch) is not None
    is_product_detail = has_buy_now or has_description_buttons

    if is_results_page:
        return 1.0
    if is_product_detail:
        return -1.0

    # Ambiguous mismatch (action expected results but choice doesn't clearly match results or product)
    return -0.5

# Rule 27
def rule_reward(state, action, choice):
    import re
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    ch = choice or ""
    act = norm(action)

    # Only apply when action is a click for description or previous navigation.
    is_click_description = re.search(r"click\[\s*description\s*\]", act) is not None
    is_click_prev = re.search(r"click\[\s*<\s*prev\s*\]", act) is not None

    # If neither action matches, rule does not apply.
    if not (is_click_description or is_click_prev):
        return 0.0

    st_norm = norm(st)
    ch_norm = norm(ch)

    # Helper: detect presence of Description button in the current page.
    has_description_button = re.search(r"\[button\]\s*description\s*\[button_?\]", st, re.IGNORECASE) is not None

    # Helper: detect presence of < Prev button in the current page.
    has_prev_button = re.search(r"\[button\]\s*<\s*prev\s*\[button_?\]", st, re.IGNORECASE) is not None

    # Branch for description click
    if is_click_description:
        if not has_description_button:
            return 0.0  # precondition not met, do not apply

        # Consider the choice a valid description page if it contains at least one
        # reasonably long line/paragraph of free-form text (no [button] tag) with >40 chars.
        # This filters out pages that only show option buttons (sizes, colors) without description.
        paragraphs = [p.strip() for p in re.split(r"\n{1,}", choice or "") if p.strip()]
        has_long_freeform = False
        for p in paragraphs:
            if "[button" in p.lower():
                continue
            if len(p) >= 40 and re.search(r"[a-z]\s", p, re.IGNORECASE):
                has_long_freeform = True
                break

        return 1.0 if has_long_freeform else -1.0

    # Branch for < Prev click
    if is_click_prev:
        if not has_prev_button:
            return 0.0  # precondition not met, do not apply

        # A results page commonly contains 'page' or 'total results' indicators and/or a Next > button.
        looks_like_results = False
        if re.search(r"total results", ch, re.IGNORECASE) or re.search(r"\bpage\s*\d+", ch, re.IGNORECASE):
            looks_like_results = True
        if re.search(r"\[button\]\s*next\s*>\s*\[button_?\]", ch, re.IGNORECASE):
            looks_like_results = True

        # Also require that the results page lists multiple product-like entries (lines with $ or item ids)
        multi_entries = len(re.findall(r"\$\s*\d{1,3}(?:\.\d{2})?", ch)) >= 1 or len(re.findall(r"\[button\]\s*[a-z0-9]{4,}\s*\[button_?\]", ch, re.IGNORECASE)) >= 2

        # If it looks like results and has entries, accept it.
        if looks_like_results and multi_entries:
            return 1.0
        # If it strongly looks like results (page/total/next) even without many $ prices, accept.
        if looks_like_results:
            return 0.8
        return -1.0

    return 0.0

# Rule 28
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = norm(action)
    ch = choice or ""

    # Only apply when action is click[< prev]
    if act != "click[< prev]":
        return 0.0

    # Ensure the current state actually has a < Prev button
    has_prev = re.search(r"(?i)\[button\]\s*<\s*prev\s*\[button_\]", st) is not None
    if not has_prev:
        return 0.0

    # Detect a dollar price like $12 or $12.34 anywhere in the choice string
    has_price = re.search(r"\$\s*\d+(?:\.\d{2})?", ch) is not None

    return 1.0 if has_price else -1.0

# Rule 29
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    ch = choice or ""
    act = norm(action)

    # Only apply for exact action click[< prev]
    if act != "click[< prev]":
        return 0.0

    # Check that current state actually has a < Prev button
    has_prev = re.search(r"(?i)\[button\]\s*<\s*prev\s*\[button_\]", st) is not None
    if not has_prev:
        return 0.0

    schl = norm(ch)

    # Pattern A: results/list page evidence
    is_results = ("total results" in schl) or (re.search(r"(?i)\[button\]\s*next\s*>\s*\[button_\]", ch) is not None) \
                 or re.search(r"(?i)page\s*\d+", schl) is not None

    # Pattern B: product detail evidence: has price and at least one product-detail button
    has_price = re.search(r"(?i)price\s*:\s*\$?\d+(\.\d{2})?", ch) is not None or re.search(r"(?i)\$\d+(\.\d{2})?", ch) is not None
    has_detail_button = re.search(r"(?i)\[button\]\s*(description|features|buy now|reviews)\s*\[button_\]", ch) is not None

    is_product_detail = has_price and has_detail_button

    if is_results or is_product_detail:
        return 1.0
    else:
        return -1.0

# Rule 30
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = norm(state or "")
    act = norm(action or "")
    ch = norm(choice or "")

    # Only apply for clicks on items or the < Prev button
    is_click_prev = act == "click[< prev]"
    is_click_item = act.startswith("click[item")
    if not (is_click_prev or is_click_item):
        return 0.0

    # Require that the current page looks like a listing or a product detail with a Prev button
    has_listing_marker = ("page" in st and re.search(r"total results", st) is not None) or ("total results" in st)
    has_prev_marker = re.search(r"\[button\]\s*<\s*prev\s*\[button_\]", state or "", re.IGNORECASE) is not None
    # also accept a generic 'page' mention as listing indicator
    has_page_word = re.search(r"\bpage\b", st) is not None

    if not (has_listing_marker or has_prev_marker or has_page_word):
        return 0.0

    # Expected: next page should include an explicit "Price:" line
    has_price = re.search(r"price\s*:", choice or "", re.IGNORECASE) is not None

    return 1.0 if has_price else -1.0

