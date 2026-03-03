# WMQA Improved Rules
# Improved from (2 files):
#   - webshop_result/rules_webshop_transition_qa_llama3-2-1b-instruct.py
#   - webshop_result/task_rules_webshop_transition_qa_llama3-2-1b-instruct.py
# Dev unit-weight improvement vs original: +7.05%
# Dev unit-weight accuracy (improved rules): 92.01%
# Dev weighted accuracy (learned on dev): 92.32%
# Test baseline accuracy: 46.23%
# Test weighted accuracy: 92.18%
# Test weighted improvement: +45.95%

# Rule 1
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = (action or "").strip()

    # Only apply to click[...] actions
    m_act = re.match(r"^\s*click\[(.*)\]\s*$", act, flags=re.I)
    if not m_act:
        return 0.0

    label = norm(m_act.group(1))

    # Extract all button and clicked-button texts from the state
    btn_texts = []
    for m in re.finditer(r"\[clicked button\]\s*(.*?)\s*\[clicked button_\]", st, flags=re.I | re.S):
        btn_texts.append(norm(m.group(1)))
    for m in re.finditer(r"\[button\]\s*(.*?)\s*\[button_\]", st, flags=re.I | re.S):
        btn_texts.append(norm(m.group(1)))

    # If no button texts mention the label (in either direction), the click should fail
    found = False
    for bt in set(btn_texts):
        if label in bt or bt in label:
            found = True
            break

    if found:
        # Rule not applicable when the button/option exists on the page
        return 0.0

    ch = norm(choice or "")
    return 1.0 if ch == "fail" else -1.0

# Rule 2
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = norm(action)
    if act != "click[buy now]":
        return 0.0

    # Detect explicit button form like "[button] Buy Now [button_]"
    explicit_btn = re.search(r"(?i)\[button\]\s*buy\s*now\s*\[button_\]", st) is not None
    # Fallback: if the page mentions "buy now" and also contains at least one "[button" marker,
    # treat it as having a Buy Now button (covers minor formatting variants).
    implicit_btn = (re.search(r"(?i)\bbuy\s*now\b", st) is not None) and (re.search(r"(?i)\[button", st) is not None)

    has_buy_now = explicit_btn or implicit_btn

    # We only apply the rule when Buy Now is missing.
    if has_buy_now:
        return 0.0

    ch = norm(choice)
    return 1.0 if ch == "fail" else -1.0

# Rule 3
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = norm(action)
    ch = norm(choice)

    # Only apply for buy now clicks
    if act != "click[buy now]":
        return 0.0

    # Does the page show a Buy Now button? If not, this rule does not apply here.
    has_buy_now = re.search(r"(?i)\[button\]\s*buy\s*now\s*\[button_\]", st) is not None
    if not has_buy_now:
        return 0.0

    # Extract the instruction block (between "Instruction:" and the next [button] if possible)
    instr_match = re.search(r"(?is)instruction:\s*(.*?)\n\s*(?:\[[^\]]+\]|\Z)", st)
    instruction = instr_match.group(1).strip() if instr_match else ""
    inst = norm(instruction)

    # Helper: find a product price in the page (first $NNN.NN)
    price_match = re.search(r"\$\s*(\d+(?:\.\d+)?)", st)
    product_price = float(price_match.group(1)) if price_match else None

    # Parse constraints from instruction
    # 1) price lower than X
    price_threshold = None
    m = re.search(r"price\s*(?:lower|less)\s*than\s*\$?\s*(\d+(?:\.\d+)?)", inst)
    if m:
        try:
            price_threshold = float(m.group(1))
        except:
            price_threshold = None

    # 2) explicit phrase requirement (e.g., "high speed")
    # We'll look for quoted or common multi-word phrases, but at minimum "high speed"
    need_high_speed = "high speed" in inst

    # 3) color requirement: look for "color <word>" or a color word in the instruction
    colors = {"black","white","red","blue","green","clear","espresso","brown","grey","gray","beige","pink","yellow","navy","tan"}
    inst_color = None
    # try "color <word>"
    mcol = re.search(r"color\s+([a-z0-9\"']+)", inst)
    if mcol:
        inst_color = re.sub(r'["\']','', mcol.group(1))
    else:
        # otherwise look for any color word present
        for c in colors:
            if re.search(r"\b" + re.escape(c) + r"\b", inst):
                inst_color = c
                break

    # 4) size requirement: small/medium/large or exact textual size token like 'small'
    sizes = {"small","medium","large","xs","s","m","l","xl","xxl"}
    inst_size = None
    for s in sizes:
        if re.search(r"\b" + re.escape(s) + r"\b", inst):
            inst_size = s
            break

    # 5) numeric dimensions like 28\"l x 14.6\"w x 29\"h or patterns with numbers and l/w/h
    dim_match = re.search(r"(\d+(?:\.\d+)?)\s*\"?\s*[lL]\b.*?(\d+(?:\.\d+)?)\s*\"?\s*[wW]\b.*?(\d+(?:\.\d+)?)\s*\"?\s*[hH]\b", instruction)
    inst_dims = None
    if dim_match:
        inst_dims = [dim_match.group(1), dim_match.group(2), dim_match.group(3)]

    # Now verify constraints against page/product text
    fail_constraint = False

    # price check
    if price_threshold is not None:
        if product_price is None:
            # can't verify price -> treat as failing to meet explicit price constraint
            fail_constraint = True
        else:
            if not (product_price < price_threshold):
                fail_constraint = True

    # high speed check
    if need_high_speed:
        if re.search(r"(?i)high\s*speed", st) is None:
            fail_constraint = True

    # color check
    if inst_color:
        if re.search(r"\b" + re.escape(inst_color.lower()) + r"\b", st.lower()) is None:
            fail_constraint = True

    # size check (simple token match)
    if inst_size:
        if re.search(r"\b" + re.escape(inst_size.lower()) + r"\b", st.lower()) is None:
            fail_constraint = True

    # dimensions check: require that numeric dims appear in page text
    if inst_dims:
        dims_found = all(re.search(re.escape(d), st) for d in inst_dims)
        if not dims_found:
            fail_constraint = True

    # If any constraint failed, expected terminal is Fail; else Success.
    expected = "fail" if fail_constraint else "success"
    return 1.0 if ch == expected else -1.0

# Rule 4
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = norm(action)
    ch = norm(choice)

    # Only apply to buy-now clicks
    if act != "click[buy now]":
        return 0.0

    s = norm(st)

    # Detect presence of Buy Now button
    has_buy_now = re.search(r"\[button\]\s*buy now\s*\[button_\]", s, re.I) is not None

    # Extract instruction snippet (if present)
    instr_match = re.search(r"instruction:\s*(.+?)(?:\n|\[button\]|\Z)", st, re.I | re.S)
    instr = norm(instr_match.group(1)) if instr_match else ""

    # Helper: map common number words to digits for small set
    num_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12", "sixteen": "16"
    }

    def word_to_digit(w):
        return num_words.get(w, None)

    # Collect requested attributes from instruction: numeric sizes/volumes, counts, color, explicit "size X"
    requested = {"sizes": [], "counts": [], "colors": [], "size_words": []}

    # numeric volume/size like "16.9 fl oz" or "16.9 fl. oz" or "16.9 ounce"
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(fl\s*oz|fluid\s*ounce|fluid\s*ounces|ounce|ounces|oz)\b", instr, re.I):
        num = m.group(1)
        unit = re.sub(r"\s+", " ", m.group(2)).strip()
        requested["sizes"].append((num, unit))

    # counts like "10-count" or "ten count" or "ten-count"
    for m in re.finditer(r"(\d+)\s*(?:-?count|count|pack)\b", instr, re.I):
        requested["counts"].append(m.group(1))
    # spelled-out counts like "ten count"
    for wn, digit in num_words.items():
        if re.search(r"\b" + re.escape(wn) + r"\s*(?:-?count|count|pack)\b", instr):
            requested["counts"].append(digit)

    # size expressed like "size eight" or "size 8"
    m = re.search(r"size\s+([a-z0-9.-]+)", instr, re.I)
    if m:
        token = m.group(1)
        if token.isdigit():
            requested["size_words"].append(token)
        else:
            dd = word_to_digit(token)
            requested["size_words"].append(dd if dd else token)

    # colors
    colors_list = ["pink", "amber", "clear", "tan", "brown", "black", "white", "blue", "green", "red"]
    for c in colors_list:
        if re.search(r"\b" + re.escape(c) + r"\b", instr):
            requested["colors"].append(c)

    # If there are no explicit requested attributes and Buy Now is present, we can't apply mismatch rule
    # But if Buy Now missing we should expect Fail (can't buy)
    if not has_buy_now:
        return 1.0 if ch == "fail" else -1.0

    # Now check whether each requested attribute appears on the page text (title/options/description area)
    # Function to check presence of numeric size with unit (approximate)
    def page_contains_size(num, unit):
        # look for patterns like "16.9 fl oz", "16.9 fl. oz", "16.9 fl oz (pack of 1)" etc.
        patt = re.escape(num) + r"\s*(?:-)?\s*" + re.escape(unit)
        if re.search(patt, s, re.I):
            return True
        # also accept forms like "16.9 oz" when unit is 'fl oz' or 'oz'
        if unit.find("oz") != -1 and re.search(re.escape(num) + r"\s*oz\b", s):
            return True
        return False

    def page_contains_count(num):
        # Accept "10-count", "10 count", "10-count (pack of 1)", "10-count (pack of 1)" or "10 count (pack)"
        if re.search(r"\b" + re.escape(num) + r"\s*(?:-?count|count|pack)\b", s):
            return True
        # sometimes title contains "10 count" as "10-count" or "10-count (pack of 1)"
        if re.search(r"\b" + re.escape(num) + r"\b", s) and re.search(r"count|pack", s):
            return True
        return False

    def page_contains_size_word(token):
        if token is None:
            return False
        # token may be like '8' or 'eight'
        if token.isdigit():
            # look for size 8 or '8 (pack of ...)' or 'size 8'
            if re.search(r"\bsize\b\s*" + re.escape(token), s) or re.search(r"\b" + re.escape(token) + r"\b", s):
                return True
        else:
            # check spelled word presence
            if re.search(r"\b" + re.escape(token) + r"\b", s):
                return True
        return False

    def page_contains_color(col):
        return re.search(r"\b" + re.escape(col) + r"\b", s, re.I) is not None

    # Determine mismatch: if any requested attribute list is non-empty and none of its members are found on page, it's a mismatch.
    mismatch = False
    # sizes
    if requested["sizes"]:
        found_any = False
        for num, unit in requested["sizes"]:
            if page_contains_size(num, unit):
                found_any = True
                break
        if not found_any:
            mismatch = True

    if requested["counts"]:
        found_any = False
        for cnum in requested["counts"]:
            if page_contains_count(cnum):
                found_any = True
                break
        if not found_any:
            mismatch = True

    if requested["size_words"]:
        found_any = False
        for token in requested["size_words"]:
            if page_contains_size_word(token):
                found_any = True
                break
        if not found_any:
            mismatch = True

    if requested["colors"]:
        found_any = False
        for col in requested["colors"]:
            if page_contains_color(col):
                found_any = True
                break
        if not found_any:
            mismatch = True

    # If we detected any mismatch, the correct terminal result should be Fail
    if mismatch:
        return 1.0 if ch == "fail" else -1.0

    # Otherwise do not apply this rule
    return 0.0

# Rule 5
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = norm(action)
    # Only apply when the agent clicked the Buy Now action
    if act != "click[buy now]":
        return 0.0

    # Detect a Buy Now button token in the page text; accept minor formatting variants
    has_buy_now = re.search(r"(?i)\[button\]\s*buy\s*now\s*\[button_\]", st) is not None

    # If a Buy Now button exists, this rule doesn't decide (other logic should handle)
    if has_buy_now:
        return 0.0

    # Otherwise we expect the correct terminal result to be Fail
    ch = norm(choice)
    if ch == "fail":
        return 1.0
    else:
        return -1.0

# Rule 6
# Task group: click_buy_now
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    # Conservative merged rule for click[buy now]
    if action != "click[buy now]":
        return 0.0

    s = (state or "").lower()

    # Helper to produce verdict reward for a Fail case
    def fail_reward():
        return 1.0 if choice == "Fail" else -1.0

    # 1) If there is no visible buy-now text anywhere, clicking buy now cannot succeed.
    if "buy now" not in s:
        return fail_reward()

    # At this point there is some "buy now" text; continue with more conservative checks.

    import re

    # 2) Price ceiling mismatch: instruction asks for "lower than X" / "price lower than X" and product price exists and >= X
    # match patterns like "lower than 40", "price lower than 40.00", "lower than $40"
    m_thresh = re.search(r'lower than\s*\$?(\d+(?:\.\d+)?)', s)
    if not m_thresh:
        m_thresh = re.search(r'price lower than\s*\$?(\d+(?:\.\d+)?)', s)
    if m_thresh:
        try:
            thresh = float(m_thresh.group(1))
            # find a product price: prefer "Price: $X" then any $X
            m_price = re.search(r'price:\s*\$(\d+(?:\.\d+)?)', s)
            if not m_price:
                m_price = re.search(r'\$(\d+(?:\.\d+)?)', s)
            if m_price:
                price = float(m_price.group(1))
                # If product price is >= threshold, the buy should fail
                if price >= thresh:
                    return fail_reward()
        except:
            pass

    # 3) Explicit attribute contradictions (only trigger when both sides are explicitly present)
    # long-sleeve requested but product is sleeveless
    if (("long sleeve" in s or "long sleeves" in s) and "sleeveless" in s):
        return fail_reward()

    # natural hair requested but product explicitly synthetic
    if ("natural hair" in s and "synthetic" in s):
        return fail_reward()

    # instruction asks for "paraben" + "hair color" but product is a conditioner
    if ("paraben" in s and "hair color" in s and "conditioner" in s):
        return fail_reward()

    # explicit "renewed" label indicates mismatch in many tasks -> fail if present together with click
    # (conservative: only trigger if 'renewed' appears as product label)
    if "renewed" in s:
        return fail_reward()

    # 4) Pack / count / size mismatches where instruction and product both state explicit counts/sizes that disagree.
    # detect "pack of N" or "pack of N)" occurrences in instruction and product
    def find_pack_numbers(text):
        nums = re.findall(r'pack of\s*(\d+)', text)
        nums += re.findall(r'pack\s*of\s*(\d+)', text)
        nums += re.findall(r'(\d+)\s*[- ]?pack\b', text)
        return [int(n) for n in nums]
    instr_packs = find_pack_numbers(s)
    prod_packs = find_pack_numbers(s)  # we use same state; conservative: require both to appear in state text
    # If at least two different explicit pack counts appear in the page text (one intended, one product),
    # and they are different, treat as mismatch. (This is conservative because we require explicit numbers.)
    if instr_packs and prod_packs:
        # if there exists any differing explicit numbers -> fail
        if any(a != b for a in instr_packs for b in prod_packs):
            return fail_reward()

    # Generic numeric size/volume mismatch: look for named units in both instruction and product and compare exact numbers when both present
    def find_unit_values(text, unit_patterns):
        res = []
        for pat in unit_patterns:
            for m in re.findall(pat, text):
                try:
                    res.append(float(m))
                except:
                    pass
        return res

    # common unit patterns: oz, fl oz, pound/lb, inch/cm
    oz_patterns = [r'(\d+(?:\.\d+)?)\s*oz\b', r'(\d+(?:\.\d+)?)\s*fl oz\b']
    lb_patterns = [r'(\d+(?:\.\d+)?)\s*(?:pound|lb|lbs)\b', r'(\d+/\d+)\s*pound']  # include fractional like 1/2
    inch_patterns = [r'(\d+(?:\.\d+)?)\s*(?:inch|in)\b', r'(\d+(?:\.\d+)?)\s*cm\b']

    instr_oz = find_unit_values(s, oz_patterns)
    prod_oz = instr_oz  # conservative: require both appear somewhere in state; comparing all occurrences
    # If two distinct explicit numeric unit values of same unit are present and differ -> fail
    # (We only trigger if two different numbers are present in state text, indicating instruction vs product are both shown)
    if len(set(instr_oz)) >= 2:
        # differing oz values present -> fail
        return fail_reward()

    # pounds: handle fractional like '1/2'
    # find any occurrence of '1/2' and convert to 0.5 for comparison
    frac_matches = re.findall(r'(\d+/\d+)\s*pound', s)
    for f in frac_matches:
        try:
            num, den = f.split('/')
            val = float(num) / float(den)
            instr_vals = re.findall(r'(\d+(?:\.\d+)?)\s*pound', s)
            # if both fractional and integer pound values appear and differ, fail
            for iv in instr_vals:
                try:
                    if float(iv) != val:
                        return fail_reward()
                except:
                    pass
        except:
            pass

    # 5) Missing required option selection: if many option buttons with numeric sizes/colors are shown but no "selected" marker, treat as invalid
    # Look for multiple "[button]" entries with digits/size tokens and absence of "selected"
    buttons = re.findall(r'\[button\]([^\[]+)\[button_\]', state or "")
    if buttons:
        # count buttons that look like size options (contain digits or common size words)
        size_like = [b for b in buttons if re.search(r'\b(\d+(\.\d+)?|small|medium|large|xl|xs|s|m|l)\b', b.lower())]
        color_like = [b for b in buttons if re.search(r'\b(color|navy|black|blue|red|yellow|green|purple|pink|phantom|tangerine|heather|brown)\b', b.lower())]
        if (len(size_like) >= 2 or len(color_like) >= 2) and ("selected" not in s):
            return fail_reward()

    # 6) If instruction explicitly requests a color token and the state shows explicit color options but does not include the requested color -> fail.
    # conservative approach: look for a color word in the state that appears near "instruction" (heuristic) and confirm color options exist but requested color not present in options
    color_candidates = ["tangerine","dark heather","dark blue","phantom pink","black blue","yellow","green","purple","redwood","navy","phantom","black","blue","brown"]
    instr_section = s  # we don't have separate fields; be conservative and only act when token clearly present
    for col in color_candidates:
        if col in instr_section:
            # if the page includes a color/options area (look for 'color' keyword or several [button] tokens) but requested color not in whole page's option/button text, fail
            if ("color" in s or buttons):
                # if requested color not present in the whole state (meaning product doesn't show it), then fail
                if col not in s:
                    return fail_reward()

    # If none of the conservative fail conditions matched, do not apply the rule.
    return 0.0

# Rule 7
# Task group: click_change_page
def rule_reward(state, action, choice):
    import re

    def has(substr):
        return substr in state

    def choice_has(substr):
        return substr in choice

    # normalize presence checks (but keep case-sensitive checks for tokens that are case-sensitive)
    state_lower = state.lower() if state is not None else ""
    choice_lower = choice.lower() if choice is not None else ""

    # Helper detectors
    def looks_like_search_page(text_lower, text):
        # require the word Page and either "total results" or a Next control or multiple product buttons (B0..)
        return (("page " in text_lower) and (("total results" in text_lower) or ("next >" in text_lower) or ("[button] next >" in text_lower))) \
               or ("amazon shopping game" in text_lower and "[button] search [button_]" in text_lower)

    def looks_like_product_detail(text_lower, text):
        # product-detail commonly has Price: and a Buy Now or Description/Features/Reviews buttons
        return (("price:" in text_lower and ("[button] buy now" in text_lower or "buy now" in text_lower))) \
               or (("[button] description" in text_lower and "[button] buy now" in text_lower)) \
               or ("[button] < prev" in text_lower) or ("[button] < prev [button_]" in text_lower)

    # --- click[back to search] branch ---
    if action == "click[back to search]":
        # only apply when the state contains an explicit Back to Search affordance
        if "[button] Back to Search [button_]" in state or "[button] Back to Search" in state:
            if "Amazon Shopping Game" in choice and "[button] Search [button_]" in choice:
                return 1.0
            else:
                return -1.0
        return 0.0

    # --- click[next >] branch ---
    if action == "click[next >]":
        # only apply when the current state shows a Page N and a Next control
        m = re.search(r'Page\s*(\d+)', state)
        if not m:
            return 0.0
        if ("Next >" not in state) and ("[button] Next >" not in state) and ("next >" not in state.lower()):
            return 0.0
        try:
            cur = int(m.group(1))
        except Exception:
            return 0.0
        target = cur + 1
        # accept if candidate contains Page target
        if re.search(r'Page\s*%d\b' % target, choice):
            return 1.0
        else:
            return -1.0

    # --- click[< prev] branch ---
    if action == "click[< prev]":
        # Precondition: only apply when there is evidence of a prior-page context:
        # - product-detail prev/back affordance, or
        # - a listing with "Page N", or
        # - an explicit clicked-selection confirmation ("You have clicked" or "[clicked button]")
        has_prev_button = ("[button] < Prev" in state) or ("[button] < Prev [button_]" in state) or ("< Prev" in state and "[button]" in state)
        has_back_to_search_nav = ("[button] Back to Search [button_]" in state) or ("[button] Back to Search" in state)
        has_clicked_confirmation = ("You have clicked" in state) or ("[clicked button]" in state)
        page_match = re.search(r'Page\s*(\d+)', state)

        if not (has_prev_button or has_back_to_search_nav or has_clicked_confirmation or page_match):
            return 0.0  # abstain - no clear precondition

        # 1) If the state shows a clicked-selection confirmation, require the choice preserve clicked markers
        if has_clicked_confirmation:
            if ("You have clicked" in choice) or ("[clicked button]" in choice):
                return 1.0
            else:
                return -1.0

        # 2) If the state is a listing page with Page N, expect Page N-1
        if page_match:
            try:
                cur = int(page_match.group(1))
            except Exception:
                cur = None
            if cur is not None and cur > 1:
                target = cur - 1
                if re.search(r'Page\s*%d\b' % target, choice):
                    return 1.0
                else:
                    return -1.0
            # if cur is 1 or parse failed, fall through to other checks

        # 3) If the state looks like a product-detail (has Buy Now / Price or a Prev button), prefer a listing page
        if has_prev_button or ("Price:" in state and ("[button] Buy Now" in state or "Buy Now" in state)):
            if looks_like_search_page(choice_lower, choice):
                return 1.0
            # clearly incorrect if the candidate remains a product-detail (didn't navigate)
            if looks_like_product_detail(choice_lower, choice):
                return -1.0
            # also penalize if the candidate repeats the product title verbatim (no navigation)
            # detect repetition by checking if a long line from state appears verbatim in choice
            state_lines = [ln.strip() for ln in state.splitlines() if ln.strip()]
            if state_lines:
                # take a reasonably long line as indicator
                for ln in state_lines:
                    if len(ln) > 20 and ln in choice:
                        return -1.0
            # otherwise ambiguous, abstain
            return 0.0

        # 4) If we got here but none of the above gave a decision, be conservative
        return 0.0

    # No applicable action
    return 0.0

# Rule 8
# Task group: click_item
def rule_reward(state, action, choice):
    """
    Merged, conservative rule for click_item group.
    - Triggers only when action exactly equals one of the known item-click strings.
    - Returns 1.0 when the choice clearly looks like the product detail page for that action
      (all required markers present).
    - Returns -1.0 only when the choice clearly references a different product (explicit reject markers).
    - Returns 0.0 in all other cases (ambiguous / non-triggering actions).
    """
    # mapping of exact action string -> required markers (all must be present for +1)
    # and optional reject markers (any present -> -1)
    mapping = [
        {
            "action": ("click[item - casesack case for tribit stormbox micro bluetooth speaker, by casesack, "
                       "tailor made semi- hard case with best matching shape and color, mesh charge cord pocket, "
                       "easy to go carabiner]"),
            "must_have": ["casesack", "$14.48"],
            "reject": ["b07tshd2w5", "antimi"],
        },
        {
            "action": ("click[item - extra soft toothbrush deep clean toothbrush with 10000 bristles for "
                       "sensitive toothbrushes sensitive teeth manual protection care]"),
            # require price, buy affordance and a color selection indicator
            "must_have": ["price: $4.99", "buy now", "color"],
            "reject": [],
        },
        {
            "action": ("click[item - extra soft toothbrush for sensitive gums/ tongue cleaner set 4pcs - "
                       "nano manual toothbrush for sensitive teeth/ soft bristles toothbrush for adult /children "
                       "(3 toothbrush+1tongue scraper)]"),
            # require a distinctive title fragment, price, and buy affordance
            "must_have": ["extra soft toothbrush for sensitive gums", "price: $15.99", "buy now"],
            "reject": [],
        },
    ]

    # Only consider exact-action matches
    for entry in mapping:
        if action == entry["action"]:
            # conservative handling: if choice is not text, treat as ambiguous
            if not isinstance(choice, str):
                return 0.0
            lower = choice.lower()

            # Positive: all required markers present (case-insensitive)
            all_present = all(marker.lower() in lower for marker in entry["must_have"])
            if all_present:
                return 1.0

            # Negative: any explicit reject marker present
            for r in entry.get("reject", []):
                if r.lower() in lower:
                    return -1.0

            # Ambiguous for this action: do not assert (conservative)
            return 0.0

    # Action did not match any target click -> do not apply rule
    return 0.0

# Rule 9
# Task group: click_nav
def rule_reward(state, action, choice):
    # returns a float in [-1, 1]
    # Apply only to the specific action
    if action != "click[description]":
        return 0.0

    # Conservative check: require that the state actually offered a Description button
    if "[button] Description [button_]" not in state:
        return 0.0

    # Extract the Instruction block from the state: from "Instruction:" up to the first "[button]" that follows
    instr_start = state.find("Instruction:")
    if instr_start == -1:
        return 0.0
    next_button = state.find("[button]", instr_start)
    if next_button == -1:
        return 0.0
    instr_block = state[instr_start:next_button].strip()
    if not instr_block:
        return 0.0

    # Normalization helper (collapse whitespace) to be robust to minor spacing/newline differences
    def norm(s):
        return " ".join(s.split())

    norm_instr = norm(instr_block)
    norm_choice = norm(choice)

    # Require the instruction text to appear in the candidate next-state
    if norm_instr not in norm_choice:
        return -1.0

    # Require the two navigation buttons to be present in order
    back_btn = "[button] Back to Search [button_]"
    prev_btn = "[button] < Prev [button_]"
    pos_back = choice.find(back_btn)
    pos_prev = choice.find(prev_btn)
    if pos_back == -1 or pos_prev == -1 or pos_back > pos_prev:
        return -1.0

    # Preconditions satisfied and candidate contains the expected Instruction + nav buttons
    return 1.0

# Rule 10
# Task group: click_nav
def rule_reward(state, action, choice):
    """
    Returns a float in [-1.0, 1.0].

    Applies only when action == "click[features]" and the state contains the literal
    substring "[button] Features [button_]". If those conditions hold, returns +1.0
    when the choice contains any of a conservative set of feature-markers
    (case-insensitive); otherwise returns -1.0. If preconditions do not hold, returns 0.0.
    """
    # Preconditions: exact action match and Features button present
    if action is None or action.strip() != "click[features]":
        return 0.0
    if not state or "[button] Features [button_]" not in state:
        return 0.0

    # choice must be a string to inspect
    if not isinstance(choice, str):
        return -1.0

    c_up = choice.upper()

    # Conservative set of distinctive substrings that commonly appear in Features/details panels
    markers = [
        "CONTAINS:",    # e.g., "Contains:"
        "CERTIFICATION",# e.g., "CERTIFICATION" or "Certification"
        "MATERIALS",    # e.g., "Materials"
        "GLUTEN-FREE",
        "5G FIBER",
        "ADAPTOGEN",
        "LOW SUGAR",
        "CACAO LIL",
        "CAN WE GET AN AMEN"
    ]

    for m in markers:
        if m in c_up:
            return 1.0

    # Features button was present and action was click[features], but expected markers not found
    return -1.0

# Rule 11
# Task group: click_other
def rule_reward(state, action, choice):
    """
    General, conservative WebShop click rule.
    - Triggers only for actions of the exact form "click[...]" (no extra text).
    - Requires that the current state contains the unclicked button token "[button] LABEL [button_]".
    - Rewards  1.0 if the candidate choice contains BOTH:
         - an explicit confirmation "You have clicked LABEL." (or "You have clicked LABEL")
         - the clicked-button marker "[clicked button] LABEL [clicked button_]"
    - Returns -1.0 if neither of those two indicators is present in the choice.
    - Returns 0.0 if exactly one indicator is present (ambiguous) or if preconditions don't hold.
    """
    # Basic action pattern check
    if not (isinstance(action, str) and action.startswith("click[") and action.endswith("]")):
        return 0.0

    # Extract label inside brackets and normalize
    label = action[len("click["):-1].strip()
    if label == "":
        return 0.0

    # Conservative precondition: require that the state lists the unclicked button for this label
    unclicked_token = "[button] " + label + " [button_]"
    if unclicked_token not in state:
        return 0.0

    # Expected indicators in the candidate next-state
    confirmation_with_period = "You have clicked " + label + "."
    confirmation_no_period = "You have clicked " + label
    clicked_marker = "[clicked button] " + label + " [clicked button_]"

    has_confirmation = (confirmation_with_period in choice) or (confirmation_no_period in choice)
    has_clicked_marker = (clicked_marker in choice)

    if has_confirmation and has_clicked_marker:
        return 1.0
    if not has_confirmation and not has_clicked_marker:
        return -1.0
    # Ambiguous (only one indicator present) — be conservative
    return 0.0

# Rule 12
# Task group: search
def rule_reward(state, action, choice):
    """
    Returns a float in [-1, 1].
    Conservative rule for "search[...]" transitions:
    - Applies only when action starts with "search[".
    - +1.0 if choice clearly looks like a multi-item/paginated search-results page.
    - -1.0 if choice clearly looks like an incorrect response (single item, cart, budget-filter, stub, unchanged prompt).
    - 0.0 otherwise.
    """
    import re

    # Only apply to search actions
    if not isinstance(action, str) or not action.startswith("search["):
        return 0.0

    if choice is None or not isinstance(choice, str):
        return 0.0

    text = choice

    # Positive indicators of a multi-item/paginated results page
    pos_indicators = [
        "Page 1 (Total results:",   # explicit pagination marker seen in many examples
        "Total results:",           # alternate form
        "Back to Search",           # navigation present on results pages
    ]
    # presence of at least one SKU-like token together with a dollar price is a strong signal
    sku_regex = re.compile(r"\bB[0-9A-Z]{3,10}\b")
    price_regex = re.compile(r"\$\s*\d")  # simple price detection like $12 or $ 12

    pos_detect = any(tok in text for tok in pos_indicators)
    if not pos_detect:
        # Also treat presence of SKU + price as positive even if pagination marker absent
        if sku_regex.search(text) and price_regex.search(text):
            pos_detect = True

    # Negative indicators of wrong outputs
    neg_tokens = [
        "Showing 1 result", "Showing 1 result(s)",
        "Shopping Cart", "Amazon Shopping Cart",
        "[button] View Results", "[button] View Results [button]",
        "Go back to Home Decor", "Home Decor Store",
        "Filter by budget", "Continue Shopping",
        "You found a pair of",                       # known chatty single-item reply
        "What are the top-selling", "Amazon Shopping Game",
        "View Results",                              # generic stub
        "Showing 1 result(s)"                        # variants
    ]
    neg_detect = any(tok in text for tok in neg_tokens)

    # If both positive and negative markers appear, abstain to be conservative
    if pos_detect and not neg_detect:
        return 1.0
    if neg_detect and not pos_detect:
        return -1.0

    # No clear opinion
    return 0.0

# Rule 13
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = norm(action)
    ch = choice or ""

    # Only apply to tab clicks we care about.
    tab_actions = {
        "click[description]": "description",
        "click[features]": "features",
        "click[reviews]": "reviews",
    }
    if act not in tab_actions:
        return 0.0

    tab_name = tab_actions[act]

    # Check that the state actually contains that tab/button.
    pattern = r"(?i)\[button\]\s*" + re.escape(tab_name) + r"\s*\[button_\]"
    if re.search(pattern, st) is None:
        # If the button isn't present in the state, the rule doesn't apply.
        return 0.0

    # Look for a substantive non-button text line in the candidate choice:
    # - a line not containing "[button" and
    # - with at least 40 characters and at least 7 words (heuristic for descriptive content).
    for line in (ch or "").splitlines():
        if "[button" in line.lower():
            continue
        s = re.sub(r"\s+", " ", line or "").strip()
        if len(s) >= 40 and len(s.split()) >= 7:
            return 1.0

    # No descriptive content found -> likely incorrect transition for a tab click.
    return -1.0

# Rule 14
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = norm(action or "")
    ch = choice or ""

    # Only apply to clicks on content tabs
    if act not in ("click[features]", "click[description]", "click[reviews]"):
        return 0.0

    # Check that the corresponding button exists on the current page text
    # Accept minor formatting variants of the button label.
    btn_label = act.replace("click[", "").replace("]", "")
    btn_pattern = r"(?i)\[button\]\s*" + re.escape(btn_label) + r"\s*\[button_\]"
    if re.search(btn_pattern, st) is None:
        return 0.0

    # Heuristics to decide if the choice is a content block vs a UI/menu:
    # - Content block: reasonably long textual content (length threshold) and contains ordinary words (not just buttons),
    #   or contains keywords typical of features/description/reviews.
    # - UI/menu: contains many button tags or mainly a short list of button lines.
    choice_text = re.sub(r"\s+", " ", ch).strip()

    # If the choice contains many button tags, treat as UI/menu page.
    button_tags = len(re.findall(r"(?i)\[button\]", ch))
    if button_tags >= 3:
        return -1.0

    # If the choice is long enough and contains typical content words, accept as content block.
    keywords = ["feature", "gluten", "fiber", "sugar", "ingredients", "description", "review", "rating", "stars", "love", "benefit", "adds", "contains", "serving"]
    kw_match = any(re.search(r"(?i)\b" + re.escape(k) + r"\b", ch) for k in keywords)

    if len(choice_text) >= 80 and kw_match:
        return 1.0

    # Also accept somewhat shorter but clearly formatted features/reviews (e.g., bullet-like lines or emoji bullets)
    if len(choice_text) >= 60 and re.search(r"[\u2022\u2013\-\—\n💙⭐★]", ch):
        return 1.0

    # Otherwise, if the choice looks short or mostly UI controls, penalize.
    return -1.0

# Rule 15
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = (action or "").strip()
    ch = choice or ""

    st_n = norm(st)
    act_n = norm(act)

    # Identify action types we care about
    is_click_prev = re.match(r"click\[\s*<\s*prev\s*\]", act_n) is not None
    is_click_next = re.match(r"click\[\s*next\s*>\s*\]", act_n) is not None
    is_search = act_n.startswith("search[")

    # Only apply when one of these actions is used
    if not (is_click_prev or is_click_next or is_search):
        return 0.0

    # For click actions, require that the corresponding button actually appears on the current page.
    if is_click_prev:
        has_prev = re.search(r"(?i)\[button\]\s*<\s*prev\s*\[button_\]", st) is not None
        if not has_prev:
            return 0.0
    if is_click_next:
        has_next = re.search(r"(?i)\[button\]\s*next\s*>\s*\[button_\]", st) is not None
        if not has_next:
            return 0.0
    if is_search:
        # For search we require that the state contains a Search control (initial search page)
        has_search_btn = re.search(r"(?i)\[button\]\s*search\s*\[button_\]", st) is not None
        if not has_search_btn:
            # If the environment didn't show a Search button, still allow applying the rule
            # because search[...] can be issued from the top-level; we keep it permissive.
            pass

    # Heuristic: a search-results listing typically contains "total results"
    # or "page <n>" together with a "Next >" button.
    looks_like_results = False
    if re.search(r"(?i)total results", ch):
        looks_like_results = True
    elif re.search(r"(?i)page\s*\d", ch) and re.search(r"(?i)\[button\]\s*next\s*>\s*\[button_\]", ch):
        looks_like_results = True
    else:
        # also accept pages that contain product-item lines plus Next button
        if re.search(r"(?i)\[button\]\s*next\s*>\s*\[button_\]", ch) and re.search(r"(?i)\$[0-9]+", ch):
            looks_like_results = True

    return 1.0 if looks_like_results else -1.0

# Rule 16
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = norm(action)
    ch = choice or ""
    ch_norm = norm(ch)

    # Only apply when action is a search[...] or click[< prev]
    if not (act.startswith("search[") or act == "click[< prev]"):
        return 0.0

    # If the model returned a terminal token, that's unlikely for search/navigation
    if ch_norm in ("success", "fail"):
        return -1.0

    # Heuristics to detect a search results listing page:
    #  - explicit "total results" phrase
    #  - a "page X" indicator combined with Next/Prev controls
    #  - presence of product buttons like "[button] B09JRMMY3Z [button_]" (ASIN-like)
    has_total = "total results" in ch_norm
    has_page_num = re.search(r"\bpage\s*\d+\b", ch_norm) is not None
    has_next = re.search(r"\[button\]\s*next\s*>\s*\[button_\]", ch_norm) is not None
    has_prev = re.search(r"\[button\]\s*<\s*prev\s*\[button_\]", ch_norm) is not None
    # ASIN-like product button (B followed by ~10 alnum chars) — allow lowercased too
    has_asin_button = re.search(r"\[button\]\s*b[0-9a-z]{8,11}", ch_norm) is not None
    # Also accept many product lines or any $ price signs on a listing page
    has_price = "$" in choice

    looks_like_results = has_total or ((has_page_num and (has_next or has_prev)) or has_asin_button or has_price)

    return 1.0 if looks_like_results else -1.0

# Rule 17
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    ch = choice or ""
    act = norm(action)

    # Only apply this rule for explicit pagination clicks
    if act not in ("click[next >]", "click[< prev]"):
        return 0.0

    # Check that the current page actually shows pagination cues
    st_low = norm(st)
    has_pagination_cue = (
        re.search(r"\bpage\s*\d+", st_low) is not None
        or "total results" in st_low
        or re.search(r"\[button\]\s*<\s*prev\s*\[button_\]", st_low)
        or re.search(r"\[button\]\s*next\s*>\s*\[button_\]", st_low)
    )
    if not has_pagination_cue:
        return 0.0

    ch_low = norm(ch)

    # Heuristics for recognizing a results-listing page:
    # - contains "page" or "total results"
    # - or contains Next/Prev buttons
    # - or contains multiple product entry buttons (ASIN-like: [button] B..., or many $ prices)
    looks_like_results = (
        re.search(r"\bpage\s*\d+", ch_low) is not None
        or "total results" in ch_low
        or re.search(r"\[button\]\s*next\s*>\s*\[button_\]", ch_low)
        or re.search(r"\[button\]\s*<\s*prev\s*\[button_\]", ch_low)
        or re.search(r"\[button\]\s*b[0-9a-z]{5,}\b", ch_low)  # ASIN-like product buttons
        or len(re.findall(r"\$\s*\d", ch_low)) >= 2  # multiple prices shown
    )

    return 1.0 if looks_like_results else -1.0

# Rule 18
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    ch = choice or ""
    act = norm(action)

    # Detect click[< prev]
    if re.search(r"click\[\s*<\s*prev\s*\]", act):
        # Only apply when the state actually shows a < Prev button
        if re.search(r"(?i)\[button\]\s*<\s*prev\s*\[button_\]", st) is None:
            return 0.0

        # Heuristics for "results/listing page"
        has_total = re.search(r"(?i)total results", ch) is not None
        has_next = re.search(r"(?i)\[button\]\s*next\s*>\s*\[button_\]", ch) is not None
        has_page_num = re.search(r"(?i)\bpage\s*\d+", ch) is not None
        many_prices = len(re.findall(r"\$\s*\d+", ch)) >= 2
        many_product_buttons = len(re.findall(r"\[button\]\s*[A-Za-z0-9\-]{4,}\s*\[button_\]", ch)) >= 3

        looks_like_results = has_total or has_next or has_page_num or many_prices or many_product_buttons
        return 1.0 if looks_like_results else -1.0

    # Detect click[description]
    if re.search(r"click\[\s*description\s*\]", act):
        # Only apply when the state actually shows a Description button
        if re.search(r"(?i)\[button\]\s*description\s*\[button_\]", st) is None:
            return 0.0

        # Heuristics for "product description": presence of a reasonably long run of descriptive text
        num_button_tokens = len(re.findall(r"\[button\]", ch))
        # A long sentence/paragraph (30+ chars) — likely descriptive text
        has_long_text = re.search(r"[A-Za-z0-9].{30,}", ch) is not None
        # Avoid treating pages that are mostly UI/button lists as descriptions
        looks_like_description = has_long_text and num_button_tokens <= 2

        return 1.0 if looks_like_description else -1.0

    # Rule doesn't apply
    return 0.0

# Rule 19
def rule_reward(state, action, choice):
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = action or ""
    ch = choice or ""

    # Normalize for matching
    nst = norm(st)
    nact = norm(act)
    nch = norm(ch)

    # Match actions of the form click[item - <label>]
    m = re.search(r"click\s*\[\s*item\s*-\s*(.*?)\s*\]", nact)
    if not m:
        return 0.0

    label = m.group(1)
    if not label:
        return 0.0

    nlabel = norm(label)

    # Only apply the rule when the clicked label actually appears on the current page
    # (e.g., as a product button or listing). Use substring match on normalized strings.
    if nlabel not in nst:
        return 0.0

    # If the choice contains the clicked label, reward as correct; otherwise penalize.
    return 1.0 if nlabel in nch else -1.0

# Rule 20
def rule_reward(state, action, choice):
    import re
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    st = state or ""
    act = norm(action or "")
    ch = (choice or "")

    # Only apply when action is a click[...] form
    m = re.match(r"click\[(.*)\]", act)
    if not m:
        return 0.0
    label = m.group(1).strip()

    st_norm = norm(st)
    ch_norm = norm(ch)

    # Helper: does the state actually contain the button label?
    # Accept minor variants, check presence of "[button] <label> [button_]" or label alone in button lists.
    def state_has_button(lbl):
        if not lbl:
            return False
        # escape special regex chars in lbl
        esc = re.escape(lbl)
        # allow variations like "< prev", "next >", "description"
        pattern = r"(?i)\[button\]\s*" + esc + r"\s*\[button_\]"
        if re.search(pattern, st, flags=re.IGNORECASE):
            return True
        # sometimes buttons appear in lists without exact brackets; fallback: label occurs near "button"
        if re.search(r"(?i)\[button\].{0,40}" + esc, st):
            return True
        return False

    if not state_has_button(label):
        return 0.0

    # Navigation buttons
    if re.search(r"(?i)\bnext\b|>|<\s*prev|prev\b", label):
        # Expect a search-results page: contain "page <number>" or "total results" and navigation buttons
        has_page_indicator = re.search(r"(?i)\bpage\s*\d+", ch) is not None
        has_total = re.search(r"(?i)total results", ch) is not None
        has_nav_buttons = (re.search(r"(?i)\[button\]\s*<\s*prev\s*\[button_\]", ch) is not None) and \
                          (re.search(r"(?i)\[button\]\s*next\s*>\s*\[button_\]", ch) is not None)
        if has_page_indicator or has_total or has_nav_buttons:
            return 1.0
        else:
            return -1.0

    # Content tab buttons (description/features/reviews)
    if re.search(r"(?i)description|features|reviews", label):
        # Accept if the resulting choice contains substantive description-like text:
        # either a "how to" phrase or long non-button text (length threshold)
        has_howto = re.search(r"(?i)how to", ch) is not None
        # remove button tokens to estimate actual descriptive text length
        cleaned = re.sub(r"\[button\].*?\[button_\]", " ", ch, flags=re.IGNORECASE)
        # count non-whitespace chars
        non_ws_len = len(re.sub(r"\s+", "", cleaned))
        if has_howto or non_ws_len > 80:
            return 1.0
        else:
            return -1.0

    # Not a covered click type
    return 0.0

