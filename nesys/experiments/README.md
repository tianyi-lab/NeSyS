# Open-ended WebShop evaluation

This directory contains only the two agent runs reported in the COLM 2026
paper. Both use WebShop sessions 0--99, a 15-step cap, and seed 0.

```bash
# WebShop-7B-SFT baseline: 0.2755 average reward
bash track_a0_webshop_baseline.sh

# NeSyS rule-guided one-step lookahead: 0.3276 average reward
bash track_a2_webshop_wm_rerank.sh
```

The rule-guided run considers at most 10 clickable actions, samples one
successor per action from the Llama-3.2-1B transition model (temperature 0.8,
top-p 0.95), adds unit-weight executable-rule scores, and provides the top
three action--successor pairs to the acting agent. The acting agent makes the
final choice, so this is advisor-style one-step lookahead rather than direct
execution of the WM argmax.

The acting checkpoint is
[`Jianwen/Webshop-7B-SFT`](https://huggingface.co/Jianwen/Webshop-7B-SFT),
released with [SkillRL](https://arxiv.org/abs/2602.08234). Install the original
[WebShop](https://github.com/princeton-nlp/WebShop) environment at
`<repo>/webshop`, or set `WEBSHOP_ROOT=/path/to/WebShop`.

Defaults are defined in `_common.sh` and can be overridden through environment
variables. Extra command-line flags are forwarded to `eval_webshop_agent.py`:

```bash
WEBSHOP_NUM_SESSIONS=5 bash track_a0_webshop_baseline.sh
WEBSHOP_ROOT=/path/to/WebShop bash track_a2_webshop_wm_rerank.sh --resume
```

Outputs are written to `nesys/eval_results_experiments/`, which is ignored by
Git so trajectories and local result files are not accidentally committed.

