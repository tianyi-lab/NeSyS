# NeSyS: Neuro-Symbolic Synergy for Interactive World Modeling

This is the official code and data release for the paper:

> **Neuro-Symbolic Synergy for Interactive World Modeling**
> Hongyu Zhao, Siyu Zhou, Haolin Yang, Zengyi Qin, Tianyi Zhou
> [arXiv:2602.10480](https://arxiv.org/abs/2602.10480)

Large language models exhibit strong general-purpose reasoning capabilities, yet they frequently hallucinate when used as world models. NeSyS bridges this gap by integrating the probabilistic semantic priors of LLMs with executable symbolic rules. The symbolic world model directly constrains the LLM by modifying its output probability distribution, and the neural world model is fine-tuned only on transitions not covered by symbolic rules -- reducing training data by 50% without loss of accuracy.

## Results

<table>
  <thead>
    <tr>
      <th align="left">Backbone</th>
      <th align="left">Method</th>
      <th align="center">ScienceWorld</th>
      <th align="center">Webshop</th>
      <th align="center">Plancraft</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Strong Baseline</strong></td>
      <td>GPT-5-mini</td>
      <td align="center">55.4</td>
      <td align="center">81.4</td>
      <td align="center">73.8</td>
    </tr>
    <tr>
      <td rowspan="2"><strong>Llama3.2-1B</strong></td>
      <td>SFT (100% data)</td>
      <td align="center">64.4</td>
      <td align="center">47.5</td>
      <td align="center">80.5</td>
    </tr>
    <tr>
      <td><strong>Ours</strong> (Reduced data)</td>
      <td align="center"><strong>68.3</strong> <br><em>(45%)</em></td>
      <td align="center"><strong>92.2</strong> <br><em>(60%)</em></td>
      <td align="center"><strong>87.7</strong> <br><em>(35%)</em></td>
    </tr>
    <tr>
      <td rowspan="2"><strong>Qwen3-4B</strong></td>
      <td>SFT (100% data)</td>
      <td align="center">68.3</td>
      <td align="center">47.3</td>
      <td align="center"><strong>90.1</strong></td>
    </tr>
    <tr>
      <td><strong>Ours</strong> (Reduced data)</td>
      <td align="center"><strong>71.0</strong> <br><em>(45%)</em></td>
      <td align="center"><strong>92.6</strong> <br><em>(60%)</em></td>
      <td align="center">88.4 <br><em>(35%)</em></td>
    </tr>
  </tbody>
</table>

<em>Note: Percentages in parentheses indicate the fraction of data used by our method.</em>
## Overview

```
nesys/
  dataset/              # Transition MCQ benchmark (3 envs × 2 splits)
  eval_results/         # Pre-computed neural evaluation logs (logprobs + predictions)
  final_rules/          # Final symbolic rule files used in the paper
  create_transition_mcq_rules.py   # Rule evaluation / creation tool
  eval_transition_mcq_logprob.py   # Neural MCQ evaluator 
  replicate_our_main_results.sh    # Reproduce NeSyS re-ranking results
  generate_eval_summaries.sh  # Regenerate eval logs
  example_evaluation_script.sh     # Quick-start example
```

## Hugging Face Resources

We release datasets and model adapters on the Hugging Face Hub.

**Dataset**: [`cindermond/nesys-world-model-benchmark`](https://huggingface.co/datasets/cindermond/nesys-world-model-benchmark)
Three tasks (`plancraft`, `scienceworld`, `webshop`) with `dev` and `test` splits, stored as `data/<task>/<split>.jsonl`.

**Model adapters** (LoRA, PEFT):

| Environment | Llama-3.2-1B-Instruct | Qwen3-4B |
|---|---|---|
| PlanCraft | [`cindermond/world-model-plancraft-llama3-2-1b-instruct-filtered`](https://huggingface.co/cindermond/world-model-plancraft-llama3-2-1b-instruct-filtered) | [`cindermond/world-model-plancraft-qwen3-4b-filtered`](https://huggingface.co/cindermond/world-model-plancraft-qwen3-4b-filtered) |
| ScienceWorld | [`cindermond/world-model-scienceworld-llama3-2-1b-instruct-filtered`](https://huggingface.co/cindermond/world-model-scienceworld-llama3-2-1b-instruct-filtered) | [`cindermond/world-model-scienceworld-qwen3-4b-filtered`](https://huggingface.co/cindermond/world-model-scienceworld-qwen3-4b-filtered) |
| WebShop | [`cindermond/world-model-webshop-llama3-2-1b-instruct-filtered`](https://huggingface.co/cindermond/world-model-webshop-llama3-2-1b-instruct-filtered) | [`cindermond/world-model-webshop-qwen3-4b-filtered`](https://huggingface.co/cindermond/world-model-webshop-qwen3-4b-filtered) |

Each adapter was trained on the *filtered* subset of transitions (those not covered by the learned symbolic rules).

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start: Reproduce NeSyS Results

The fastest way to verify our main results is to run the symbolic re-ranking on the **pre-computed** neural evaluation logs (no GPU needed):

```bash
cd nesys
bash replicate_our_main_results.sh
```

This loads the neural logprobs from `eval_results/`, applies the final symbolic rules from `final_rules/`, learns per-rule weights on the dev set, and evaluates on the test set -- printing the NeSyS accuracy for each environment and model.

## Run Neural Evaluation

To evaluate a base model or fine-tuned adapter on the benchmark from Hugging Face:

```bash
cd nesys

# Base model only
python eval_transition_mcq_logprob.py \
  --base_model meta-llama/Llama-3.2-1B-Instruct \
  --dataset_repo_id cindermond/nesys-world-model-benchmark \
  --task scienceworld \
  --split test \
  --output_prefix example_test_scienceworld

# With a fine-tuned adapter from HF
python eval_transition_mcq_logprob.py \
  --base_model meta-llama/Llama-3.2-1B-Instruct \
  --adapter cindermond/world-model-scienceworld-llama3-2-1b-instruct-filtered \
  --dataset_repo_id cindermond/nesys-world-model-benchmark \
  --task scienceworld \
  --split test \
  --output_prefix eval_results/scienceworld_sft_test_llama3-2-1b-instruct_filtered
```

You can also pass local JSONL files via `--dataset_paths` instead of `--dataset_repo_id/--task/--split`.

To regenerate all the evaluation summaries used in the paper:

```bash
cd nesys
bash generate_eval_summaries.sh
```

## Citation
If you find our work useful in your research or applications, please consider citing the following paper and starring this repository.

```bibtex
@article{zhao2026nesys,
  title   = {Neuro-Symbolic Synergy for Interactive World Modeling},
  author  = {Zhao, Hongyu and Zhou, Siyu and Yang, Haolin and Qin, Zengyi and Zhou, Tianyi},
  journal = {arXiv preprint arXiv:2602.10480},
  year    = {2026}
}
```

## License

The code in this repository is released for research purposes. The WebShop dataset files contain web page text from the original WebShop environment; please comply with the original dataset terms when redistributing derivatives.
