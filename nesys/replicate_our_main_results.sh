python create_transition_mcq_rules.py \
    --eval_summary eval_results/plancraft_sft_test_llama3-2-1b-instruct_filtered_summary.json \
    --dev_eval_summary eval_results/plancraft_sft_dev_llama3-2-1b-instruct_filtered_summary.json \
    --evaluate_rules_file final_rules/rules_plancraft_llama3-2-1b-instruct_final.py \
    > nesys_plancraft_llama3-2-1b-instruct.log

python create_transition_mcq_rules.py \
    --eval_summary eval_results/plancraft_sft_test_qwen3-4b_filtered_summary.json \
    --dev_eval_summary eval_results/plancraft_sft_dev_qwen3-4b_filtered_summary.json \
    --evaluate_rules_file final_rules/rules_plancraft_qwen3-4b_final.py \
    > nesys_plancraft_qwen3-4b.log

python create_transition_mcq_rules.py \
    --eval_summary eval_results/scienceworld_sft_test_llama3-2-1b-instruct_filtered_summary.json \
    --dev_eval_summary eval_results/scienceworld_sft_dev_llama3-2-1b-instruct_filtered_summary.json \
    --evaluate_rules_file final_rules/rules_scienceworld_llama3-2-1b-instruct_final.py \
    > nesys_scienceworld_llama3-2-1b-instruct.log

python create_transition_mcq_rules.py \
    --eval_summary eval_results/scienceworld_sft_test_qwen3-4b_filtered_summary.json \
    --dev_eval_summary eval_results/scienceworld_sft_dev_qwen3-4b_filtered_summary.json \
    --evaluate_rules_file final_rules/rules_scienceworld_qwen3-4b_final.py \
    > nesys_scienceworld_qwen3-4b.log

python create_transition_mcq_rules.py \
    --eval_summary eval_results/webshop_transition_qa_sft_test_llama3-2-1b-instruct_filtered_summary.json \
    --dev_eval_summary eval_results/webshop_transition_qa_sft_dev_llama3-2-1b-instruct_filtered_summary.json \
    --evaluate_rules_file final_rules/rules_webshop_transition_qa_llama3-2-1b-instruct_final.py \
    > nesys_webshop_llama3-2-1b-instruct.log

python create_transition_mcq_rules.py \
    --eval_summary eval_results/webshop_transition_qa_sft_test_qwen3-4b_filtered_summary.json \
    --dev_eval_summary eval_results/webshop_transition_qa_sft_dev_qwen3-4b_filtered_summary.json \
    --evaluate_rules_file final_rules/rules_webshop_transition_qa_qwen3-4b_final.py \
    > nesys_webshop_qwen3-4b.log
