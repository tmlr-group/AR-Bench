#!/bin/bash

# zero_shot, few_shot, few_shot_inst, tot, proactive_cot, uot
METHOD="uot"
MAX_TURN=25
BRANCH=3
POLICY_MODEL="qwen2.5-32b-instruct"
RESPONSE_MODEL="qwen2.5-32b-instruct"
POLICY_TEMPERATURE=0.7
POLICY_TOP_P=0.7
RESPONSE_TEMPERATURE=0.7
RESPONSE_TOP_P=0.7
SIMULATE_DEPTH=1
DATA_PATH="data/sp/test.json"

OUTPUT_PATH="logs/sp/log_${METHOD}_${POLICY_MODEL}.json"
# Create output directory if not exists
mkdir -p "$(dirname "$OUTPUT_PATH")"

python3 -m arbench.reasoner.sp.sp_evaluator \
    --method="$METHOD" \
    --data_path="$DATA_PATH" \
    --output_path="$OUTPUT_PATH" \
    --policy_model="$POLICY_MODEL" \
    --response_model="$RESPONSE_MODEL" \
    --branch="$BRANCH" \
    --max_turn="$MAX_TURN" \
    --policy_temperature="$POLICY_TEMPERATURE" \
    --policy_top_p="$POLICY_TOP_P" \
    --response_temperature="$RESPONSE_TEMPERATURE" \
    --response_top_p="$RESPONSE_TOP_P" \
    --simulate_depth="$SIMULATE_DEPTH"
