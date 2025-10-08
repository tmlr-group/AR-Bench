#!/bin/bash

# Set default values
DATA_PATH="data/gn/test.json"
MODEL="qwen2.5-32b-instruct"
# zero_shot, few_shot, few_shot_inst, tot, proactive_cot, uot
METHOD="uot"
MAX_TURN=25
TEMPERATURE=0.7
TOP_P=0.7
SIMULATE_DEPTH=1
OUTPUT_PATH="logs/gn/log_${METHOD}_${MODEL}.json"
# Create output directory if not exists
mkdir -p "$(dirname "$OUTPUT_PATH")"

python3 -m arbench.reasoner.gn.gn_evaluator \
    --model="$MODEL" \
    --method="$METHOD" \
    --data_path="$DATA_PATH" \
    --output_path="$OUTPUT_PATH" \
    --max_turn="$MAX_TURN" \
    --temperature="$TEMPERATURE" \
    --top_p="$TOP_P" \
    --simulate_depth="$SIMULATE_DEPTH" \
