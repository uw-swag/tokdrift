#!/bin/bash

# Change to the correct directory
cd ..

# Set variables
TASK_NAME="codenettranslate"
LANGUAGE="python"
VARIANT="lparentheses_rparentheses"
MODEL="deepseek-ai/deepseek-coder-33b-instruct"
PROMPT="deepseek"
PRECISION="bf16"
MAX_LENGTH_GENERATION=1024
DO_SAMPLE=False
N_SAMPLES=1
BATCH_SIZE=1

# Hardcode for selecting the prompt
if [[ "$MODEL" == "Qwen/CodeQwen1.5-7B-Chat" || "$MODEL" == "Qwen/Qwen2.5-Coder-7B-Instruct" || "$MODEL" == "Qwen/Qwen2.5-Coder-1.5B-Instruct" || "$MODEL" == "Qwen/Qwen2.5-Coder-32B-Instruct" || "$MODEL" == "Qwen/Qwen2.5-Coder-7B" ]]; then
    PROMPT="codeqwen"
elif [[ "$MODEL" == "deepseek-ai/deepseek-coder-1.3b-instruct" || "$MODEL" == "deepseek-ai/deepseek-coder-6.7b-instruct" || "$MODEL" == "deepseek-ai/deepseek-coder-33b-instruct" ]]; then
    PROMPT="deepseek"
else
    PROMPT="instruct"
fi

# Split the model name into model name and model version
MODEL_NAME=${MODEL##*/}

OUTPUT_DIR="./data/output/$TASK_NAME-$LANGUAGE/$MODEL_NAME/$VARIANT"
SAVE_GENERATIONS_PATH="$OUTPUT_DIR/generations.json"
METRIC_OUTPUT_PATH="$OUTPUT_DIR/evaluation_results.json"

HIDDEN_STATES_SAVE_PATH="./data/output/hidden-states-data/$TASK_NAME-$LANGUAGE/$MODEL_NAME/baseline/generations_$TASK_NAME-$LANGUAGE"
HIDDEN_STATES_DIFFERENCE_SAVE_PATH="./data/output/hidden-states-data/analysis_hidden_states/$MODEL_NAME/$VARIANT"

# Create output directory
mkdir -p $OUTPUT_DIR $HIDDEN_STATES_SAVE_PATH $HIDDEN_STATES_DIFFERENCE_SAVE_PATH

# Disable online mode
export HF_DATASETS_OFFLINE=1

echo "Starting experiment for array task $SLURM_ARRAY_TASK_ID: $TASK_NAME-$LANGUAGE-$COMBINED_TOKEN_VARIANT with $MODEL_NAME"
accelerate launch --num_processes 1 -m src.tokdrift.run_experiments \
  --model $MODEL \
  --max_length_generation $MAX_LENGTH_GENERATION \
  --prompt $PROMPT \
  --tasks $TASK_NAME-$LANGUAGE-$VARIANT \
  --precision $PRECISION \
  --do_sample $DO_SAMPLE \
  --n_samples $N_SAMPLES \
  --batch_size $BATCH_SIZE \
  --allow_code_execution \
  --trust_remote_code \
  --save_generations \
  --save_generations_path $SAVE_GENERATIONS_PATH \
  --metric_output_path $METRIC_OUTPUT_PATH \
  --hidden_states_save_path $HIDDEN_STATES_SAVE_PATH \
  --hidden_states_difference_save_path $HIDDEN_STATES_DIFFERENCE_SAVE_PATH \
  --max_memory_per_gpu "auto"

echo "Experiment completed at $(date) for array task $SLURM_ARRAY_TASK_ID: $COMBINED_TOKEN_VARIANT"