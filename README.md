# TokDrift


## Environment Setup

### Prerequisites
- [uv](https://github.com/astral-sh/uv) package manager (for testing)
- Git LFS
- Python 3.8+


### Setup Environment
Run the provided setup script:

```bash
bash prepare-env.sh
```

### Setup Testing Environment

For testing generated code, set up a virtual environment with required dependencies:

```bash
mkdir venv && cd venv
uv venv ./python3_8 --python 3.8
uv pip install --python ./python3_8/bin/python numpy scipy networkx
cd ..
```

This environment is used to execute and validate generated code during experiments.

## Dataset Preparation

### Avatar Dataset

```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=0 git clone https://huggingface.co/datasets/iidai/avatar
python -m src.tokdrift.split_avatar
python -m src.tokdrift.data_generator --process_avatar
```

## Running Experiments

### Environment Variables

Set these variables before running experiments:

```bash
MODEL="your-model-name"
MAX_LENGTH_GENERATION=1024
PROMPT="prompt-type"
LANGUAGE="python"
PRECISION="bf16"
DO_SAMPLE=False
N_SAMPLES=1
BATCH_SIZE=1
SAVE_GENERATIONS_PATH="path/to/save/generations"
METRIC_OUTPUT_PATH="path/to/save/metrics"
HIDDEN_STATES_SAVE_PATH="path/to/save/hidden_states"
```

### HumanEval Explain Tasks (Two-Stage)

This task requires two stages: describe then synthesize.

**Stage 1: Describe**

```bash
TASK_1_NAME="humanevalexplaindescribe"

accelerate launch --num_processes 1 -m src.tokdrift.run_experiments \
  --model $MODEL \
  --max_length_generation $MAX_LENGTH_GENERATION \
  --prompt $PROMPT \
  --tasks $TASK_1_NAME-$LANGUAGE \
  --precision $PRECISION \
  --do_sample $DO_SAMPLE \
  --n_samples $N_SAMPLES \
  --batch_size $BATCH_SIZE \
  --allow_code_execution \
  --trust_remote_code \
  --save_generations \
  --save_generations_path $SAVE_GENERATIONS_PATH \
  --generation_only \
  --hidden_states_save_path $HIDDEN_STATES_SAVE_PATH \
  --max_memory_per_gpu "auto"
```

**Stage 2: Synthesize**

```bash
TASK_2_NAME="humanevalexplainsynthesize"
LOAD_GENERATIONS_PATH="path/from/stage1"

accelerate launch --num_processes 1 -m src.tokdrift.run_experiments \
  --model $MODEL \
  --max_length_generation $MAX_LENGTH_GENERATION \
  --prompt $PROMPT \
  --load_data_path $LOAD_GENERATIONS_PATH \
  --tasks $TASK_2_NAME-$LANGUAGE \
  --precision $PRECISION \
  --do_sample $DO_SAMPLE \
  --n_samples $N_SAMPLES \
  --batch_size $BATCH_SIZE \
  --allow_code_execution \
  --trust_remote_code \
  --save_generations \
  --save_generations_path $SAVE_GENERATIONS_PATH \
  --metric_output_path $METRIC_OUTPUT_PATH \
  --max_memory_per_gpu "auto"
```

### Other Tasks (Single-Stage)

For tasks like CodeNet Translate, Avatar Translate, and HumanEval Fix Tests:

```bash
# Choose one:
TASK_1_NAME="codenettranslate"
# TASK_1_NAME="avatartranslate"
# TASK_NAME="humanevalfixtests"

accelerate launch --num_processes 1 -m src.tokdrift.run_experiments \
  --model $MODEL \
  --max_length_generation $MAX_LENGTH_GENERATION \
  --prompt $PROMPT \
  --tasks $TASK_1_NAME-$LANGUAGE \
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
  --max_memory_per_gpu "auto"
```

## Task Variants

### HumanEval Fix Task

For HumanEval Fix tasks with variants:

```bash
TASK_NAME="humanevalfixtests"
COMBINED_TOKEN_VARIANT="snake_case"  # or any variant from list below

accelerate launch --num_processes 1 -m src.tokdrift.run_experiments \
  --model $MODEL \
  --tasks $TASK_NAME-$LANGUAGE-$COMBINED_TOKEN_VARIANT-fix \
  [other parameters...]
```

### Other Tasks with Variants

For CodeNet Translate, Avatar Translate, and other tasks:

```bash
TASK_1_NAME="codenettranslate"  # or avatartranslate
SPECIFIC_CASE="snake_case"  # or any variant from list below

accelerate launch --num_processes 1 -m src.tokdrift.run_experiments \
  --model $MODEL \
  --tasks $TASK_1_NAME-$LANGUAGE-$SPECIFIC_CASE \
  [other parameters...]
```

### Available Variants

- `snake_case`
- `pascal_case`
- `screaming_snake_case`
- `camel_case`
- `op_dash`
- `op_lsquarebracket`
- `rparentheses_period`
- `rsquarebracket_rparentheses`
- `op_rsquarebracket`
- `op_lparentheses`
- `lsquarebracket_name`
- `double_plus_rparentheses`
- `period_asterisk`
- `rparentheses_colon`
- `rparentheses_semicolon`
- `op_semicolon`
- `rparentheses_rparentheses`
- `lparentheses_rparentheses`
- `period_name`
- `lparentheses_name`
- `op_name`
- `op_all`

