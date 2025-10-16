# TokDrift


## Environment Setup
#### Prerequisites
- [uv](https://github.com/astral-sh/uv) package manager (for testing)
- Git LFS (for downloading large datasets from Hugging Face)


#### Setup Environment
Run the provided setup script:

```bash
bash prepare-env.sh
```

For testing generated code, set up a virtual environment with required dependencies:

```bash
mkdir venv && cd venv
uv venv ./python3_8 --python 3.8
uv pip install --python ./python3_8/bin/python numpy scipy networkx
cd ..
```

This environment is used to execute and validate generated code during experiments.

## Dataset Preparation

**Download the Avatar dataset from Hugging Face:**

```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=0 git clone https://huggingface.co/datasets/iidai/avatar
# Normalize the dataset
python scripts/split_avatar.py
```

**Generate rewrite dataset for Avatar tasks only:**

```bash
python -m src.tokdrift.data_generator --process_avatar
# Move the dataset config file to the datasets folder
mv ./data/input/avatar/var.py ./datasets/avatar/var/var.py
```

(Optional) Generate rewrite dataset for all tasks (already prepared for humaneval and codenet tasks):

```bash
python -m src.tokdrift.data_generator --all
```

## Example Scripts

Two example scripts for running baseline and variant experiments are provided in the [`scripts`](scripts) directory:

- [`baseline_example.sh`](scripts/baseline_example.sh) - Example for running baseline experiments
- [`variant_example.sh`](scripts/variant_example.sh) - Example for running variant experiments

For detailed usage instructions, see the [Running Experiments](#running-experiments) and [Task Variants](#task-variants) sections below.

## Result Analysis
After running experiments, analyze the results using the following commands:

#### Extract All Result Datapoints

First, extract all result datapoints from the log files in the output directory:

```bash
python -m src.tokdrift.result_extractor --all
```

This processes all tasks, models, naming variants, and spacing variants to generate evaluation JSON files with detailed result datapoints.

#### Summarize Results to CSV

Generate CSV summary files for all results:

```bash
python -m src.tokdrift.result_evaluator --sum_to_csv
```

This creates comprehensive CSV files in `./data/output/` containing:
- Accuracy results
- Accuracy deltas comparing baseline vs variant
- Sensitivity analysis across all variants
- Per-task and per-model breakdowns

#### Additional Analysis Options

**Get Summary and Sensitivity Results:**

```bash
python -m src.tokdrift.result_evaluator --diff
```

This outputs:
- Total number of processed tasks across all experiments
- Sensitivity results showing how naming and spacing variants affect task results
- Including breakdown by fragment change types (merged, split, mixed, unchanged)

Output files are saved to:
- `./data/output/sensitivity/` - Sensitivity percentages
- `./data/output/sample_info/` - Sample counts and statistics

**Wilcoxon Signed-Rank Test:**

Test the statistical significance of performance differences between various model sizes within one model series:

```bash
python -m src.tokdrift.result_evaluator --wilcoxon_test
```

This compares small, medium, and large model variants (e.g., Llama-3 3B vs 8B vs 70B) to determine if larger models show significantly different sensitivity to token boundary changes.


## Running Experiments
#### Environment Variables

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

*Note that tokenizer's behavior varies across different model series. Special tokenizers may raise errors when analyzing the results (fragment analysis).* 

#### HumanEval Explain Tasks (Two-Stage)

Following the setup in [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness), this task requires two stages: describe then synthesize.

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

#### Other Tasks (Single-Stage)

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
#### HumanEval Fix Task

For HumanEval Fix tasks with variants:

```bash
TASK_NAME="humanevalfixtests"
COMBINED_TOKEN_VARIANT="snake_case"  # or any variant from list below

accelerate launch --num_processes 1 -m src.tokdrift.run_experiments \
  --model $MODEL \
  --tasks $TASK_NAME-$LANGUAGE-$COMBINED_TOKEN_VARIANT-fix \
  [other parameters...]
```

#### Other Tasks with Variants

For CodeNet Translate, Avatar Translate, and other tasks:

```bash
TASK_1_NAME="codenettranslate"  # or avatartranslate
SPECIFIC_CASE="snake_case"  # or any variant from list below

accelerate launch --num_processes 1 -m src.tokdrift.run_experiments \
  --model $MODEL \
  --tasks $TASK_1_NAME-$LANGUAGE-$SPECIFIC_CASE \
  [other parameters...]
```

#### Available Variants

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

