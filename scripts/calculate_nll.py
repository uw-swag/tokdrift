import json
import os
import argparse
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from vllm import LLM, SamplingParams

from tokdrift.tasks.humanevalpack import create_task
from tokdrift.config import Config


def calculate_nll_from_logprobs(
    prompt_logprobs: List[Dict],
    start_token_idx: int,
    actual_tokens: List[int]
) -> float:
    """
    Calculate negative log likelihood from prompt logprobs.

    Args:
        prompt_logprobs: List of logprob dicts from VLLM (one per token)
        start_token_idx: Token index where generation starts
        actual_tokens: List of actual token IDs in the sequence

    Returns:
        Average NLL for the generation tokens
    """
    nll_sum = 0.0
    count = 0

    for i in range(start_token_idx, len(prompt_logprobs)):
        if prompt_logprobs[i] is not None and i < len(actual_tokens):
            actual_token_id = actual_tokens[i]

            # Find the logprob for the actual token
            if actual_token_id in prompt_logprobs[i]:
                logprob_obj = prompt_logprobs[i][actual_token_id]
                nll_sum += -logprob_obj.logprob
                count += 1
            elif len(prompt_logprobs[i]) > 0:
                logprob_obj = list(prompt_logprobs[i].values())[0]
                nll_sum += -logprob_obj.logprob
                count += 1

    return nll_sum / count if count > 0 else 0.0


def process_task(
    task_name: str,
    language: str,
    variant: str,
    llm: LLM,
    tokenizer,
    args
) -> None:
    """Process a single task (either Python or Java)"""

    if variant != "baseline":
        input_language = f"{language}-{variant}-fix"
    else:
        input_language = language
    full_task_name = f"humanevalfixtests-{input_language}"
    print(f"\n{'='*60}")
    print(f"Processing task: {full_task_name}")
    print(f"{'='*60}")

    # Create task
    task_class = create_task(input_language, task_name)
    task = task_class(language=input_language, prompt=args.prompt)

    # Build output directory path
    output_dir = os.path.join(
        args.output_base_dir,
        args.folder,
        f"humanevalfixtests-{language}",
        args.model_name,
        variant
    )

    # Load generations
    generation_filename = f"generations_{full_task_name}.json"
    generation_path = os.path.join(output_dir, generation_filename)

    if not os.path.exists(generation_path):
        print(f"Warning: Generation file not found: {generation_path}")
        print(f"Skipping {full_task_name}...")
        return

    print(f"Loading generations from: {generation_path}")
    with open(generation_path, 'r') as f:
        generations = json.load(f)

    # Get dataset
    dataset = task.get_dataset()

    if len(dataset) != len(generations):
        print(f"Warning: Dataset size ({len(dataset)}) != Generations size ({len(generations)})")

    print(f"Dataset: {len(dataset)} problems")
    print(f"Samples per problem: {len(generations[0]) if generations else 0}")

    # Prepare all sequences for batch processing
    print("Preparing sequences for batch processing...")
    all_sequences = []
    all_prompt_lengths = []
    all_full_tokens = []
    metadata = []

    for idx, (doc, gens) in enumerate(tqdm(zip(dataset, generations), total=len(generations), desc="Preparing")):
        full_prompt = task.get_prompt(doc)
        prompt_base = task.get_prompt_base(doc).rstrip()

        for gen_idx, gen in enumerate(gens):
            if gen.startswith(prompt_base):
                actual_generation = gen[len(prompt_base):]
            else:
                actual_generation = gen

            full_sequence = full_prompt + actual_generation

            prompt_tokens = tokenizer.encode(full_prompt)
            full_tokens = tokenizer.encode(full_sequence)

            all_sequences.append(full_sequence)
            all_prompt_lengths.append(len(prompt_tokens))
            all_full_tokens.append(full_tokens)
            metadata.append({
                'problem_idx': idx,
                'sample_idx': gen_idx
            })

    print(f"Total sequences to process: {len(all_sequences)}")

    print("Computing logprobs with VLLM (batched)...")
    sampling_params = SamplingParams(
        max_tokens=1,
        prompt_logprobs=True,
        temperature=1.0,
        top_p=1.0,
    )

    outputs = llm.generate(all_sequences, sampling_params)

    # Process results
    print("Processing results...")
    results = [{"sample_nlls": [], "avg_nll": 0.0} for _ in range(len(dataset))]

    for output_idx, (output, meta, prompt_len, full_tokens) in enumerate(tqdm(
        zip(outputs, metadata, all_prompt_lengths, all_full_tokens),
        total=len(outputs),
        desc="Computing NLL"
    )):
        problem_idx = meta['problem_idx']
        sample_idx = meta['sample_idx']

        # Extract prompt logprobs
        prompt_logprobs = output.prompt_logprobs

        if prompt_logprobs is None or len(prompt_logprobs) == 0:
            print(f"Warning: No logprobs for problem {problem_idx}, sample {sample_idx}")
            nll = 0.0
        else:
            # Calculate NLL for generation tokens only (skip prompt tokens)
            nll = calculate_nll_from_logprobs(
                prompt_logprobs,
                prompt_len,
                full_tokens
            )

        results[problem_idx]["sample_nlls"].append(nll)

    # Calculate averages
    for result in results:
        if result["sample_nlls"]:
            result["avg_nll"] = sum(result["sample_nlls"]) / len(result["sample_nlls"])

    # Save results
    output_path = os.path.join(output_dir, "nll_log.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print summary statistics
    all_nlls = [nll for result in results for nll in result["sample_nlls"]]
    if all_nlls:
        print(f"Overall average NLL: {sum(all_nlls) / len(all_nlls):.4f}")
        print(f"Min NLL: {min(all_nlls):.4f}")
        print(f"Max NLL: {max(all_nlls):.4f}")


def main(args):
    print(f"{'='*60}")
    print(f"NLL Calculation")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Model name: {args.model_name}")
    print(f"Folder: {args.folder}")
    print(f"Output base directory: {args.output_base_dir}")
    print(f"{'='*60}")

    # Initialize VLLM once for all tasks
    print(f"\nInitializing VLLM with model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.num_gpus,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.9,
    )

    # Get tokenizer
    tokenizer = llm.get_tokenizer()
    print("VLLM initialized successfully!")

    # Process both tasks
    tasks = [
        ("fixtests", "python"),
        ("fixtests", "java")
    ]

    config = Config()
    all_variants, python_variants, java_variants = config.get_all_generation_variants()

    for task_name, language in tasks:
        if language == "python":
            variants = python_variants
        elif language == "java":
            variants = java_variants
        else:
            raise ValueError(f"Invalid language: {language}")

        variants = ["baseline"] + variants

        for variant in variants:
            try:
                process_task(task_name, language, variant, llm, tokenizer, args)
            except Exception as e:
                print(f"\nError processing {task_name}-{language}: {e}")
                import traceback
                traceback.print_exc()
                print("Continuing with next task...")
                continue

    print(f"\n{'='*60}")
    print(f"All tasks completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate NLL for model generations")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path (e.g., meta-llama/Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name for directory structure (e.g., Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Folder name (e.g., k, prev_py)"
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        help="Base output directory"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="instruct",
        help="Prompt type used during generation"
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=4096,
        help="Maximum model context length"
    )

    args = parser.parse_args()
    main(args)
