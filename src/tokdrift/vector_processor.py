
from transformers import AutoTokenizer
import torch
import os
import torch.nn.functional as F
import json

from .config import Config
from .data_generator import DataExtractor, DataGenerator
from . import tasks


class VectorProcessor:
    def __init__(self, config: Config, tokenizer=None, hidden_states_save_path=None, hidden_states_difference_save_path=None, variant_hidden_states_save_path=None):
        self.config = config
        self.tokenizer = tokenizer
        self.hidden_states_save_path = hidden_states_save_path
        self.hidden_states_difference_save_path = hidden_states_difference_save_path
        self.variant_hidden_states_save_path = variant_hidden_states_save_path

    def extract_hidden_states(self):
        return None
        
    def process_multi_token_identifiers(self, all_multi_token_identifiers, hidden_states=None):
        prompt = self.config.prompt
        model = self.config.model
        
        class Args:
            def __init__(self):
                self.prompt = prompt
                self.model = model
                self.load_data_path = "none"

        args = Args()
        baseline_task = tasks.get_task(self.config.task, args)
        baseline_dataset = baseline_task.get_dataset()
        
        variant_task_name = f"{self.config.task}-{self.config.target_type}" if self.config.output_dataset_name != "humanevalfixtests" else f"{self.config.task}-{self.config.target_type}-fix"
        variant_task = tasks.get_task(variant_task_name, args)
        variant_dataset = variant_task.get_dataset()

        baseline_ids = [baseline_task.get_task_id(baseline_dataset[i]) for i in range(len(baseline_dataset))]
        variant_ids = [variant_task.get_task_id(variant_dataset[i]) for i in range(len(variant_dataset))]

        task_idx = [baseline_ids.index(variant_id) for variant_id in variant_ids]

        if len(task_idx) != len(all_multi_token_identifiers):
            raise ValueError(f"Length of task_idx and all_multi_token_identifiers must be the same, but got {len(task_idx)} and {len(all_multi_token_identifiers)}")
        
        # Baseline hidden states paths
        baseline_hidden_states_path = self.hidden_states_save_path
        if hidden_states is not None:
            variant_hidden_states_path = None
        else:
            variant_hidden_states_path = self.variant_hidden_states_save_path
        # Get the number of multi-token identifiers
        number_of_multi_token_identifiers = 0
        for task_id in all_multi_token_identifiers:
            number_of_multi_token_identifiers += len(all_multi_token_identifiers[task_id])
        
        output_all_layers = {}
        output_all_layers_summary = {"total_identifiers": number_of_multi_token_identifiers}
        all_similarity_list = []
        
        # Initialize storage for vector differences by layer
        layer_vector_differences = {}
        
        # Process all multi-token identifiers
        for i in range(len(task_idx)):

            if i % 50 == 0:
                print(f"Processing task {i} of {len(task_idx)}")

            baseline_sample = baseline_dataset[task_idx[i]]
            variant_sample = variant_dataset[i]

            baseline_hidden_states = torch.load(os.path.join(baseline_hidden_states_path, f"sample_{task_idx[i]}.pt"))
            if variant_hidden_states_path is not None:
                variant_hidden_states = torch.load(os.path.join(variant_hidden_states_path, f"sample_{i}.pt"))
            else:
                variant_hidden_states = hidden_states[i]


            task_id = baseline_task.get_task_id(baseline_sample)
            variant_task_id = variant_task.get_task_id(variant_sample)

            if task_id != variant_task_id:
                raise ValueError(f"Task id and variant task id must be the same, but got {task_id} and {variant_task_id}")

            baseline_prompt = baseline_task.get_prompt(baseline_sample)
            variant_prompt = variant_task.get_prompt(variant_sample)

            baseline_prompt_LLM_tokens = self.tokenizer.tokenize(baseline_prompt)
            variant_prompt_LLM_tokens = self.tokenizer.tokenize(variant_prompt)

            baseline_context = baseline_task.get_context_only(baseline_sample)
            variant_context = variant_task.get_context_only(variant_sample)

            baseline_prompt_before_context = baseline_prompt.split(baseline_context)[0]
            variant_prompt_before_context = variant_prompt.split(variant_context)[0]

            len_baseline_prompt_before_context_LLM_tokens = len(self.tokenizer.tokenize(baseline_prompt_before_context))
            len_variant_prompt_before_context_LLM_tokens = len(self.tokenizer.tokenize(variant_prompt_before_context))

            if len(baseline_prompt_LLM_tokens) == len_baseline_prompt_before_context_LLM_tokens:
                baseline_prompt_before_context = baseline_prompt.split(baseline_context[:30])[0]
                len_baseline_prompt_before_context_LLM_tokens = len(self.tokenizer.tokenize(baseline_prompt_before_context))
            if len(variant_prompt_LLM_tokens) == len_variant_prompt_before_context_LLM_tokens:
                variant_prompt_before_context = variant_prompt.split(variant_context[:30])[0]
                len_variant_prompt_before_context_LLM_tokens = len(self.tokenizer.tokenize(variant_prompt_before_context))

            if len(baseline_prompt_LLM_tokens) > 2048 or len(variant_prompt_LLM_tokens) > 2048:
                print("Skip task: ", task_id)
                continue

            # Process all layers
            output_layer = {}
            similarity_list = []
            for layer in range(baseline_hidden_states.shape[0]):
                
                output_similarity = {}
                layer_similarity_list = []

                for token_idx in range(len(all_multi_token_identifiers[task_id])):
                    
                    # Get the LLM tokens indices
                    try:
                        baseline_LLM_tokens_idx = all_multi_token_identifiers[task_id][token_idx]['LLM_tokens_idx']
                        variant_LLM_tokens_idx = all_multi_token_identifiers[task_id][token_idx]['new_LLM_tokens_idx']
                    except:
                        print(f"Error: {all_multi_token_identifiers[task_id][token_idx]}")
                        raise ValueError("Stop here")

                    baseline_LLM_tokens_idx_start = int(baseline_LLM_tokens_idx.split(",")[0].strip('(')) + len_baseline_prompt_before_context_LLM_tokens
                    baseline_LLM_tokens_idx_end = int(baseline_LLM_tokens_idx.split(",")[1].strip(')')) + len_baseline_prompt_before_context_LLM_tokens
                    variant_LLM_tokens_idx_start = int(variant_LLM_tokens_idx.split(",")[0].strip('(')) + len_variant_prompt_before_context_LLM_tokens
                    variant_LLM_tokens_idx_end = int(variant_LLM_tokens_idx.split(",")[1].strip(')')) + len_variant_prompt_before_context_LLM_tokens

                    # Get the LLM tokens
                    baseline_LLM_tokens = baseline_prompt_LLM_tokens[baseline_LLM_tokens_idx_start:baseline_LLM_tokens_idx_end]
                    variant_LLM_tokens = variant_prompt_LLM_tokens[variant_LLM_tokens_idx_start:variant_LLM_tokens_idx_end]
                    output_tokens = f"{token_idx}: {baseline_LLM_tokens} -> {variant_LLM_tokens}"

                    # Hardcoded
                    # Adjust the LLM tokens indices for CodeQwen
                    if self.config.prompt == "codeqwen":
                        baseline_LLM_tokens_idx_end -= 1
                        variant_LLM_tokens_idx_end -= 1

                    max_idx = 2048 if self.config.output_dataset_name == "codenettranslate" else 1024
                    if baseline_LLM_tokens_idx_end > max_idx or variant_LLM_tokens_idx_end > max_idx:
                        print(f"Token index out of range: {baseline_LLM_tokens_idx_end} > {max_idx}")
                        continue
                    
                    try:
                        baseline_hidden_state = baseline_hidden_states[layer, baseline_LLM_tokens_idx_end, :]
                        variant_hidden_state = variant_hidden_states[layer, variant_LLM_tokens_idx_end, :]
                    except:
                        # Print the length of the hidden states
                        print(f"length of hidden states: {baseline_hidden_states.shape}")
                        print(f"length of variant hidden states: {variant_hidden_states.shape}")

                        print(f"token: {all_multi_token_identifiers[task_id][token_idx]}")
                        raise ValueError("Stop here")

                    # Calculate vector difference (variant - baseline)
                    vector_difference = variant_hidden_state - baseline_hidden_state
                    
                    # Store vector difference for this layer
                    if layer not in layer_vector_differences:
                        layer_vector_differences[layer] = []
                    layer_vector_differences[layer].append(vector_difference)

                    cosine_sim = F.cosine_similarity(baseline_hidden_state.unsqueeze(0), variant_hidden_state.unsqueeze(0))

                    layer_similarity_list.append(cosine_sim.item())
                    output_similarity[output_tokens] = cosine_sim.item()

                output_layer[layer] = output_similarity
                similarity_list.append(layer_similarity_list)
            
            all_similarity_list.append(similarity_list)
            output_all_layers[task_id] = output_layer
        
        # Calculate the average similarity for each layer on all tasks
        num_layers = len(all_similarity_list[0])  # Get number of layers from first task
        layer_averages = {}
        
        for layer_idx in range(num_layers):
            # Collect all cosine similarities for this layer across all tasks
            layer_cosine_sims = []
            for task_similarities in all_similarity_list:
                layer_cosine_sims.extend(task_similarities[layer_idx])
            
            # Calculate average for this layer
            layer_average = sum(layer_cosine_sims) / len(layer_cosine_sims)
            layer_averages[f"layer_{layer_idx}"] = layer_average
        
        # Add layer averages to summary
        output_all_layers_summary["layer_averages"] = layer_averages

        # Save vector differences as .pt files
        shifts_dir = os.path.join(self.hidden_states_difference_save_path, f"{self.config.task}/shifts")
        os.makedirs(shifts_dir, exist_ok=True)
        
        for layer, differences in layer_vector_differences.items():
            # Stack all vector differences for this layer into a single tensor
            layer_tensor = torch.stack(differences)  # Shape: [num_identifiers, hidden_dim]
            torch.save(layer_tensor, os.path.join(shifts_dir, f"layer_{layer}.pt"))

        # Save the summary and output in a single file
        output_averages = {
            "averages": output_all_layers_summary,
            "all_layers": output_all_layers
        }

        output_path = os.path.join(self.hidden_states_difference_save_path, f"{self.config.task}.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_averages, f, indent=4)

        print(f"Saved vector differences output to {output_path}")

        return output_all_layers, output_all_layers_summary

    def process_combined_token_operators(self, all_combined_token_operators, hidden_states=None):
        prompt = self.config.prompt
        model = self.config.model
        
        class Args:
            def __init__(self):
                self.prompt = prompt
                self.model = model
                self.load_data_path = "none"

        args = Args()
        baseline_task = tasks.get_task(self.config.task, args)
        baseline_dataset = baseline_task.get_dataset()
        
        variant_task_name = f"{self.config.task}-{self.config.combination_name}" if self.config.output_dataset_name != "humanevalfixtests" else f"{self.config.task}-{self.config.combination_name}-fix"
        variant_task = tasks.get_task(variant_task_name, args)
        variant_dataset = variant_task.get_dataset()

        baseline_ids = [baseline_task.get_task_id(baseline_dataset[i]) for i in range(len(baseline_dataset))]
        variant_ids = [variant_task.get_task_id(variant_dataset[i]) for i in range(len(variant_dataset))]

        task_idx = [baseline_ids.index(variant_id) for variant_id in variant_ids]

        if len(task_idx) != len(all_combined_token_operators):
            raise ValueError(f"Length of task_idx and all_combined_token_operators must be the same, but got {len(task_idx)} and {len(all_combined_token_operators)}")
        
        baseline_hidden_states_path = self.hidden_states_save_path
        if hidden_states is not None:
            variant_hidden_states_path = None
        else:
            variant_hidden_states_path = self.variant_hidden_states_save_path

        # Get the number of combined token operators
        number_of_combined_token_operators = 0
        for task_id in all_combined_token_operators:
            number_of_combined_token_operators += len(all_combined_token_operators[task_id])

        
        output_all_layers = {}
        output_all_layers_prev = {}
        output_all_layers_summary = {"total_operators": number_of_combined_token_operators}
        output_all_layers_summary_prev = {"total_operators": number_of_combined_token_operators}
        all_similarity_list = []
        all_similarity_list_prev = []
        
        # Initialize storage for vector differences by layer
        layer_vector_differences = {}
        layer_vector_differences_prev = {}
        
        # Process all combined token operators
        for i in range(len(task_idx)):
            if i % 50 == 0:
                print(f"Processing task {i} of {len(task_idx)}")

            baseline_sample = baseline_dataset[task_idx[i]]
            variant_sample = variant_dataset[i]

            baseline_hidden_states = torch.load(os.path.join(baseline_hidden_states_path, f"sample_{task_idx[i]}.pt"))
            if variant_hidden_states_path is not None:
                variant_hidden_states = torch.load(os.path.join(variant_hidden_states_path, f"sample_{i}.pt"))
            else:
                variant_hidden_states = hidden_states[i]

            task_id = baseline_task.get_task_id(baseline_sample)
            variant_task_id = variant_task.get_task_id(variant_sample)

            if len(all_combined_token_operators[task_id]) == 0:
                continue

            if task_id != variant_task_id:
                raise ValueError(f"Task id and variant task id must be the same, but got {task_id} and {variant_task_id}")

            baseline_prompt = baseline_task.get_prompt(baseline_sample)
            variant_prompt = variant_task.get_prompt(variant_sample)

            baseline_prompt_LLM_tokens = self.tokenizer.tokenize(baseline_prompt)
            variant_prompt_LLM_tokens = self.tokenizer.tokenize(variant_prompt)

            baseline_context = baseline_task.get_context_only(baseline_sample)
            variant_context = variant_task.get_context_only(variant_sample)

            baseline_prompt_before_context = baseline_prompt.split(baseline_context)[0]
            variant_prompt_before_context = variant_prompt.split(variant_context)[0]

            len_baseline_prompt_before_context_LLM_tokens = len(self.tokenizer.tokenize(baseline_prompt_before_context))
            len_variant_prompt_before_context_LLM_tokens = len(self.tokenizer.tokenize(variant_prompt_before_context))

            if len(baseline_prompt_LLM_tokens) == len_baseline_prompt_before_context_LLM_tokens:
                baseline_prompt_before_context = baseline_prompt.split(baseline_context[:30])[0]
                len_baseline_prompt_before_context_LLM_tokens = len(self.tokenizer.tokenize(baseline_prompt_before_context))
            if len(variant_prompt_LLM_tokens) == len_variant_prompt_before_context_LLM_tokens:
                variant_prompt_before_context = variant_prompt.split(variant_context[:30])[0]
                len_variant_prompt_before_context_LLM_tokens = len(self.tokenizer.tokenize(variant_prompt_before_context))
            

            if len(baseline_prompt_LLM_tokens) > 2048 or len(variant_prompt_LLM_tokens) > 2048:
                print("Skip task: ", task_id)
                continue

            # Process all layers
            output_layer = {}
            similarity_list = []

            prev_output_layer = {}
            prev_similarity_list = []
            for layer in range(baseline_hidden_states.shape[0]):

                output_similarity = {}
                layer_similarity_list = []

                prev_output_similarity = {}
                prev_layer_similarity_list = []

                for token_idx in range(len(all_combined_token_operators[task_id])):
                    
                    # Get the LLM tokens indices
                    baseline_LLM_tokens_idx = all_combined_token_operators[task_id][token_idx]['LLM_tokens_idx']
                    variant_LLM_tokens_idx = all_combined_token_operators[task_id][token_idx]['new_LLM_tokens_idx']
                    baseline_prev_LLM_tokens_idx = all_combined_token_operators[task_id][token_idx]['target_LLM_tokens_idx']
                    variant_prev_LLM_tokens_idx = all_combined_token_operators[task_id][token_idx]['new_target_LLM_tokens_idx']

                    baseline_LLM_tokens_idx_start = int(baseline_LLM_tokens_idx.split(",")[0].strip('(')) + len_baseline_prompt_before_context_LLM_tokens
                    baseline_LLM_tokens_idx_end = int(baseline_LLM_tokens_idx.split(",")[1].strip(')')) + len_baseline_prompt_before_context_LLM_tokens
                    variant_LLM_tokens_idx_start = int(variant_LLM_tokens_idx.split(",")[0].strip('(')) + len_variant_prompt_before_context_LLM_tokens
                    variant_LLM_tokens_idx_end = int(variant_LLM_tokens_idx.split(",")[1].strip(')')) + len_variant_prompt_before_context_LLM_tokens
                    baseline_prev_LLM_tokens_idx_start = int(baseline_prev_LLM_tokens_idx.split(",")[0].strip('(')) + len_baseline_prompt_before_context_LLM_tokens
                    baseline_prev_LLM_tokens_idx_end = int(baseline_prev_LLM_tokens_idx.split(",")[1].strip(')')) + len_baseline_prompt_before_context_LLM_tokens
                    variant_prev_LLM_tokens_idx_start = int(variant_prev_LLM_tokens_idx.split(",")[0].strip('(')) + len_variant_prompt_before_context_LLM_tokens
                    variant_prev_LLM_tokens_idx_end = int(variant_prev_LLM_tokens_idx.split(",")[1].strip(')')) + len_variant_prompt_before_context_LLM_tokens

                    # Get the LLM tokens
                    baseline_LLM_tokens = baseline_prompt_LLM_tokens[baseline_prev_LLM_tokens_idx_start:baseline_LLM_tokens_idx_end]
                    variant_LLM_tokens = variant_prompt_LLM_tokens[variant_prev_LLM_tokens_idx_start:variant_LLM_tokens_idx_end]
                    output_tokens = f"{token_idx}: {baseline_LLM_tokens} -> {variant_LLM_tokens}"

                    baseline_prev_LLM_tokens = baseline_prompt_LLM_tokens[baseline_prev_LLM_tokens_idx_start:baseline_LLM_tokens_idx_end]
                    variant_prev_LLM_tokens = variant_prompt_LLM_tokens[variant_prev_LLM_tokens_idx_start:variant_prev_LLM_tokens_idx_end]
                    output_prev_tokens = f"{token_idx}: {baseline_prev_LLM_tokens} -> {variant_prev_LLM_tokens}"

                    # Adjust the LLM tokens indices for CodeQwen
                    if self.config.prompt == "codeqwen":
                        baseline_LLM_tokens_idx_end -= 1
                        variant_LLM_tokens_idx_end -= 1

                    max_idx = 2048 if self.config.output_dataset_name == "codenettranslate" else 1024
                    if baseline_LLM_tokens_idx_end > max_idx:
                        print(f"Token index out of range: {baseline_LLM_tokens_idx_end} > {max_idx}")
                        continue
                        
                    # Get the hidden states
                    try:
                        baseline_hidden_state = baseline_hidden_states[layer, baseline_LLM_tokens_idx_end, :]
                        variant_hidden_state = variant_hidden_states[layer, variant_LLM_tokens_idx_end, :]

                        baseline_prev_hidden_state = baseline_hidden_states[layer, baseline_prev_LLM_tokens_idx_end, :]
                        variant_prev_hidden_state = variant_hidden_states[layer, variant_prev_LLM_tokens_idx_end, :]
                    except:
                        print(f"token: {all_combined_token_operators[task_id][token_idx]}")
                        print(f"length of baseline hidden states: {baseline_hidden_states.shape}")
                        print(f"length of variant hidden states: {variant_hidden_states.shape}")
                        raise ValueError("Stop here")

                    # Calculate vector difference (variant - baseline)
                    vector_difference = variant_hidden_state - baseline_hidden_state
                    vector_difference_prev = variant_prev_hidden_state - baseline_prev_hidden_state
                    
                    # Store vector difference for this layer
                    if layer not in layer_vector_differences:
                        layer_vector_differences[layer] = []
                    if layer not in layer_vector_differences_prev:
                        layer_vector_differences_prev[layer] = []
                    layer_vector_differences[layer].append(vector_difference)
                    layer_vector_differences_prev[layer].append(vector_difference_prev)

                    cosine_sim = F.cosine_similarity(baseline_hidden_state.unsqueeze(0), variant_hidden_state.unsqueeze(0))
                    cosine_sim_prev = F.cosine_similarity(baseline_prev_hidden_state.unsqueeze(0), variant_prev_hidden_state.unsqueeze(0))
    
                    layer_similarity_list.append(cosine_sim.item())
                    output_similarity[output_tokens] = cosine_sim.item()

                    prev_layer_similarity_list.append(cosine_sim_prev.item())
                    prev_output_similarity[output_prev_tokens] = cosine_sim_prev.item()

                output_layer[layer] = output_similarity
                similarity_list.append(layer_similarity_list)

                prev_output_layer[layer] = prev_output_similarity
                prev_similarity_list.append(prev_layer_similarity_list)

            all_similarity_list.append(similarity_list)
            output_all_layers[task_id] = output_layer

            all_similarity_list_prev.append(prev_similarity_list)
            output_all_layers_prev[task_id] = prev_output_layer
        
        # Calculate the average similarity for each layer on all tasks
        if len(all_similarity_list) == 0:
            print(f"No similarity list found for task {self.config.task}")
            return None, None
        
        if len(all_similarity_list_prev) == 0:
            print(f"No similarity list found for task {self.config.task}")
            return None, None
        
        num_layers = len(all_similarity_list[0]) # Get number of layers from first task
        layer_averages = {}
        layer_averages_prev = {}
        
        for layer_idx in range(num_layers):
            # Collect all cosine similarities for this layer across all tasks
            layer_cosine_sims = []
            layer_cosine_sims_prev = []

            for task_similarities in all_similarity_list:
                layer_cosine_sims.extend(task_similarities[layer_idx])
            for task_similarities_prev in all_similarity_list_prev:
                layer_cosine_sims_prev.extend(task_similarities_prev[layer_idx])
            
            # Calculate average for this layer
            layer_average = sum(layer_cosine_sims) / len(layer_cosine_sims)
            layer_averages[f"layer_{layer_idx}"] = layer_average

            layer_average_prev = sum(layer_cosine_sims_prev) / len(layer_cosine_sims_prev)
            layer_averages_prev[f"layer_{layer_idx}"] = layer_average_prev
        
        # Add layer averages to summary
        output_all_layers_summary["layer_averages"] = layer_averages
        output_all_layers_summary_prev["layer_averages"] = layer_averages_prev

        # Save vector differences as .pt files
        shifts_dir = os.path.join(self.hidden_states_difference_save_path, f"{self.config.task}/shifts")
        os.makedirs(shifts_dir, exist_ok=True)
        
        for layer, differences in layer_vector_differences.items():
            # Stack all vector differences for this layer into a single tensor
            layer_tensor = torch.stack(differences)  # Shape: [num_operators, hidden_dim]
            torch.save(layer_tensor, os.path.join(shifts_dir, f"layer_{layer}.pt"))

        # Save the summary and output in a single file
        output_averages = {
            "averages": output_all_layers_summary,
            "all_layers": output_all_layers
        }

        output_path = os.path.join(self.hidden_states_difference_save_path, f"{self.config.task}.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_averages, f, indent=4)
        
        print(f"Saved output to {output_path}")
        

        # Save previous token vector differences as .pt files
        prev_token_shifts_dir = os.path.join(self.hidden_states_difference_save_path, f"{self.config.task}/prev_shifts")
        os.makedirs(prev_token_shifts_dir, exist_ok=True)
        
        for layer, differences in layer_vector_differences_prev.items():
            # Stack all vector differences for this layer into a single tensor
            layer_tensor = torch.stack(differences)  # Shape: [num_operators, hidden_dim]
            torch.save(layer_tensor, os.path.join(prev_token_shifts_dir, f"layer_{layer}.pt"))
        
        output_averages_prev = {
            "averages": output_all_layers_summary_prev,
            "all_layers": output_all_layers_prev
        }

        output_path_prev = os.path.join(self.hidden_states_difference_save_path, f"prev/{self.config.task}.json")
        os.makedirs(os.path.dirname(output_path_prev), exist_ok=True)
        with open(output_path_prev, "w") as f:
            json.dump(output_averages_prev, f, indent=4)
        
        print(f"Saved previous token vector differences output to {output_path_prev}")

        return output_all_layers, output_all_layers_summary
    
    def process_hidden_states(self, hidden_states):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        data_extractor = DataExtractor(self.tokenizer, self.config)
        data_generator = DataGenerator(self.tokenizer, self.config)
        
        if self.config.processing_mode == "multi_token_identifiers":
            print(f"Processing multi-token identifiers with target type: {self.config.target_type}")
            print(f"Filter type: {self.config.filter_type}")
            all_multi_token_identifiers, all_test_identifiers = data_extractor.extract_new_identifiers()

            new_contexts, all_selected_multi_token_identifiers, all_token_boundary_changed, all_modified_tests, all_modified_entry_points, all_modified_declarations = data_generator.process_all_multi_token_identifiers(all_multi_token_identifiers, all_test_identifiers)

            if new_contexts:
                print(f"Processing multi-token vector differences...")
                _, _ = self.process_multi_token_identifiers(all_selected_multi_token_identifiers, hidden_states)
                print(f"Multi-token vector differences processed.")
            else:
                print("No contexts were modified. No result evaluated.")

        elif self.config.processing_mode == "combined_token_operators":
            print(f"Processing combined token operators with target combinations: {self.config.target_combinations}")
            all_combined_token_operators = data_extractor.extract_combined_token_operators()
            new_contexts, all_processed_operators, _, _ = data_generator.process_all_combined_token_operators(all_combined_token_operators)

            if new_contexts:
                print(f"Processing combined token operators vector differences...")
                _, _ = self.process_combined_token_operators(all_processed_operators, hidden_states)
                print(f"Combined token operators vector differences processed.")
            else:
                print("No contexts were modified. No result evaluated.")

        else:
            raise ValueError(f"Invalid processing mode: {self.config.processing_mode}")
        
        