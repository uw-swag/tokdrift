import os
import json
import glob
import csv
import pandas as pd
from transformers import AutoTokenizer

from .data_generator import DataExtractor, DataGenerator
from .config import Config
from . import tasks

class ResultExtractor:
    
    def __init__(self, config: Config, task_name=None, tokenizer=None):
        self.config = config
        self.task_name = task_name or config.task
        self.tokenizer = tokenizer
        
        # Load task instance
        self.baseline_task = tasks.get_task(self.task_name, data_preprocessing=True, model=self.config.model)
        self.baseline_dataset = self.baseline_task.get_dataset()
        
        # Determine task type
        self.task_type = self._determine_task_type()
        
        # Set up paths
        self.result_dir = config.result_dir
        self.evaluator_output_file = config.evaluator_output_file
        self.baseline_result_dir = config.baseline_result_dir

    def _determine_task_type(self):
        """Determine the task type based on task name"""
        if 'codenet' in self.task_name.lower():
            return 'codenet'
        elif 'humaneval' in self.task_name.lower():
            return 'humaneval'
        elif 'avatar' in self.task_name.lower():
            return 'avatar'
        else:
            raise ValueError(f"Unknown task type for: {self.task_name}")

    def _save_results(self, result_data, changed_boundary=False):
        """Save evaluation results to file"""
        os.makedirs(os.path.dirname(self.evaluator_output_file), exist_ok=True)
        
        if changed_boundary:
            output_file = f"{self.evaluator_output_file.replace('.json', '')}-changed-boundary.json"
        else:
            output_file = self.evaluator_output_file

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)

    def _print_summary(self, result_data):
        """Print summary statistics"""
        summary = result_data.get('summary', {})
        
        print(f"Task: {summary.get('task_name', '')}")
        print(f"Total samples: {summary.get('total_samples', 0)}")
        print(f"Baseline passed: {summary.get('baseline_passed', 0)}")
        print(f"New passed: {summary.get('new_passed', 0)}")
        print(f"Baseline passed idx: {summary.get('baseline_passed_idx', [])}")
        print(f"New passed idx: {summary.get('new_passed_idx', [])}")
        print(f"Baseline wins: {summary.get('baseline_wins', [])}")
        print(f"New wins: {summary.get('new_wins', [])}")
        
        # Print additional info for combined token operators
        if 'total_transformations_processed' in summary:
            print(f"Total transformations processed: {summary['total_transformations_processed']}")

    def check_exceed_token_limit(self, baseline_prompt, variant_prompt):
        """Check if the prompt exceeds the token limit"""
        limit = 2048 if self.config.output_dataset_name == "codenettranslate" else 1024

        if len(self.tokenizer.tokenize(baseline_prompt)) > limit or len(self.tokenizer.tokenize(variant_prompt)) > limit:
            return True
        return False


    def extract_results(self, processing_mode, all_processed_transformations, all_fragment_changed_types=None):
        if processing_mode == "multi_token_identifiers":
            variant_name = self.config.target_type
        elif processing_mode == "combined_token_operators":
            variant_name = self.config.combination_name

        # Add fix if task is humanevalfix
        variant_task_name = f"{self.task_name}-{variant_name}"
        if self.config.output_dataset_name == "humanevalfixtests":
            variant_task_name = f"{self.task_name}-{variant_name}-fix"
        variant_task = tasks.get_task(variant_task_name, data_preprocessing=True, model=self.config.model)
        variant_dataset = variant_task.get_dataset()

        baseline_logs_path = os.path.join(self.baseline_result_dir, "logs.json")
        variant_logs_path = os.path.join(self.result_dir, "logs.json")

        baseline_ids = [self.baseline_task.get_task_id(self.baseline_dataset[i]) for i in range(len(self.baseline_dataset))]
        variant_ids = [variant_task.get_task_id(variant_dataset[i]) for i in range(len(variant_dataset))]

        task_idx = [baseline_ids.index(variant_id) for variant_id in variant_ids]
        
        # Use task-specific processing method
        logs_results = self.baseline_task.process_evaluation_logs(
            baseline_logs_path, variant_logs_path, task_idx, variant_ids
        )

        baseline_passed_count = []
        new_passed_count = []
        total_transformations_processed = 0

        baseline_generations_path = os.path.join(self.baseline_result_dir, f"generations_{self.task_name}.json")
        variant_generations_path = os.path.join(self.result_dir, f"generations_{variant_task_name}.json")

        # Load generations
        with open(baseline_generations_path, 'r') as f:
            baseline_generations = json.load(f)
        with open(variant_generations_path, 'r') as f:
            variant_generations = json.load(f)
        
        for i, idx in enumerate(task_idx):
            baseline_sample = self.baseline_dataset[idx]
            variant_sample = variant_dataset[i]

            log_result = logs_results[i]
            task_id = log_result['task_id']

            if self.check_exceed_token_limit(self.baseline_task.get_prompt(baseline_sample), variant_task.get_prompt(variant_sample)):
                print(f"Prompt exceeds the token limit for task {task_id}")
                continue
            if task_id not in all_processed_transformations:
                print(f"Error: task_id '{task_id}' not found in all_processed_transformations")
                print(f"Available task_ids: {list(all_processed_transformations.keys())}")
                raise ValueError(f"Task_id '{task_id}' not found in all_processed_transformations")

            # Check if the task passed
            if log_result['baseline_passed']:
                baseline_passed_count.append(idx)
            if log_result['new_passed']:
                new_passed_count.append(idx)

            total_transformations_processed += len(all_processed_transformations[task_id])

            baseline_context = self.baseline_task.get_context_only(self.baseline_dataset[idx])
            new_context = variant_task.get_context_only(variant_dataset[i])

            log_result['total_length_difference'] = len(self.tokenizer.tokenize(new_context)) - len(self.tokenizer.tokenize(baseline_context))

            length_increase = 0
            length_decrease = 0
            for cto in all_processed_transformations[task_id]:
                if cto['length_difference'] > 0:
                    length_increase += cto['length_difference']
                elif cto['length_difference'] < 0:
                    length_decrease += cto['length_difference']
            
            log_result['length_increase'] = length_increase
            log_result['length_decrease'] = length_decrease
            log_result['relative_length_difference'] = length_increase + length_decrease

            if all_fragment_changed_types and task_id in all_fragment_changed_types:
                log_result['fragment_changed_types'] = all_fragment_changed_types[task_id]
            else:
                if length_increase > 0 and length_decrease < 0:
                    log_result['fragment_changed_types'] = 'mixed'
                elif length_increase > 0 and length_decrease == 0:
                    log_result['fragment_changed_types'] = 'split'
                elif length_increase == 0 and length_decrease < 0:
                    log_result['fragment_changed_types'] = 'merged'
                else:
                    log_result['fragment_changed_types'] = 'remained'
            

            # NOTE: Currently N_SAMPLES=1, so we can just use the first generation
            baseline_generation = baseline_generations[idx][0]
            new_generation = variant_generations[i][0]

            log_result['baseline_context'] = baseline_context
            log_result['new_context'] = new_context
            log_result['baseline_generation'] = baseline_generation
            log_result['new_generation'] = new_generation
            log_result['transformation_count'] = len(all_processed_transformations[task_id])
            # log_result['processed_combinations'] = all_processed_operators[task_id]
            log_result['processed_transformations'] = []
            for transformation in all_processed_transformations[task_id]:
                # Select some attributes
                if processing_mode == "multi_token_identifiers":
                    log_result['processed_transformations'].append({
                        'transformation': transformation['transformation'],
                        'is_fragment_changed_token': transformation['is_fragment_changed_token'],
                        'length_difference': transformation['length_difference'],
                        'target_identifier': transformation['target_identifier'],
                        'new_target_identifier': transformation['new_target_identifier'],
                        'code_token': transformation['tokenize_token'],
                        'new_code_token': transformation['new_tokenize_token'],
                        'LLM_tokens': transformation['LLM_tokens'],
                        'new_LLM_tokens': transformation['new_LLM_tokens'],
                    })
                    if transformation['length_difference'] == 0 and transformation['is_fragment_changed_token']:
                        log_result['fragment_changed_types'] = 'mixed'
                        
                elif processing_mode == "combined_token_operators":
                    log_result['processed_transformations'].append({
                        'transformation': transformation['transformation'],
                        'is_combined_token': transformation['is_combined_token'],
                        'is_fragment_changed_token': transformation['is_fragment_changed_token'],
                        'length_difference': transformation['length_difference'],
                        'transformation_prev_tokens': transformation['transformation_prev_tokens'],
                        'target_code_token': transformation['operator'],
                        'following_code_token': transformation['following_token'],
                        'LLM_tokens_idx': transformation['LLM_tokens_idx'],
                        'new_LLM_tokens_idx': transformation['new_LLM_tokens_idx'],
                        'target_LLM_tokens_idx': transformation['target_LLM_tokens_idx'],
                        'new_target_LLM_tokens_idx': transformation['new_target_LLM_tokens_idx']
                    })

        summary = {
            'task_name': variant_task_name,
            'total_samples': len(task_idx),
            'baseline_passed': len(baseline_passed_count),
            'new_passed': len(new_passed_count),
            'baseline_passed_idx': baseline_passed_count,
            'new_passed_idx': new_passed_count,
            'baseline_wins': [idx for idx in task_idx if idx in baseline_passed_count and idx not in new_passed_count],
            'new_wins': [idx for idx in task_idx if idx in new_passed_count and idx not in baseline_passed_count],
            'total_transformations_processed': total_transformations_processed
        }

        result_data = {
            'summary': summary,
            'results': logs_results
        }

        self._save_results(result_data)
        self._print_summary(result_data)
    

    def sum_to_csv(self, output_file="./data/output/results_summary.csv"):
        output_dir = "./data/output"
        
        # Collect all unique JSON file names from all task/model combinations
        all_json_names = set()
        task_model_data = []
        
        for task in self.config.all_tasks:
            task_dir = os.path.join(output_dir, task)
            if not os.path.exists(task_dir):
                continue
                
            # Get all model directories
            for model_dir in os.listdir(task_dir):
                model_path = os.path.join(task_dir, model_dir)
                if not os.path.isdir(model_path):
                    continue
                    
                # Collect JSON files in this model directory
                json_files = [f for f in os.listdir(model_path) if f.endswith('.json')]
                all_json_names.update(json_files)
                task_model_data.append((task, model_dir, model_path, json_files))
        
        # Sort column names for consistency and remove .json extension
        json_columns = sorted([name.replace('.json', '') for name in all_json_names])

        temp_config = Config()
        temp_config.task = "humanevalexplaindescribe-python"
        temp_config.config_task()
        target_types = [f"{temp_config.filter_type}-{tt}" for tt in temp_config.target_types]
        python_columns = target_types + temp_config.operator_variants

        temp_config.task = "humanevalexplaindescribe-java"
        temp_config.config_task()
        target_types = [f"{temp_config.filter_type}-{tt}" for tt in temp_config.target_types]
        java_columns = target_types + temp_config.operator_variants

        # Helper function to get language from task
        def get_task_lang(task_name):
            temp_config = Config()
            temp_config.task = task_name
            temp_config.config_task()
            return temp_config.lang
        
        # Helper function to get baseline accuracy
        def get_baseline_accuracy(task_name, model_name, model_path):
            try:
                # Create temporary config and task instance
                temp_config = Config()
                temp_config.task = task_name
                
                # Get full model name
                full_model_name = None
                for full_name in temp_config.model_list:
                    if model_name in full_name:
                        full_model_name = full_name
                        break
                
                temp_config.model = full_model_name
                temp_config.config_task()
                
                baseline_task = tasks.get_task(task_name, data_preprocessing=True, model=temp_config.model)
                baseline_dataset = baseline_task.get_dataset()
                
                # Get all task IDs from baseline dataset
                all_task_ids = [baseline_task.get_task_id(baseline_dataset[i]) for i in range(len(baseline_dataset))]
                total_samples = len(all_task_ids)
                
                # Process baseline results
                baseline_logs_path = os.path.join(model_path, "baseline", "logs.json")
                if os.path.exists(baseline_logs_path):
                    # Use the same baseline for both baseline and variant logs since we want baseline results
                    baseline_ids = all_task_ids
                    logs_results = baseline_task.process_evaluation_logs(
                        baseline_logs_path, baseline_logs_path, list(range(len(all_task_ids))), baseline_ids
                    )
                    
                    # Count baseline passed
                    baseline_passed_count = sum(1 for log_result in logs_results if log_result['baseline_passed'])
                    baseline_accuracy = baseline_passed_count / total_samples * 100
                    
                    return baseline_accuracy, baseline_passed_count, total_samples
                else:
                    return 0, 0, total_samples
            except Exception as e:
                print(f"Error calculating baseline accuracy for {task_name} {model_name}: {e}")
                return 0, 0, 0
        
        # Prepare data for all CSVs
        def prepare_data_rows(calculation_type, target_columns=None):
            if calculation_type in ['python_avg_accuracy', 'java_avg_accuracy']:
                # For language-specific average accuracy, group by model and calculate aggregated metrics
                target_lang = 'python' if calculation_type == 'python_avg_accuracy' else 'java'
                model_metrics = {}

                for task, model_name, model_path, json_files in task_model_data:
                    # Only process tasks for the target language
                    task_lang = get_task_lang(task)
                    if task_lang != target_lang:
                        continue
                    if model_name not in model_metrics:
                        model_metrics[model_name] = {
                            'baseline_total_passed': 0,
                            'baseline_total_samples': 0,
                            'json_data': {}
                        }
                    
                    # Get baseline metrics for this task
                    baseline_accuracy, baseline_passed_count, total_samples = get_baseline_accuracy(task, model_name, model_path)
                    model_metrics[model_name]['baseline_total_passed'] += baseline_passed_count
                    model_metrics[model_name]['baseline_total_samples'] += total_samples
                    
                    # Process JSON files for this task
                    for json_file in json_files:
                        json_path = os.path.join(model_path, json_file)
                        column_name = json_file.replace('.json', '')
                        
                        if column_name not in model_metrics[model_name]['json_data']:
                            model_metrics[model_name]['json_data'][column_name] = {
                                'adjusted_passed': 0,
                                'total_samples': 0
                            }
                        
                        try:
                            with open(json_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                summary = data.get('summary', {})
                                
                                total_samples = summary.get('total_samples', 0)
                                baseline_passed = summary.get('baseline_passed', 0)
                                new_passed = summary.get('new_passed', 0)
                                baseline_wins = len(summary.get('baseline_wins', []))
                                new_wins = len(summary.get('new_wins', []))
                                
                                # Get baseline accuracy info for adjusted calculation
                                baseline_accuracy, all_baseline_passed_count, full_dataset_size = get_baseline_accuracy(task, model_name, model_path)
                                # adjusted_passed = all_baseline_passed_count - baseline_passed + new_passed
                                adjusted_passed = all_baseline_passed_count - baseline_wins + new_wins
                                
                                model_metrics[model_name]['json_data'][column_name]['adjusted_passed'] += adjusted_passed
                                model_metrics[model_name]['json_data'][column_name]['total_samples'] += full_dataset_size
                                
                        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                            print(f"Error reading {json_path}: {e}")
                            # Skip this file for the aggregation
                            continue
                
                # Convert to rows
                rows = []
                for model_name, metrics in model_metrics.items():
                    # Calculate baseline average
                    baseline_avg = (metrics['baseline_total_passed'] / metrics['baseline_total_samples'] * 100) if metrics['baseline_total_samples'] > 0 else 0
                    
                    row = {'model_name': model_name, 'baseline': round(baseline_avg, 2)}
                    
                    # Initialize columns based on target language
                    if target_columns is None:
                        target_columns = json_columns
                    for json_name in target_columns:
                        row[json_name] = ""
                    
                    # Fill columns with adjusted averages (only if column exists in target_columns)
                    for json_name, json_metrics in metrics['json_data'].items():
                        if json_name in target_columns and json_metrics['total_samples'] > 0:
                            adjusted_avg = (json_metrics['adjusted_passed'] / json_metrics['total_samples'] * 100)
                            row[json_name] = round(adjusted_avg, 2)
                    
                    rows.append(row)
                
                return rows
            
            else:
                # For other CSV results
                rows = []
                for task, model_name, model_path, json_files in task_model_data:
                    lang = get_task_lang(task)
                    
                    if calculation_type == 'accuracy':
                        # For accuracy CSV, get baseline accuracy
                        baseline_accuracy, baseline_passed_count, total_samples = get_baseline_accuracy(task, model_name, model_path)
                        row = {'task_name': task, 'lang': lang, 'model_name': model_name, 'baseline': round(baseline_accuracy, 2)}
                    else:
                        # For sensitivity and acc_delta CSVs
                        row = {'task_name': task, 'lang': lang, 'model_name': model_name}
                
                    # Initialize all JSON columns with empty values
                    for json_name in json_columns:
                        row[json_name] = ""
                    
                    # Fill in data for existing JSON files
                    for json_file in json_files:
                        json_path = os.path.join(model_path, json_file)
                        column_name = json_file.replace('.json', '')
                        try:
                            with open(json_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                summary = data.get('summary', {})
                                
                                # Extract key metrics from summary
                                total_samples = summary.get('total_samples', 0)
                                baseline_passed = summary.get('baseline_passed', 0)
                                new_passed = summary.get('new_passed', 0)
                                baseline_wins = len(summary.get('baseline_wins', []))
                                new_wins = len(summary.get('new_wins', []))

                                if calculation_type == 'sensitivity':
                                    value = (baseline_wins + new_wins) / total_samples * 100 if total_samples > 0 else 0
                                elif calculation_type == 'acc_delta':
                                    value = (new_passed - baseline_passed) / total_samples * 100 if total_samples > 0 else 0
                                elif calculation_type == 'accuracy':
                                    if total_samples > 0:
                                        # Get baseline accuracy info
                                        baseline_accuracy, all_baseline_passed_count, full_dataset_size = get_baseline_accuracy(task, model_name, model_path)
                                        adjusted_accuracy = (all_baseline_passed_count - baseline_passed + new_passed) / full_dataset_size * 100
                                        value = adjusted_accuracy
                                    else:
                                        value = 0
                                
                                row[column_name] = round(value, 2)
                                
                        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                            print(f"Error reading {json_path}: {e}")
                            row[column_name] = "error"
                    
                    rows.append(row)
                
                return rows
            
            return rows
        
        # Helper function to calculate averages and max
        def calculate_summary_rows(rows, calculation_type, target_columns=None):
            if calculation_type == 'accuracy':
                average_row = {'task_name': 'average', 'lang': 'average', 'model_name': 'average', 'baseline': ''}
                max_row = {'task_name': 'max', 'lang': 'max', 'model_name': 'max', 'baseline': ''}
            elif calculation_type in ['python_avg_accuracy', 'java_avg_accuracy']:
                average_row = {'model_name': 'average', 'baseline': ''}
                max_row = {'model_name': 'max', 'baseline': ''}
            else:
                average_row = {'task_name': 'average', 'lang': 'average', 'model_name': 'average'}
                max_row = {'task_name': 'max', 'lang': 'max', 'model_name': 'max'}
            
            # Handle baseline column for accuracy and language-specific avg_accuracy types
            if calculation_type in ['accuracy', 'python_avg_accuracy', 'java_avg_accuracy']:
                baseline_values = []
                for row in rows:
                    value = row.get('baseline', None)
                    if isinstance(value, (int, float)) and value != "error":
                        baseline_values.append(value)
                    elif isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                        baseline_values.append(float(value))
                
                if baseline_values:
                    avg_baseline = sum(baseline_values) / len(baseline_values)
                    max_baseline = max(baseline_values)
                    average_row['baseline'] = round(avg_baseline, 2)
                    max_row['baseline'] = round(max_baseline, 2)
                else:
                    average_row['baseline'] = ""
                    max_row['baseline'] = ""

            # Use target_columns if provided, otherwise use json_columns
            columns_to_process = target_columns if target_columns is not None else json_columns
            for json_name in columns_to_process:
                # Collect all valid numeric values for this column
                valid_values = []
                for row in rows:
                    value = row.get(json_name, None)
                    # Include if it's a number (including 0), exclude if None, empty string, or "error"
                    if isinstance(value, (int, float)) and value != "error":
                        valid_values.append(value)
                    elif isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                        valid_values.append(float(value))
                
                # Calculate average and max
                if valid_values:
                    avg_value = sum(valid_values) / len(valid_values)
                    max_value = max(valid_values)
                    average_row[json_name] = round(avg_value, 2)
                    max_row[json_name] = round(max_value, 2)
                else:
                    average_row[json_name] = ""
                    max_row[json_name] = ""
            
            return average_row, max_row
        
        # Generate six CSV files (including separate Python and Java avg_accuracy)
        csv_configs = [
            ('sensitivity', '_sensitivity.csv', ['task_name', 'lang', 'model_name'] + json_columns, json_columns),
            ('acc_delta', '_acc_delta.csv', ['task_name', 'lang', 'model_name'] + json_columns, json_columns),
            ('accuracy', '_accuracy.csv', ['task_name', 'lang', 'model_name', 'baseline'] + json_columns, json_columns),
            ('python_avg_accuracy', '_python_avg_accuracy.csv', ['model_name', 'baseline'] + python_columns, python_columns),
            ('java_avg_accuracy', '_java_avg_accuracy.csv', ['model_name', 'baseline'] + java_columns, java_columns)
        ]
        
        output_files = []
        for calc_type, suffix, headers, target_cols in csv_configs:
            # Prepare data rows
            rows = prepare_data_rows(calc_type, target_cols)

            # Calculate summary rows
            average_row, max_row = calculate_summary_rows(rows, calc_type, target_cols)
            rows.append(average_row)
            rows.append(max_row)
            
            # Generate output file name
            base_name = output_file.replace('.csv', '')
            current_output_file = base_name + suffix
            
            # Write to CSV
            os.makedirs(os.path.dirname(current_output_file), exist_ok=True)
            with open(current_output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)
            
            output_files.append(current_output_file)
            print(f"Results summary saved to: {current_output_file}")
        
        return output_files


if __name__ == "__main__":
    import argparse
    from .config_iterator import ConfigIterator

    parser = argparse.ArgumentParser(description="Evaluate TokenDrift results")
    parser.add_argument("--all", action="store_true", help="Process all results in the output directory")
    parser.add_argument("--all_tasks", action="store_true", help="Process all tasks", default=False)
    parser.add_argument("--all_models", action="store_true", help="Process all models", default=False)
    parser.add_argument("--all_multi_token_identifiers", action="store_true", help="Process all multi-token identifiers", default=False)
    parser.add_argument("--all_combined_token_operators", action="store_true", help="Process all combined token operators", default=False)
    parser.add_argument("--task", help="Process a specific task name", default=None)

    parser.add_argument("--sum_to_csv", action="store_true", help="Tidy up all summaries to a CSV file", default=False)

    args = parser.parse_args()

    config = Config()

    if args.all:
        args.all_tasks = True
        args.all_models = True
        args.all_multi_token_identifiers = True
        args.all_combined_token_operators = True

    if args.sum_to_csv:
        result_evaluator = ResultExtractor(config)
        result_evaluator.sum_to_csv()
        exit(0)

    elif args.task:
        if args.task == "avatar":
            config.all_tasks = ["avatartranslate-python2java", "avatartranslate-java2python"]
        elif args.task == "codenet":
            config.all_tasks = ["codenettranslate-python2java", "codenettranslate-java2python"]
        elif args.task == "explain":
            config.all_tasks = ["humanevalexplaindescribe-python", "humanevalexplaindescribe-java"]
        elif args.task == "fix":
            config.all_tasks = ["humanevalfixtests-python", "humanevalfixtests-java"]
        else:
            raise ValueError(f"Unknown task: {args.task}")
        args.all_tasks = True
        args.all_models = True
        args.all_multi_token_identifiers = True
        args.all_combined_token_operators = True


    # Use ConfigIterator for main processing loop
    iterator = ConfigIterator(config)
    

    for task, model, processing_mode, tokenizer in iterator.iterate_all(
        all_tasks=args.all_tasks,
        all_models=args.all_models,
        all_multi_token_identifiers=args.all_multi_token_identifiers,
        all_combined_token_operators=args.all_combined_token_operators
    ):
        data_extractor = DataExtractor(tokenizer, config)
        data_generator = DataGenerator(tokenizer, config)

        if processing_mode == "multi_token_identifiers":
            # Extract identifiers
            print(f"Extracting identifiers...")
            all_multi_token_identifiers, all_test_identifiers = data_extractor.extract_new_identifiers()

            # Generate new multi-token dataset
            print(f"Generating new multi-token dataset...")
            new_contexts, all_selected_multi_token_identifiers, all_token_boundary_changed, all_modified_tests, all_modified_entry_points, all_modified_declarations = data_generator.process_all_multi_token_identifiers(all_multi_token_identifiers, all_test_identifiers)

            if new_contexts:
                print(f"Evaluating the results...")
                result_evaluator = ResultExtractor(config, tokenizer=tokenizer)
                result_evaluator.extract_results(config.processing_mode, all_selected_multi_token_identifiers)
            else:
                print("No contexts were modified. No result evaluated.")

        elif processing_mode == "combined_token_operators":
            # Extract combined token operators
            print(f"Extracting combined token operators...")
            all_combined_token_operators = data_extractor.extract_combined_token_operators()

            # Process combined token operators
            print(f"Finding and processing combined token operators...")
            new_contexts, all_processed_operators, all_token_boundary_changed, all_fragment_changed_types = data_generator.process_all_combined_token_operators(all_combined_token_operators)

            if new_contexts:
                print(f"Evaluating the results...")
                result_evaluator = ResultExtractor(config, tokenizer=tokenizer)
                result_evaluator.extract_results(config.processing_mode, all_processed_operators, all_fragment_changed_types)
            else:
                print("No contexts were modified. No result evaluated.")