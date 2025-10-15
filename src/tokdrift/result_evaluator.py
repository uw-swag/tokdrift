from tkinter.font import names
from .config import Config
from dataclasses import dataclass
from typing import Optional
import numpy as np
import csv
import os
import json


@dataclass
class FilterCriteria:
    task: Optional[str] = None
    model: Optional[str] = None
    lang: Optional[str] = None
    mode: Optional[str] = None
    variant: Optional[str] = None
    tokenizer: Optional[str] = None



class ResultEvaluator:
    def __init__(self, config: Config):
        self.config = config

        self.fragment_changed = []
        self.fragment_changed_merging = []
        self.fragment_changed_spliting = []
        self.fragment_changed_mixing = []
        self.fragment_unchanged = []

        self.filtered_fragment_changed = []
        self.filtered_fragment_changed_merging = []
        self.filtered_fragment_changed_spliting = []
        self.filtered_fragment_changed_mixing = []
        self.filtered_fragment_unchanged = []

        self.all_columns = []

        self.all_p = {}
        self.filtered_transformations = {}

    def load_data_points(self):
        """Load data points from the evaluator output file and categorize them."""

        # Check if the evaluator output file exists
        if not os.path.exists(self.config.evaluator_output_file):
            raise FileNotFoundError(f"Evaluator output file not found: {self.config.evaluator_output_file}")

        # Load the JSON data
        with open(self.config.evaluator_output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Get the results list
        results = data.get('results', [])

        # Process each data point
        for data_point in results:
            # Add metadata fields
            data_point['task'] = self.config.output_dataset_name
            data_point['model'] = self.config.model_name
            data_point['lang'] = self.config.lang
            data_point['mode'] = self.config.processing_mode
            data_point['variant'] = variant

            # Categorize based on fragment_changed
            fragment_changed_type = data_point.get('fragment_changed_types', 'remained')

            # if fragment_changed:
            if fragment_changed_type != "remained":
                self.fragment_changed.append(data_point)

                # Further categorize
                if fragment_changed_type == "split":
                    # Only splitting occurred
                    self.fragment_changed_spliting.append(data_point)
                elif fragment_changed_type == "merged":
                    # Only merging occurred
                    self.fragment_changed_merging.append(data_point)
                elif fragment_changed_type == "mixed":
                    # Both splitting and merging occurred
                    self.fragment_changed_mixing.append(data_point)

            else:
                self.fragment_unchanged.append(data_point)


    def _matches_criteria(self, data_point, criteria: FilterCriteria) -> bool:
        """Check if a data point matches the filter criteria."""
        if criteria.task is not None and data_point.get('task') != criteria.task:
            return False
        if criteria.model is not None:
            # Handle model filtering - if '/' exists in model, use the second part
            data_model = data_point.get('model', '')
            if '/' in data_model:
                data_model = data_model.split('/')[-1]
            if '/' in criteria.model:
                criteria.model = criteria.model.split('/')[-1]
            if data_model != criteria.model:
                return False
        if criteria.lang is not None and data_point.get('lang') != criteria.lang:
            return False
        if criteria.mode is not None and data_point.get('mode') != criteria.mode:
            return False
        if criteria.variant is not None and data_point.get('variant') != criteria.variant:
            return False
        if criteria.tokenizer is not None:
            if criteria.tokenizer not in data_point.get('model', ''):
                return False
        return True

    def filter_data_points(self, filter_criteria: FilterCriteria, filter_out_criteria: FilterCriteria):
        """Filter data points based on filter and filter_out criteria."""
        all_data_points = (self.fragment_changed + self.fragment_unchanged)

        # Clear filtered lists
        self.filtered_fragment_changed = []
        self.filtered_fragment_changed_merging = []
        self.filtered_fragment_changed_spliting = []
        self.filtered_fragment_changed_mixing = []
        self.filtered_fragment_unchanged = []

        for data_point in all_data_points:
            # Check if it matches filter criteria
            matches_filter = self._matches_criteria(data_point, filter_criteria)

            # Check if it matches filter_out criteria (should be excluded)
            matches_filter_out = self._matches_criteria(data_point, filter_out_criteria)

            # Include if it matches filter and doesn't match filter_out
            if matches_filter and not matches_filter_out:
                fragment_changed_type = data_point.get('fragment_changed_types', 'remained')

                if fragment_changed_type != "remained":
                    self.filtered_fragment_changed.append(data_point)

                    if fragment_changed_type == "split":
                        self.filtered_fragment_changed_spliting.append(data_point)
                    elif fragment_changed_type == "merged":
                        self.filtered_fragment_changed_merging.append(data_point)
                    elif fragment_changed_type == "mixed":
                        self.filtered_fragment_changed_mixing.append(data_point)
                else:
                    self.filtered_fragment_unchanged.append(data_point)

    def get_wilcoxon_test_p(self, model_s: str, model_l: str, tokenizer: str):
        """Calculate Wilcoxon signed-rank test p-value for the filtered data points."""
        from scipy.stats import wilcoxon
        self.all_columns = ["all", "multi_token_identifiers", "combined_token_operators"]

        tokenizer = tokenizer
        if tokenizer not in self.all_p:
            self.all_p[tokenizer] = {
                "all": {col: 1 for col in self.all_columns},
                "fragment_changed": {col: 1 for col in self.all_columns},
                "fragment_unchanged": {col: 1 for col in self.all_columns},
            }

        data_points = [self.all_sensitivity, self.fragment_changed_sensitivity, self.fragment_unchanged_sensitivity]
        names = ["all", "fragment_changed", "fragment_unchanged"]
        for i, data_point in enumerate(data_points):
            id_large = []
            id_small = []
            op_large = []
            op_small = []
            all_large = []
            all_small = []

            for name, pct in data_point[model_s]["pct"].items():
                if "-" in name:
                    id_small.append(pct)
                    id_large.append(data_point[model_l]["pct"].get(name, 0.0))
                else:
                    op_small.append(pct)
                    op_large.append(data_point[model_l]["pct"].get(name, 0.0))
                all_small.append(pct)
                all_large.append(data_point[model_l]["pct"].get(name, 0.0))

            if len(id_small) > 0 and len(id_large) > 0 and len(id_small) == len(id_large):
                res = wilcoxon(id_small, id_large, alternative='greater')
                self.all_p[tokenizer][names[i]]["multi_token_identifiers"] = res.pvalue
                print(f"Tokenizer: {tokenizer}, multi_token_identifiers Wilcoxon test p-value: {res.pvalue}")
            if len(op_small) > 0 and len(op_large) > 0 and len(op_small) == len(op_large):
                res = wilcoxon(op_small, op_large, alternative='greater')
                self.all_p[tokenizer][names[i]]["combined_token_operators"] = res.pvalue
                print(f"Tokenizer: {tokenizer}, combined_token_operators Wilcoxon test p-value: {res.pvalue}")
            if len(all_small) > 0 and len(all_large) > 0 and len(all_small) == len(all_large):
                res = wilcoxon(all_small, all_large, alternative='greater')
                self.all_p[tokenizer][names[i]]["all"] = res.pvalue
                print(f"Tokenizer: {tokenizer}, all Wilcoxon test p-value: {res.pvalue}")
    
    def output_wilcoxon_test_p(self, test_name: str):
        """Output Wilcoxon signed-rank test p-values to CSV files."""
        output_dir = "./data/output/sensitivity/wilcoxon_test"
        os.makedirs(output_dir, exist_ok=True)
        wilcoxon_file = os.path.join(output_dir, f"{test_name}.csv")

        with open(wilcoxon_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            header = ["tokenizer"] + [col for col in self.all_columns]
            writer.writerow(header)

            for tokenizer in self.all_p:
                row = [tokenizer]
                row += [f"{self.all_p[tokenizer]['all'][col]:.6f}" for col in self.all_columns]
                writer.writerow(row)

        print(f"Wilcoxon signed-rank test p-value CSV files saved to: {output_dir}")
        self.all_p = {}
    

    def get_combined_transformations(self, filter_criteria: FilterCriteria):
        """Get combined transformations for the filtered data points."""
        self.filtered_transformations = {}

        for data_point in self.filtered_fragment_changed:
            processed_trans = data_point.get('processed_transformations', [])
            for trans in processed_trans:
                if trans["length_difference"] not in self.filtered_transformations:
                    self.filtered_transformations[trans["length_difference"]] = {}
                if trans["transformation"] not in self.filtered_transformations[trans["length_difference"]]:
                    self.filtered_transformations[trans["length_difference"]][trans["transformation"]] = 0
                self.filtered_transformations[trans["length_difference"]][trans["transformation"]] += 1
        
        # Sort the transformations by length_difference and then by count
        for length_diff in self.filtered_transformations:
            self.filtered_transformations[length_diff] = dict(sorted(self.filtered_transformations[length_diff].items(), key=lambda item: item[1], reverse=True))
        
        # Output the combined transformations to a json file
        output_dir = f"./data/output/combined_token_analysis/trans_changed/{filter_criteria.tokenizer}"
        os.makedirs(output_dir, exist_ok=True)
        transformations_file = os.path.join(output_dir, f"{filter_criteria.variant}.json")
        import json
        with open(transformations_file, 'w', encoding='utf-8') as f:
            json.dump(self.filtered_transformations, f, indent=4)

        self.filtered_transformations = {}
        for data_point in self.filtered_fragment_unchanged:
            processed_trans = data_point.get('processed_transformations', [])
            for trans in processed_trans:
                if trans["length_difference"] not in self.filtered_transformations:
                    self.filtered_transformations[trans["length_difference"]] = {}
                if trans["transformation"] not in self.filtered_transformations[trans["length_difference"]]:
                    self.filtered_transformations[trans["length_difference"]][trans["transformation"]] = 0
                self.filtered_transformations[trans["length_difference"]][trans["transformation"]] += 1
        
        # Sort the transformations by length_difference and then by count
        for length_diff in self.filtered_transformations:
            self.filtered_transformations[length_diff] = dict(sorted(self.filtered_transformations[length_diff].items(), key=lambda item: item[1], reverse=True))
        
        # Output the combined transformations to a json file
        output_dir = f"./data/output/combined_token_analysis/trans_unchanged/{filter_criteria.tokenizer}"
        os.makedirs(output_dir, exist_ok=True)
        transformations_file = os.path.join(output_dir, f"{filter_criteria.variant}.json")
        with open(transformations_file, 'w', encoding='utf-8') as f:
            json.dump(self.filtered_transformations, f, indent=4)
        print(f"Combined transformations JSON file saved to: {transformations_file}")

    def get_sensitivity(self):
        """Output sensitivity CSV file."""
        all_columns, python_columns, java_columns = self.config.get_all_variants()

        self.all_sensitivity = {}
        self.fragment_changed_sensitivity = {}
        self.fragment_unchanged_sensitivity = {}
        self.fragment_merging_sensitivity = {}
        self.fragment_splitting_sensitivity = {}
        self.fragment_mixing_sensitivity = {}

        for data_point in self.fragment_changed:

            model = data_point.get('model', 'unknown')
            if model not in self.all_sensitivity:
                self.all_sensitivity[model] = {
                    "total": {col: 0 for col in all_columns},
                    "count": {col: 0 for col in all_columns},
                    "pct": {col: 0.00 for col in all_columns}
                }
            if model not in self.fragment_changed_sensitivity:
                self.fragment_changed_sensitivity[model] = {
                    "total": {col: 0 for col in all_columns},
                    "count": {col: 0 for col in all_columns},
                    "pct": {col: 0.00 for col in all_columns}
                }
                self.fragment_merging_sensitivity[model] = {
                    "total": {col: 0 for col in all_columns},
                    "count": {col: 0 for col in all_columns},
                    "pct": {col: 0.00 for col in all_columns}
                }
                self.fragment_splitting_sensitivity[model] = {
                    "total": {col: 0 for col in all_columns},
                    "count": {col: 0 for col in all_columns},
                    "pct": {col: 0.00 for col in all_columns}
                }
                self.fragment_mixing_sensitivity[model] = {
                    "total": {col: 0 for col in all_columns},
                    "count": {col: 0 for col in all_columns},
                    "pct": {col: 0.00 for col in all_columns}
                }
            
            mode = data_point.get('mode', 'unknown')
            variant = data_point.get('variant', 'unknown')
            if mode == "multi_token_identifiers":
                lang = data_point.get('lang', 'unknown')
                if lang == "python":
                    # FIXME: hardcoded for now
                    variant = "snake_case-" + variant
                if lang == "java":
                    variant = "camel_case-" + variant

            self.all_sensitivity[model]["total"][variant] += 1
            self.fragment_changed_sensitivity[model]["total"][variant] += 1
            fragment_changed_type = data_point.get('fragment_changed_types', 'remained')
            if fragment_changed_type == "split":
                self.fragment_splitting_sensitivity[model]["total"][variant] += 1
            elif fragment_changed_type == "merged":
                self.fragment_merging_sensitivity[model]["total"][variant] += 1
            elif fragment_changed_type == "mixed":
                self.fragment_mixing_sensitivity[model]["total"][variant] += 1

            if data_point.get('baseline_passed', False) != data_point.get('new_passed', False):
                self.all_sensitivity[model]["count"][variant] += 1
                self.fragment_changed_sensitivity[model]["count"][variant] += 1
                if fragment_changed_type == "split":
                    self.fragment_splitting_sensitivity[model]["count"][variant] += 1
                elif fragment_changed_type == "merged":
                    self.fragment_merging_sensitivity[model]["count"][variant] += 1
                elif fragment_changed_type == "mixed":
                    self.fragment_mixing_sensitivity[model]["count"][variant] += 1

        for data_point in self.fragment_unchanged:

            model = data_point.get('model', 'unknown')
            if model not in self.all_sensitivity:
                self.all_sensitivity[model] = {
                    "total": {col: 0 for col in all_columns},
                    "count": {col: 0 for col in all_columns},
                    "pct": {col: 0.00 for col in all_columns}
                }
            if model not in self.fragment_unchanged_sensitivity:
                self.fragment_unchanged_sensitivity[model] = {
                    "total": {col: 0 for col in all_columns},
                    "count": {col: 0 for col in all_columns},
                    "pct": {col: 0.00 for col in all_columns}
                }
            mode = data_point.get('mode', 'unknown')
            variant = data_point.get('variant', 'unknown')
            if mode == "multi_token_identifiers":
                lang = data_point.get('lang', 'unknown')
                if lang == "python":
                    variant = "snake_case-" + variant
                if lang == "java":
                    variant = "camel_case-" + variant

            self.all_sensitivity[model]["total"][variant] += 1
            self.fragment_unchanged_sensitivity[model]["total"][variant] += 1

            if data_point.get('baseline_passed', False) != data_point.get('new_passed', False):
                self.all_sensitivity[model]["count"][variant] += 1
                self.fragment_unchanged_sensitivity[model]["count"][variant] += 1

        # Calculate sensitivity
        for model in self.all_sensitivity:
            for col in all_columns:
                total = self.all_sensitivity[model]["total"][col]
                count = self.all_sensitivity[model]["count"][col]
                self.all_sensitivity[model]["pct"][col] = (count / total * 100) if total > 0 else 0.0

                total = self.fragment_changed_sensitivity[model]["total"][col]
                count = self.fragment_changed_sensitivity[model]["count"][col]
                self.fragment_changed_sensitivity[model]["pct"][col] = (count / total * 100) if total > 0 else 0.0

                total = self.fragment_unchanged_sensitivity[model]["total"][col]
                count = self.fragment_unchanged_sensitivity[model]["count"][col]
                self.fragment_unchanged_sensitivity[model]["pct"][col] = (count / total * 100) if total > 0 else 0.0

                total = self.fragment_merging_sensitivity[model]["total"][col]
                count = self.fragment_merging_sensitivity[model]["count"][col]
                self.fragment_merging_sensitivity[model]["pct"][col] = (count / total * 100) if total > 0 else 0.0

                total = self.fragment_splitting_sensitivity[model]["total"][col]
                count = self.fragment_splitting_sensitivity[model]["count"][col]
                self.fragment_splitting_sensitivity[model]["pct"][col] = (count / total * 100) if total > 0 else 0.0

                total = self.fragment_mixing_sensitivity[model]["total"][col]
                count = self.fragment_mixing_sensitivity[model]["count"][col]
                self.fragment_mixing_sensitivity[model]["pct"][col] = (count / total * 100) if total > 0 else 0.0            
            
                
    
    def output_sensitivity(self):
        """Output sensitivity CSV file."""
        # Write to CSV files
        sample_count_dir = "./data/output/sample_info"
        sensitivity_output_dir = "./data/output/sensitivity"
        os.makedirs(sample_count_dir, exist_ok=True)
        os.makedirs(sensitivity_output_dir, exist_ok=True)
        all_columns, python_columns, java_columns = self.config.get_all_variants()

        for name in ["all", "fragment_changed", "fragment_unchanged", "merging", "splitting", "mixing"]:
            sample_count_file = os.path.join(sample_count_dir, f"{name}_total_samples.csv")
            diff_samples_file = os.path.join(sample_count_dir, f"{name}_diff_samples.csv")
            sensitivity_file = os.path.join(sensitivity_output_dir, f"{name}_sensitivity.csv")
            models_sensitivity_file = os.path.join(sensitivity_output_dir, f"models/{name}_sensitivity.csv")
            with open(sample_count_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                header = ["model"] + all_columns
                writer.writerow(header)

                for model in self.all_sensitivity:
                    if name == "all":
                        row = [model] + [self.all_sensitivity[model]["total"][col] for col in all_columns]
                    elif name == "fragment_changed":
                        row = [model] + [self.fragment_changed_sensitivity[model]["total"][col] for col in all_columns]
                    elif name == "fragment_unchanged":
                        row = [model] + [self.fragment_unchanged_sensitivity[model]["total"][col] for col in all_columns]
                    elif name == "merging":
                        row = [model] + [self.fragment_merging_sensitivity[model]["total"][col] for col in all_columns]
                    elif name == "splitting":
                        row = [model] + [self.fragment_splitting_sensitivity[model]["total"][col] for col in all_columns]
                    elif name == "mixing":
                        row = [model] + [self.fragment_mixing_sensitivity[model]["total"][col] for col in all_columns]
                    writer.writerow(row)

            with open(diff_samples_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                header = ["model"] + all_columns
                writer.writerow(header)

                for model in self.all_sensitivity:
                    if name == "all":
                        row = [model] + [self.all_sensitivity[model]["count"][col] for col in all_columns]
                    elif name == "fragment_changed":
                        row = [model] + [self.fragment_changed_sensitivity[model]["count"][col] for col in all_columns]
                    elif name == "fragment_unchanged":
                        row = [model] + [self.fragment_unchanged_sensitivity[model]["count"][col] for col in all_columns]
                    elif name == "merging":
                        row = [model] + [self.fragment_merging_sensitivity[model]["count"][col] for col in all_columns]
                    elif name == "splitting":
                        row = [model] + [self.fragment_splitting_sensitivity[model]["count"][col] for col in all_columns]
                    elif name == "mixing":
                        row = [model] + [self.fragment_mixing_sensitivity[model]["count"][col] for col in all_columns]
                    writer.writerow(row)
            
            with open(sensitivity_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                header = ["model"] + all_columns
                writer.writerow(header)

                for model in self.all_sensitivity:
                    if name == "all":
                        row = [model] + [f"{self.all_sensitivity[model]['pct'][col]:.2f}" for col in all_columns]
                    elif name == "fragment_changed":
                        row = [model] + [f"{self.fragment_changed_sensitivity[model]['pct'][col]:.2f}" for col in all_columns]
                    elif name == "fragment_unchanged":
                        row = [model] + [f"{self.fragment_unchanged_sensitivity[model]['pct'][col]:.2f}" for col in all_columns]
                    elif name == "merging":
                        row = [model] + [f"{self.fragment_merging_sensitivity[model]['pct'][col]:.2f}" for col in all_columns]
                    elif name == "splitting":
                        row = [model] + [f"{self.fragment_splitting_sensitivity[model]['pct'][col]:.2f}" for col in all_columns]
                    elif name == "mixing":
                        row = [model] + [f"{self.fragment_mixing_sensitivity[model]['pct'][col]:.2f}" for col in all_columns]
                    writer.writerow(row)
            

    def output_sensitivity_by_models(self):
        """Output sensitivity by models to CSV file."""
        sample_count_dir = "./data/output/sample_info"
        output_dir = "./data/output/sensitivity/models"
        os.makedirs(output_dir, exist_ok=True)
        sensitivity_by_models_file = os.path.join(output_dir, f"sensitivity_by_models.csv")
        all_columns = ["naming", "spacing"]
        all_names = ["all", "fragment_changed", "fragment_unchanged", "merging", "splitting", "mixing"]
        self.sensitivity_by_models = {}

        for name in all_names:
            sample_count_file = os.path.join(sample_count_dir, f"{name}_total_samples.csv")
            diff_samples_file = os.path.join(sample_count_dir, f"{name}_diff_samples.csv")

            total_naming = {}
            total_spacing = {}
            count_naming = {}
            count_spacing = {}

            # Read total and count from sample_count_file
            with open(sample_count_file, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader)
                # Naming index is the index of the first column that contains "-"
                naming_index = []
                spacing_index = []
                # Skip the first column
                for i, col in enumerate(header[1:]):
                    if "-" in col:
                        naming_index.append(i + 1)
                    else:
                        spacing_index.append(i + 1)
                for row in reader:
                    model = row[0]
                    if model not in self.sensitivity_by_models:
                        self.sensitivity_by_models[model] = {
                            name: {col: 0.00 for col in all_columns} for name in all_names
                        }
                    total_naming[model] = sum([int(row[i]) for i in naming_index])
                    total_spacing[model] = sum([int(row[i]) for i in spacing_index])

            with open(diff_samples_file, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader)
                naming_index = []
                spacing_index = []
                # Skip the first column
                for i, col in enumerate(header[1:]):
                    if "-" in col:
                        naming_index.append(i + 1)
                    else:
                        spacing_index.append(i + 1)
                for row in reader:
                    model = row[0]
                    count_naming[model] = sum([int(row[i]) for i in naming_index])
                    count_spacing[model] = sum([int(row[i]) for i in spacing_index])
            
            for model in total_naming:
                self.sensitivity_by_models[model][name]["naming"] = (count_naming[model] / total_naming[model] * 100) if total_naming[model] > 0 else 0.0
                self.sensitivity_by_models[model][name]["spacing"] = (count_spacing[model] / total_spacing[model] * 100) if total_spacing[model] > 0 else 0.0
            
        with open(sensitivity_by_models_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            header = ["model"] + [f"{name}_{col}" for name in all_names for col in all_columns]
            writer.writerow(header)

            for model in self.sensitivity_by_models:
                row = [model] + [self.sensitivity_by_models[model][name][col] for name in all_names for col in all_columns]
                writer.writerow(row)

        print(f"Diff pct by models CSV file saved to: {output_dir}")


    def target_output(self, filter_criteria: FilterCriteria):
        """Output target output to JSON file."""

        all_filtered = self.filtered_fragment_changed + self.filtered_fragment_unchanged
        target_output = []

        for data_point in all_filtered:
            if data_point.get('fragment_changed_types', 'remained') == 'remained':
                continue
            if data_point.get('baseline_passed', False) and not data_point.get('new_passed', False):
                target_output.append(data_point)
            elif not data_point.get('baseline_passed', False) and data_point.get('new_passed', False):
                target_output.append(data_point)

        output_dir = "./data/output/llama_example"
        os.makedirs(output_dir, exist_ok=True)
        if filter_criteria.mode == "multi_token_identifiers":
            if filter_criteria.lang == "python":
                variant = "snake_case-" + filter_criteria.variant
            elif filter_criteria.lang == "java":
                variant = "camel_case-" + filter_criteria.variant
        else:
            variant = filter_criteria.variant
        

        output_path = os.path.join(output_dir, f"{variant}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(target_output, f, indent=4, ensure_ascii=False)
        print(f"Target output saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    from .config_iterator import ConfigIterator

    parser = argparse.ArgumentParser(description="Evaluate TokenDrift results")
    parser.add_argument("--task", help="Process a specific task name", default=None)
    parser.add_argument("--diff", action="store_true", help="Process diff results", default=False)
    parser.add_argument("--spacing_transformations", action="store_true", help="Output combined transformations", default=False)
    parser.add_argument("--wilcoxon_test", action="store_true", help="Output wilcoxon signed-rank test", default=False)
    parser.add_argument("--example", action="store_true", help="Output target output", default=False)
    args = parser.parse_args()

    config = Config()
    if args.task:
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
    
    evaluator = ResultEvaluator(config)
    iterator = ConfigIterator(config)

    for task, model, subtask, tokenizer in iterator.iterate_all(
            all_tasks=True,
            all_models=True,
            all_multi_token_identifiers=True,
            all_combined_token_operators=True
    ):

        evaluator.load_data_points()

    print(f"Total data points: {len(evaluator.fragment_changed) + len(evaluator.fragment_unchanged)}")
    print(f"Fragment changed: {len(evaluator.fragment_changed)}")
    print(f"  - Fragment changed (merging only): {len(evaluator.fragment_changed_merging)}")
    print(f"  - Fragment changed (spliting only): {len(evaluator.fragment_changed_spliting)}")
    print(f"  - Fragment changed (mixing): {len(evaluator.fragment_changed_mixing)}")
    print(f"Fragment unchanged: {len(evaluator.fragment_unchanged)}")

    if args.diff:
        evaluator.get_sensitivity()
        evaluator.output_sensitivity()
        exit(0)

    if args.edit_len:
        evaluator.get_transformation_length()
        evaluator.output_transformation_length()
        exit(0)

    if args.example:
        all_columns, python_columns, java_columns = config.get_all_variants()
        for column in all_columns:
            if "-" in column:
                for lang in ["python", "java"]:
                    column = column.split("-")[-1]
                    if column == "camel_case" and lang == "java":
                        continue
                    if column == "snake_case" and lang == "python":
                        continue
                    filter_criteria = FilterCriteria(model="Llama-3.1-8B-Instruct", mode="multi_token_identifiers", variant=column, lang=lang)
                    filter_out_criteria = FilterCriteria(task="none")
                    evaluator.filter_data_points(filter_criteria, filter_out_criteria)
                    evaluator.target_output(filter_criteria)
            else:
                filter_criteria = FilterCriteria(model="Llama-3.1-8B-Instruct", mode="combined_token_operators", variant=column)
                filter_out_criteria = FilterCriteria(task="none")
                evaluator.filter_data_points(filter_criteria, filter_out_criteria)
                evaluator.target_output(filter_criteria)
        exit(0)

    
    # Small vs. Medium vs. Large models
    if args.wilcoxon_test:
        evaluator.get_sensitivity()
        for test_name in ["small-large", "small-medium", "medium-large"]:
            for name in ["Llama-3", "Qwen2.5-Coder", "deepseek-coder"]:
                if test_name == "small-medium":
                    model_s = [model for model in config.small_models if name in model][0].split("/")[1]
                    model_l = [model for model in config.medium_models if name in model][0].split("/")[1]
                elif test_name == "small-large":
                    model_s = [model for model in config.small_models if name in model][0].split("/")[1]
                    model_l = [model for model in config.large_models if name in model][0].split("/")[1]
                elif test_name == "medium-large":
                    model_s = [model for model in config.medium_models if name in model][0].split("/")[1]
                    model_l = [model for model in config.large_models if name in model][0].split("/")[1]
                evaluator.get_wilcoxon_test_p(model_s, model_l, name)
            evaluator.output_wilcoxon_test_p(test_name=test_name)
        exit(0)


    if args.spacing_transformations:
        for name in ["Llama-3", "Qwen2.5-Coder", "deepseek-coder", "CodeQwen1.5"]:
            for variant in config.all_spacing_variants:
                filter_criteria = FilterCriteria(mode="combined_token_operators", variant=variant, tokenizer=name)
                evaluator.filter_data_points(filter_criteria, FilterCriteria(task="none"))
                evaluator.get_combined_transformations(filter_criteria)
        exit(0)