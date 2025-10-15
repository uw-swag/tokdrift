from typing import Iterator, Tuple, Optional
from transformers import AutoTokenizer

from .config import Config


class ConfigIterator:
    """Centralized iterator for task, model, and processing mode combinations"""

    def __init__(self, config: Config):
        self.config = config

    def iterate_all(self,
                   all_tasks: bool = False,
                   all_models: bool = False,
                   all_multi_token_identifiers: bool = False,
                   all_combined_token_operators: bool = False,
                   tokenizer_list: bool = False) -> Iterator[Tuple[str, str, str, Optional[AutoTokenizer]]]:
        """
        Iterate through all combinations of tasks, models, and processing modes

        Args:
            all_tasks: If True, iterate through all tasks
            all_models: If True, iterate through all models
            all_multi_token_identifiers: If True, process multi-token identifiers
            all_combined_token_operators: If True, process combined token operators

        Yields:
            Tuple of (task, model, processing_mode, tokenizer)
        """
        tasks_to_process = self.config.all_tasks if all_tasks else [self.config.task]
        if tokenizer_list:
            models_to_process = self.config.tokenizer_model_list if all_models else [self.config.model]
        else:
            models_to_process = self.config.model_list if all_models else [self.config.model]

        for task in tasks_to_process:
            if not all_tasks and task != self.config.task:
                continue

            print(f"Processing task: {task}")
            self.config.task = task

            for model in models_to_process:
                if not all_models and model != self.config.model:
                    continue

                print(f"Processing model: {model}")
                self.config.model = model
                tokenizer = AutoTokenizer.from_pretrained(model)

                self.config.config_task()

                # Process multi-token identifiers
                if all_multi_token_identifiers or self.config.processing_mode == "multi_token_identifiers":
                    if all_multi_token_identifiers:
                        print("Processing all multi-token identifiers...")
                        self.config.processing_mode = "multi_token_identifiers"

                    for target_type in self.config.target_types:
                        if not all_multi_token_identifiers and target_type != self.config.target_type:
                            continue

                        self.config.target_type = target_type
                        self.config.config_task()

                        print(f"Processing multi-token identifiers with target type: {target_type}")
                        print(f"Filter type: {self.config.filter_type}")

                        yield task, model, "multi_token_identifiers", tokenizer

                # Process combined token operators
                if all_combined_token_operators or self.config.processing_mode == "combined_token_operators":
                    if all_combined_token_operators:
                        print("Processing all combined token operators...")
                        self.config.processing_mode = "combined_token_operators"

                    for target_combinations in self.config.all_target_combinations:
                        if not all_combined_token_operators and target_combinations != self.config.target_combinations:
                            continue

                        self.config.target_combinations = target_combinations
                        self.config.config_task()

                        print(f"Processing target combinations: {target_combinations}")

                        yield task, model, "combined_token_operators", tokenizer

    def iterate_subtasks(self,
                        all_tasks: bool = False,
                        all_models: bool = False) -> Iterator[Tuple[str, str, str]]:
        """
        Iterate through subtasks for verification purposes

        Args:
            all_tasks: If True, iterate through all tasks
            all_models: If True, iterate through all models

        Yields:
            Tuple of (main_task, model, subtask)
        """
        tasks_to_process = self.config.all_tasks if all_tasks else [self.config.task]

        for task in tasks_to_process:
            if not all_tasks and task != self.config.task:
                continue

            # Skip certain tasks for verification
            if task in ["humanevalfixtests-python", "humanevalfixtests-java"]:
                continue

            print(f"Processing task: {task}")
            self.config.task = task

            models_to_process = self.config.tokenizer_model_list if all_models else [self.config.model]

            for model in models_to_process:
                if not all_models and model != self.config.model:
                    continue

                print(f"Processing model: {model}")
                self.config.model = model
                self.config.config_task()

                for i, subtask in enumerate(self.config.task_list):
                    print(f"Processing subtask {i+1}/{len(self.config.task_list)}: {subtask}")
                    yield task, model, subtask