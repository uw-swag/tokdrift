import os

class Config:
    def __init__(self):

        self.model_list = [
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            "deepseek-ai/deepseek-coder-1.3b-instruct",
            "deepseek-ai/deepseek-coder-6.7b-instruct",
            "deepseek-ai/deepseek-coder-33b-instruct",
            "Qwen/CodeQwen1.5-7B-Chat",
            "Qwen/Qwen2.5-Coder-7B",
        ]

        self.tokenizer_model_list = [
            "meta-llama/Llama-3.1-8B-Instruct",
            "deepseek-ai/deepseek-coder-1.3b-instruct",
            "Qwen/CodeQwen1.5-7B-Chat",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
        ]

        self.all_tasks = [
            "humanevalexplaindescribe-python",
            "humanevalexplaindescribe-java",
            "humanevalfixtests-python",
            "humanevalfixtests-java",
            "avatartranslate-python2java",
            "avatartranslate-java2python",
            "codenettranslate-python2java",
            "codenettranslate-java2python",
        ]

        # Model Selection
        self.model = "meta-llama/Llama-3.1-8B-Instruct"
        
        # Task Selection
        self.task = "humanevalexplaindescribe-python"

        # Choose between "multi_token_identifiers" or "combined_token_operators"
        self.processing_mode = "multi_token_identifiers"  # "multi_token_identifiers" --> naming or "combined_token_operators" --> spacing

        # Filter Selection (for multi-token identifiers)
        # Choose the filter type and target type
        self.target_types = ["camel_case", "pascal_case", "snake_case", "screaming_snake_case"]
        self.filter_type = "snake_case" # Choose one at a time
        self.target_type = "camel_case" # Choose one at a time

        # Target Combinations Selection (for combined_token_operators)
        self.target_combinations = [
            # (':', 'NAME'),           # colon + name
            # (')', 'NAME'),           # rparentheses + name
            # (']', 'NAME'),           # rsquarebracket + name
            # (')', ')'),              # rparentheses + rparentheses
            # (']', ')'),              # rsquarebracket + rparentheses
            ('.', 'NAME'),           # period + name
            # ('(', 'NAME'),           # lparentheses + name
            # ('OP', '('),              # any operator + lparentheses
            # ('OP', 'ALL'),          # any operator + all
        ]

        self.config_task()
    
    def config_task(self):
        if self.model.split("/")[0] == "Qwen":
            self.prompt = "codeqwen"
        elif self.model.split("/")[0] == "meta-llama":
            self.prompt = "instruct"
        elif self.model.split("/")[0] == "deepseek-ai":
            self.prompt = "deepseek"
        else:
            raise ValueError(f"Invalid model: {self.model}")
        
        self.model_name = self.model.split("/")[-1]

        # Split the task name into dataset name and language
        self.output_dataset_name = self.task.split("-")[0]
        self.output_language = self.task.split("-")[1]

        # Add the output jsonl file name for the dataset
        if self.output_dataset_name in ["humanevalexplaindescribe", "humanevalfixtests"]:
            self.output_jsonl_file_name = "humanevalpack"
            self.lang = self.output_language
        elif self.output_dataset_name in ["codenettranslate"]:
            self.output_jsonl_file_name = "codenet"
            self.lang = self.output_language.split("2")[0]
        elif self.output_dataset_name in ["avatartranslate"]:
            self.output_jsonl_file_name = "avatar"
            self.lang = self.output_language.split("2")[0]
        else:
            raise ValueError(f"Invalid output dataset name: {self.output_dataset_name}")
        
        # Combined Token Operators Config
        # Operator symbol to name mapping
        self.operator_symbol_to_name = {
            '(': 'lparentheses',
            ')': 'rparentheses', 
            '[': 'lsquarebracket',
            ']': 'rsquarebracket',
            ':': 'colon',
            '.': 'period',
            '-': 'dash',
            ';': 'semicolon',
            '+': 'plus',
            '*': 'asterisk',
            '++': 'double_plus',

            # Special cases handled separately: 'operator' for any operator, 'name' for any name token
        }

        # Target operator-following token combinations
        if self.lang == "python":
            self.all_target_combinations = [
                [('(', 'NAME')],           # lparentheses + name
                [('(', ')')],              # lparentheses + rparentheses
                [(')', ':')],              # rparentheses + colon
                [(')', ')')],              # rparentheses + rparentheses
                [('[', 'NAME')],           # lsquarebracket + name
                [('.', 'NAME')],           # period + name
                [(']', ')')],              # rsquarebracket + rparentheses
                # Special cases
                [('OP', '[')],             # any operator + lsquarebracket
                [('OP', ']')],              # any operator + rsquarebracket
                [('OP', '-')],              # any operator + dash
                [('OP', 'NAME')],          # any operator + name
                [('OP', 'ALL')],          # any operator + all
            ]
        elif self.lang == "java":
            self.all_target_combinations = [
                [(')', ';')],           # rparentheses + semicolon
                [('.', 'NAME')],           # period + name
                [('(', ')')],           # lparentheses + rparentheses
                [('(', 'NAME')],           # lparentheses + name
                [(')', ')')],              # rparentheses + rparentheses
                [('.', '*')],              # period + asterisk
                [('++', ')')],              # double_plus + rparentheses
                [(')', '.')],              # rparentheses + period
                # Special cases
                [('OP', ';')],              # any operator + semicolon
                [('OP', '(')],              # any operator + lparentheses
                [('OP', 'NAME')],          # any operator + name
                [('OP', 'ALL')],          # any operator + all
            ]
        else:
            raise ValueError(f"Invalid language: {self.lang}")
        
        self.config_task_list()
        
        # Auto-generate combination name if only one target combination
        # If more than one combination, manually set combination_name
        # NOTE: If there are multiple target_combinations, manually set combination_name below
        if len(self.target_combinations) == 1:
            op, follow = self.target_combinations[0]
            op_name = self.operator_symbol_to_name.get(op, op.lower())
            # follow_name = follow.lower()
            follow_name = self.operator_symbol_to_name.get(follow, follow.lower())
            self.combination_name = f"{op_name}_{follow_name}"
        else:
            self.combination_name = "custom_combination"  # Change this manually for multiple combinations

        if self.processing_mode == "multi_token_identifiers":
            self.data_generator_output_dataset = f"./datasets/{self.output_jsonl_file_name}/var/data/{self.lang}-{self.target_type}/data/{self.output_jsonl_file_name}.jsonl"
            if self.output_dataset_name == "humanevalfixtests":
                self.data_generator_output_dataset = f"./datasets/{self.output_jsonl_file_name}/var/data/{self.lang}-{self.target_type}-fix/data/{self.output_jsonl_file_name}.jsonl"
        else:  # combined_token_operators
            self.data_generator_output_dataset = f"./datasets/{self.output_jsonl_file_name}/var/data/{self.lang}-{self.combination_name}/data/{self.output_jsonl_file_name}.jsonl"
            if self.output_dataset_name == "humanevalfixtests":
                self.data_generator_output_dataset = f"./datasets/{self.output_jsonl_file_name}/var/data/{self.lang}-{self.combination_name}-fix/data/{self.output_jsonl_file_name}.jsonl"
        
        if self.processing_mode == "multi_token_identifiers":
            self.result_dir = f"./data/output/{self.task}/{self.model_name}/{self.target_type}"
            self.evaluator_output_file = f"./data/output/{self.task}/{self.model_name}/{self.filter_type}-{self.target_type}.json"
            self.generations_path = os.path.join(self.result_dir, f"generations_{self.task}-{self.target_type}.json")
        else:  # combined_token_operators
            self.result_dir = f"./data/output/{self.task}/{self.model_name}/{self.combination_name}"
            self.evaluator_output_file = f"./data/output/{self.task}/{self.model_name}/{self.combination_name}.json"
            self.generations_path = os.path.join(self.result_dir, f"generations_{self.task}-{self.combination_name}.json")
        
        if self.output_dataset_name == "humanevalfixtests":
            self.generations_path = self.generations_path.replace(".json", "-fix.json")
        
        self.baseline_result_dir = f"./data/output/{self.task}/{self.model_name}/baseline"
        self.baseline_generations_path = os.path.join(self.baseline_result_dir, "generations.json")

        # Get the path from evaluator_output_file
        self.summaries_dir = os.path.dirname(self.evaluator_output_file)
    
        self.config_group()


    def config_task_list(self):

        # Determine language and target language for translation tasks
        if "2" in self.output_language:
            is_translation = True
            verification_lang = f"{self.lang}2{self.lang}"  # For verification purposes
        else:
            is_translation = False
            verification_lang = self.lang

        # Determine target types and filter type based on language
        if self.lang == "python":
            self.target_types = ["camel_case", "pascal_case", "screaming_snake_case"]
            self.filter_type = "snake_case"
        elif self.lang == "java":
            self.target_types = ["snake_case", "pascal_case", "screaming_snake_case"]
            self.filter_type = "camel_case"
        else:
            raise ValueError(f"Unsupported source language: {self.lang}")

        # Generate base variants (target types + operator combinations)
        variants = self.target_types.copy()

        # Add operator combination variants based on language
        if self.lang == "python":
            self.operator_variants = [
                "lparentheses_name",           # ('(', 'NAME')
                "lparentheses_rparentheses",   # ('(', ')')
                "rparentheses_colon",          # (')', ':')
                "rparentheses_rparentheses",   # (')', ')')
                "lsquarebracket_name",         # ('[', 'NAME')
                "period_name",                  # ('.', 'NAME')
                "rsquarebracket_rparentheses", # (']', ')')
                "op_lsquarebracket",           # ('OP', '[')
                "op_rsquarebracket",           # ('OP', ']')
                "op_dash",                     # ('OP', '-')
                "op_name",                     # ('OP', 'NAME')
                "op_all"                       # ('OP', 'ALL')
            ]
        elif self.lang == "java":
            self.operator_variants = [
                "rparentheses_semicolon",      # (')', ';')
                "period_name",                  # ('.', 'NAME')
                "lparentheses_rparentheses",   # ('(', ')')
                "lparentheses_name",           # ('(', 'NAME')
                "rparentheses_rparentheses",   # (')', ')')
                "period_asterisk",             # ('.', '*')
                "double_plus_rparentheses",    # ('++', ')')
                "rparentheses_period",         # (')', '.')
                "op_semicolon",                # ('OP', ';')
                "op_lparentheses",             # ('OP', '(')
                "op_name",                     # ('OP', 'NAME')
                "op_all"                       # ('OP', 'ALL')
            ]

        variants.extend(self.operator_variants)

        # Generate task list
        self.task_list = []
        for variant in variants:
            if is_translation:
                # For translation tasks, use source2source for verification
                task_name = f"{self.output_dataset_name}-{verification_lang}-{variant}"
            else:
                task_name = f"{self.output_dataset_name}-{self.lang}-{variant}"

            # Add "-fix" suffix for humanevalfixtests
            if self.output_dataset_name == "humanevalfixtests":
                task_name += "-fix"

            self.task_list.append(task_name)
    

    def config_group(self):
        self.small_models = [
            "meta-llama/Llama-3.2-3B-Instruct",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "deepseek-ai/deepseek-coder-1.3b-instruct",
        ]

        self.medium_models = [
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "deepseek-ai/deepseek-coder-6.7b-instruct",
            "Qwen/CodeQwen1.5-7B-Chat",
        ]

        self.large_models = [
            "meta-llama/Llama-3.3-70B-Instruct",
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            "deepseek-ai/deepseek-coder-33b-instruct",
        ]

        
        self.llama_models = [model.split("/")[-1] for model in self.model_list if "Llama-3" in model]
        self.qwen_models = [model.split("/")[-1] for model in self.model_list if "Qwen2.5" in model]
        self.deepseek_models = [model.split("/")[-1] for model in self.model_list if "deepseek-coder" in model]
        self.special_group_models = [
            "CodeQwen1.5-7B-Chat",
            "Qwen2.5-Coder-7B-Instruct",
            "Qwen2.5-Coder-7B",
        ]

        self.grouping_models = {
            "Llama-3": self.llama_models,
            "Qwen2.5": self.qwen_models,
            "Deepseek": self.deepseek_models,
            "special": self.special_group_models
        }

        self.group_op_1 = [
            "op_lparentheses", # 1.58
            "op_dash", # 1.63
            "rsquarebracket_rparentheses", # 1.72
            "period_asterisk", # 1.75
        ]

        self.group_op_2 = [
            "op_rsquarebracket", # 1.89
            "rparentheses_period", # 1.97
            "double_plus_rparentheses", # 2.16
            "rparentheses_colon", # 2.47
        ]

        self.group_op_3 = [
            "rparentheses_rparentheses", # 2.69
            "op_lsquarebracket", # 2.81
            "lsquarebracket_name", # 3.91
            "lparentheses_rparentheses", # 4.47
        ]

        self.group_op_4 = [
            "rparentheses_semicolon", # 7.45
            "lparentheses_name", # 8.37
            "period_name", # 8.70
            "op_semicolon", # 9.22
        ]

        self.group_op_5 = [
            "op_name", # 18.74
            "op_all", # 34.31
        ]

        self.all_spacing_variants = self.group_op_1 + self.group_op_2 + self.group_op_3 + self.group_op_4 + self.group_op_5




    def get_all_variants(self):
        current_task = self.task

        self.task = "humanevalexplaindescribe-python"
        self.config_task()
        python_variants = [f"{self.filter_type}-{tt}" for tt in self.target_types] + self.operator_variants

        self.task = "humanevalexplaindescribe-java"
        self.config_task()
        java_variants = [f"{self.filter_type}-{tt}" for tt in self.target_types] + self.operator_variants

        all_variants = list(set(python_variants + java_variants))

        self.task = current_task
        self.config_task()

        return all_variants, python_variants, java_variants


    def config_experiment(self, task, model):
        if task.split("-")[0] == "humanevalexplainsynthesize":
            task = "-".join(["humanevalexplaindescribe"] + task.split("-")[1:])
        
        if len(task.split("-")) == 2:
            self.task = task
            self.model = model
        else:
            self.task = "-".join(task.split("-")[:2])
            self.model = model
            self.config_task()
            variant = task.split("-")[2]
            if variant in self.target_types:
                self.target_type = variant
                self.processing_mode = "multi_token_identifiers"
            else:
                self.name_to_combination = {
                    "lparentheses_name": [('(', 'NAME')],
                    "lparentheses_rparentheses": [('(', ')')],
                    "rparentheses_colon": [(')', ':')],
                    "rparentheses_rparentheses": [(')', ')')],
                    "lsquarebracket_name": [('[', 'NAME')],
                    "period_name": [('.', 'NAME')],
                    "rsquarebracket_rparentheses": [(']', ')')],
                    "rparentheses_semicolon": [(')', ';')],
                    "period_asterisk": [('.', '*')],
                    "double_plus_rparentheses": [('++', ')')],
                    "rparentheses_period": [(')', '.')],
                    "op_lsquarebracket": [('OP', '[')],
                    "op_rsquarebracket": [('OP', ']')],
                    "op_semicolon": [('OP', ';')],
                    "op_dash": [('OP', '-')],
                    "op_lparentheses": [('OP', '(')],
                    "op_name": [('OP', 'NAME')],
                    "op_all": [('OP', 'ALL')],
                }
                self.target_combinations = self.name_to_combination[variant]
                self.processing_mode = "combined_token_operators"
            self.config_task()


if __name__ == "__main__":
    config = Config()
    config.config_experiment("humanevalexplainsynthesize-python", "meta-llama/Llama-3.1-8B-Instruct")
    print(config.task)
    print(config.model)
    print(config.target_type)
    print(config.filter_type)
    print(config.combination_name)
    print(config.target_combinations)
