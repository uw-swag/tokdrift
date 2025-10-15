import inspect
from pprint import pprint

from . import (humanevalpack, codenet, avatar)

TASK_REGISTRY = {
    **humanevalpack.create_all_tasks(),
    **codenet.create_all_tasks(),
    **avatar.create_all_tasks(),
}

ALL_TASKS = sorted(list(TASK_REGISTRY))

DATASET_MODEL_MAPPING = {
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3_1",
    "meta-llama/Llama-3.1-8B": "Llama-3_1",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama-3_1",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama-3_1",
    "Qwen/Qwen2.5-Coder-32B-Instruct": "Qwen2_5-Coder",
    "Qwen/Qwen2.5-Coder-7B-Instruct": "Qwen2_5-Coder",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": "Qwen2_5-Coder",
    "Qwen/Qwen2.5-Coder-7B": "Qwen2_5-Coder",
    "Qwen/CodeQwen1.5-7B-Chat": "CodeQwen1_5",
    "deepseek-ai/deepseek-coder-1.3b-instruct": "DeepSeek-Coder",
    "deepseek-ai/deepseek-coder-6.7b-instruct": "DeepSeek-Coder",
    "deepseek-ai/deepseek-coder-33b-instruct": "DeepSeek-Coder",
}

LANGUAGE_VARIANTS = ["snake_case", "pascal_case", "camel_case", "screaming_snake_case"]


# Add data_preprocessing argument to get_task()
def get_task(task_name, args=None, data_preprocessing=False, model=None):
    try:
        kwargs = {}
        if not data_preprocessing:
            if "prompt" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
                kwargs["prompt"] = args.prompt
            if "load_data_path" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
                kwargs["load_data_path"] = args.load_data_path
            if "model_name" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
                kwargs["model_name"] = DATASET_MODEL_MAPPING[args.model]
        else:
            if model:
                kwargs["model_name"] = DATASET_MODEL_MAPPING[model]
            else:
                raise ValueError("model is required for data_preprocessing")
        return TASK_REGISTRY[task_name](**kwargs)
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
