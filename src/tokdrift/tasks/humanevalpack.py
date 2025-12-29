import json
import re
import os

from evaluate import load
from ..base import Task

_CITATION = """
@article{muennighoff2023octopack,
      title={OctoPack: Instruction Tuning Code Large Language Models}, 
      author={Niklas Muennighoff and Qian Liu and Armel Zebaze and Qinkai Zheng and Binyuan Hui and Terry Yue Zhuo and Swayam Singh and Xiangru Tang and Leandro von Werra and Shayne Longpre},
      journal={arXiv preprint arXiv:2308.07124},
      year={2023}
}
"""

LANGUAGES = ["python", "cpp", "js", "java", "go", "rust"]

# Add variants for all languages
LANGUAGE_VARIANTS = ["snake_case", "pascal_case", "camel_case", "screaming_snake_case"]
COMBINED_TOKEN_VARIANTS = ["lparentheses_name", "lparentheses_rparentheses", "rparentheses_colon", "rparentheses_rparentheses", "lsquarebracket_name", "period_name", "rsquarebracket_rparentheses", "rparentheses_semicolon", "period_asterisk", "double_plus_rparentheses", "rparentheses_period", "op_lsquarebracket", "op_rsquarebracket", "op_semicolon", "op_dash", "op_lparentheses", "op_name", "op_all"]
FIX_VARIANTS = ["snake_case-fix", "pascal_case-fix", "camel_case-fix", "screaming_snake_case-fix"] + [f"{variant}-fix" for variant in COMBINED_TOKEN_VARIANTS]
ALL_LANGUAGES = LANGUAGES + [f"{lang}-{variant}" for lang in LANGUAGES for variant in LANGUAGE_VARIANTS] + [f"{lang}-{variant}" for lang in LANGUAGES for variant in COMBINED_TOKEN_VARIANTS] + [f"{lang}-{variant}" for lang in LANGUAGES for variant in FIX_VARIANTS]

LANGUAGE_TO_NAME = {
    "python": "Python",
    "cpp": "C++",
    "js": "JavaScript",
    "java": "Java",
    "go": "Go",
    "rust": "Rust",
}

# Add variant names
for lang in LANGUAGES:
    for variant in LANGUAGE_VARIANTS:
        variant_name = variant.replace("_", " ").title()
        LANGUAGE_TO_NAME[f"{lang}-{variant}"] = f"{LANGUAGE_TO_NAME[lang]} ({variant_name})"

# Add combined token variant names for all languages
for lang in LANGUAGES:
    for variant in COMBINED_TOKEN_VARIANTS:
        variant_name = variant.replace("_", " + ").title()
        LANGUAGE_TO_NAME[f"{lang}-{variant}"] = f"{LANGUAGE_TO_NAME[lang]} ({variant_name})"

# Add fix variant names for all languages
for lang in LANGUAGES:
    for variant in FIX_VARIANTS:
        variant_name = variant.replace("_", " + ").replace("-fix", " - Fix").title()
        LANGUAGE_TO_NAME[f"{lang}-{variant}"] = f"{LANGUAGE_TO_NAME[lang]} ({variant_name})"

LANGUAGE_TO_EXTENSION = {
    "python": "py",
    "cpp": "cpp",
    "js": "js",
    "java": "java",
    "go": "go",
    "rust": "rs",
}

# Add variant extensions (same as base language)
for lang in LANGUAGES:
    for variant in LANGUAGE_VARIANTS:
        LANGUAGE_TO_EXTENSION[f"{lang}-{variant}"] = LANGUAGE_TO_EXTENSION[lang]

# Add combined token variant extensions for all languages
for lang in LANGUAGES:
    for variant in COMBINED_TOKEN_VARIANTS:
        LANGUAGE_TO_EXTENSION[f"{lang}-{variant}"] = LANGUAGE_TO_EXTENSION[lang]

# Add fix variant extensions for all languages
for lang in LANGUAGES:
    for variant in FIX_VARIANTS:
        LANGUAGE_TO_EXTENSION[f"{lang}-{variant}"] = LANGUAGE_TO_EXTENSION[lang]

# Taken from https://huggingface.co/datasets/nuprl/MultiPL-E/ & https://github.com/THUDM/CodeGeeX
LANGUAGE_TO_STOP_WORDS = {
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L164
    "python": ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\nassert"],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L185
    "cpp": [],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L188
    "js": [],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L177
    "go": ["\n//", "\nfunc main(", "struct", "\nfunc"],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L169
    "java": [],
    "rust": [],
}

# Add variant stop words (same as base language)
for lang in LANGUAGES:
    for variant in LANGUAGE_VARIANTS:
        LANGUAGE_TO_STOP_WORDS[f"{lang}-{variant}"] = LANGUAGE_TO_STOP_WORDS[lang]

# Add combined token variant stop words for all languages
for lang in LANGUAGES:
    for variant in COMBINED_TOKEN_VARIANTS:
        LANGUAGE_TO_STOP_WORDS[f"{lang}-{variant}"] = LANGUAGE_TO_STOP_WORDS[lang]

# Add fix variant stop words for all languages
for lang in LANGUAGES:
    for variant in FIX_VARIANTS:
        LANGUAGE_TO_STOP_WORDS[f"{lang}-{variant}"] = LANGUAGE_TO_STOP_WORDS[lang]

LANGUAGE_TO_TIMEOUT = {
    "python": 20,
    "cpp": 60,
    "js": 10,
    "java": 10,
    "go": 20,
    "rust": 300, # Necessary for first-time compilation of cargo
}

# Add variant timeouts (same as base language)
for lang in LANGUAGES:
    for variant in LANGUAGE_VARIANTS:
        LANGUAGE_TO_TIMEOUT[f"{lang}-{variant}"] = LANGUAGE_TO_TIMEOUT[lang]

# Add combined token variant timeouts for all languages
for lang in LANGUAGES:
    for variant in COMBINED_TOKEN_VARIANTS:
        LANGUAGE_TO_TIMEOUT[f"{lang}-{variant}"] = LANGUAGE_TO_TIMEOUT[lang]

# Add fix variant timeouts for all languages
for lang in LANGUAGES:
    for variant in FIX_VARIANTS:
        LANGUAGE_TO_TIMEOUT[f"{lang}-{variant}"] = LANGUAGE_TO_TIMEOUT[lang]

# Java sometimes fails with more workers; For JS it's twice as fast with 4 workers
LANGUAGE_TO_NUM_WORKERS = {
    "python": 4,
    "cpp": 4,
    "js": 4,
    "java": 1,
    "go": 4,
    "rust": 1,
}

# Add variant workers (same as base language)
for lang in LANGUAGES:
    for variant in LANGUAGE_VARIANTS:
        LANGUAGE_TO_NUM_WORKERS[f"{lang}-{variant}"] = LANGUAGE_TO_NUM_WORKERS[lang]

# Add combined token variant workers for all languages
for lang in LANGUAGES:
    for variant in COMBINED_TOKEN_VARIANTS:
        LANGUAGE_TO_NUM_WORKERS[f"{lang}-{variant}"] = LANGUAGE_TO_NUM_WORKERS[lang]

# Add fix variant workers for all languages
for lang in LANGUAGES:
    for variant in FIX_VARIANTS:
        LANGUAGE_TO_NUM_WORKERS[f"{lang}-{variant}"] = LANGUAGE_TO_NUM_WORKERS[lang]

# https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L6
IMPORT_HELPER = {
    "python": [
        "import math",
        "import re",
        "import sys",
        "import copy",
        "import datetime",
        "import itertools",
        "import collections",
        "import heapq",
        "import statistics",
        "import functools",
        "import hashlib",
        "import numpy",
        "import numpy as np",
        "import string",
        "from typing import *",
        "from collections import *",
    ],
    "go": [
        "math",
        "strings",
        "fmt",
        "strconv",
        "time",
        "bytes",
        "regexp",
        "sort",
        "math/rand",
        "crypto/md5",
    ],
    "cpp": [
        "using namespace std;",      
        "#include<stdlib.h>",
        "#include<algorithm>",
        "#include<cmath>",
        "#include<math.h>",
        "#include<numeric>",
        "#include<stdio.h>",
        "#include<vector>",
        "#include<set>",
        "#include<map>",
        "#include<queue>",
        "#include<stack>",
        "#include<list>",
        "#include<deque>",
        "#include<boost/any.hpp>",
        "#include<string>",
        "#include<climits>",
        "#include<cstring>",
        "#include<iostream>",
        "#include<sstream>",
        "#include<fstream>",
    ],
}


def create_all_tasks():
    fix = {f"humanevalfix{mode}-{language}": create_task(language, "fix" + mode) for language in ALL_LANGUAGES for mode in ["tests", "docs"]}
    explain = {f"humanevalexplain{mode}-{language}": create_task(language, "explain" + mode) for language in ALL_LANGUAGES for mode in ["describe", "synthesize"]}
    synthesize = {f"humanevalsynthesize-{language}": create_task(language, "synthesize") for language in ALL_LANGUAGES}
    return {**fix, **explain, **synthesize}

def create_task(language, name):
    class HumanEvalFixTests(HumanEvalFixBase):
        def __init__(self, language=language, prompt="instruct"):
            super().__init__(language=language, prompt=prompt, with_docs=False)
    class HumanEvalFixDocs(HumanEvalFixBase):
        def __init__(self, language=language, prompt="instruct"):            
            super().__init__(language=language, prompt=prompt, with_docs=True)
    class HumanEvalExplainDescribe(HumanEvalExplainDescribeBase):
        def __init__(self, language=language, prompt="instruct"):
            super().__init__(language=language, prompt=prompt, with_docs=False)   
    class HumanEvalExplainSynthesize(HumanEvalExplainSynthesizeBase):
        def __init__(self, language=language, prompt="instruct", load_data_path=None):
            super().__init__(language=language, prompt=prompt, with_docs=False, load_data_path=load_data_path)
    class HumanEvalSynthesize(HumanEvalSynthesizeBase):
        def __init__(self, language=language, prompt="instruct"):
            super().__init__(language=language, prompt=prompt, with_docs=True)
    
    if name == "fixtests": return HumanEvalFixTests
    elif name == "fixdocs": return HumanEvalFixDocs
    elif name == "explaindescribe": return HumanEvalExplainDescribe
    elif name == "explainsynthesize": return HumanEvalExplainSynthesize
    elif name == "synthesize": return HumanEvalSynthesize


class HumanEvalPack(Task):
    """Parent class for all HumanEvalPack tasks"""
    DATASET_PATH = "./datasets/humanevalpack"
    DATASET_NAME = None

    def __init__(self, prompt="instruct", language="python", with_docs=True):
        
        self.DATASET_NAME = language
        self.prompt = prompt
        
        # Extract base language for stop words (remove variant suffix if present)
        base_language = language.split('-')[0] if '-' in language else language
        if '-' in language:
            self.DATASET_PATH = f"./{self.DATASET_PATH}/var"
        else:
            self.DATASET_PATH = f"./{self.DATASET_PATH}/base"
        stop_words = LANGUAGE_TO_STOP_WORDS[language]
        
        if self.prompt.startswith("edit"):
            stop_words.extend([
                "<commit_before>",
                "<commit_msg>",
                "<commit_after>",
            ])
        elif self.prompt == "starchat":
            stop_words.append("<|end|>")
        elif self.prompt == "diff":
            stop_words = ["<commit_before>", "<commit_msg>", "<commit_after>"]
        elif self.prompt == "diff-carper":
            stop_words = ["<BEF>", "<MSG>", "<DFF>", "\ No newline at end of file"]          
        elif self.prompt == "issue":  
            stop_words.append("```")
        stop_words.append("<|endoftext|>")
        self.with_docs = with_docs
        super().__init__(stop_words=stop_words, requires_execution=True)

    def get_dataset(self):
        return self.dataset["test"]

    def get_prompt_base(self, doc):
        if self.with_docs: return doc["prompt"] # Already includes fn main for rust
        else:
            # Extract base language for handling rust special case
            base_language = self.DATASET_NAME.split('-')[0] if '-' in self.DATASET_NAME else self.DATASET_NAME
            declaration = doc["declaration"] if "modified_declaration" not in doc or doc["modified_declaration"] is None or doc["modified_declaration"] == "" else doc["modified_declaration"]
            if base_language == "rust":
                # See 
                # https://github.com/roG0d/CodeGeeX/blob/f66205b5f615a4eead9c26d7ec297e14738ea18d/codegeex/benchmark/evaluate_humaneval_x.py#L78
                # https://github.com/THUDM/CodeGeeX/pull/76#issuecomment-1500653190
                return "fn main(){}\n" + declaration
            else: return declaration

    def get_prompt(self, prompt_base, instruction, context=None):
        if context is None:
            inp = instruction
        # `Context first then instruction` methods
        elif self.prompt in ["continue", "instruct"]:
            inp = context + "\n" + instruction
        else:
            inp = instruction + "\n" + context
        
        if self.prompt == "continue":
            assert context is None, "The `continue` prompt should only be used for HumanEvalSynthesize. Use `instruct` for HumanEvalFix and HumanEvalExplain."
            prompt = prompt_base
        elif self.prompt == "instruct":
            prompt = inp + "\n\n" + prompt_base
        elif self.prompt == "octocoder":
            prompt = f'Question: {inp}\n\nAnswer:\n{prompt_base}'
        elif self.prompt == "octogeex":
            prompt = f'Question: {inp.strip()}\n\nAnswer:\n{prompt_base}'            
        elif self.prompt == "starchat":
            # https://hf.co/HuggingFaceH4/starchat-beta
            prompt = f'<|system|>\n<|end|>\n<|user|>\n{inp}<|end|>\n<|assistant|>\n{prompt_base}'
        elif self.prompt == "starcodercommit":
            prompt = f'<commit_before><commit_msg>{inp}<commit_after>{prompt_base}'
        elif self.prompt == "instructcodet5p":
            # https://github.com/salesforce/CodeT5/blob/main/CodeT5%2B/humaneval/generate_codet5p.py#L89
            prompt = f'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{inp}\n\n### Response:{prompt_base}'       
        elif self.prompt == "wizardcoder":
            # https://github.com/nlpxucan/WizardLM/blob/main/WizardCoder/src/humaneval_gen.py#L37
            prompt = f'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{inp}\n\n### Response:\n{prompt_base}'
        elif self.prompt == "codellama":
            # https://hf.co/codellama             
            prompt = f"[INST] {inp.strip()} [/INST] {prompt_base}"
        elif  self.prompt == "deepseek":
            prompt = f"You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n### Instruction:\n{inp.strip()}\n### Response:\n{prompt_base}"
        elif self.prompt in ["tulu", "gritlm"]:
            # https://hf.co/GritLM/GritLM-7B
            prompt = f"<|user|>\n{inp}\n<|assistant|>\n{prompt_base}"
        elif self.prompt == "zephyr":
            # https://hf.co/HuggingFaceH4/zephyr-7b-beta
            prompt = f"<|user|>\n{inp}</s>\n<|assistant|>\n{prompt_base}"
        elif self.prompt in ["yi", "starchat2", "codeqwen"]:
            # https://hf.co/01-ai/Yi-34B-Chat     
            prompt = f"<|im_start|>user\n{inp}<|im_end|>\n<|im_start|>assistant\n{prompt_base}"
            # Start: tokenizer.bos_token
            # End: tokenizer.eos_token
        elif self.prompt == "codegemma":
            prompt = f"<start_of_turn>user\n{inp}<end_of_turn>\n<start_of_turn>model\n{prompt_base}"
        elif self.prompt == "codellama-70b":
            prompt = f"Source: user\n\n {inp.strip()} Source: assistant\nDestination: user \n\n{prompt_base}"
        elif self.prompt == "aurora-m":
            prompt = f'### Instruction:\n{inp}\n### Response:\n{prompt_base}'
        else:
            raise ValueError(f"The --prompt argument {self.prompt} wasn't provided or isn't supported")
        # Strip off the final \n to make the tokens more natural
        # Essentially, we want to make sure that if there was no distinction between
        # input & output, the tokens would be the same
        # E.g. for SantaCoder:
        # tokenize("""def hi()\n   return""")
        # ['def', 'Ġhi', '()', 'ĊĠĠ', 'Ġreturn']
        # So we need to split before the \n so that the input is
        # ['def', 'Ġhi', '()'] and the model can generate ['ĊĠĠ', 'Ġreturn']
        # If instead we provide def hi()\n the tokens will be
        # ['def', 'Ġhi', '()', 'Ċ'] and the model would need to generate ['ĠĠ', 'Ġreturn']
        # Which would be harder, as it's not the usual way these tokens are tokenized
        # i.e. the model has never seen the token sequence of ['()', 'Ċ', 'ĠĠ'], but only ['()', 'ĊĠĠ']
        # The same holds for Java, JS, Go, Rust, C++ tho the start sequences are slightly different

        # print(prompt)
        # raise ValueError("Stop here")
        return prompt.strip()
            
    def get_reference(self, doc, get_solution=False):
        if get_solution:
            return doc["prompt"] + doc["canonical_solution"]
        else:
            if "modified_test" in doc and doc["modified_test"] is not None and doc["modified_test"] != "":
                return doc["modified_test"]
            else:
                return "\n" + doc["test"] # check(func_name) is already included
    
    def get_test_case(self, doc):
        return "\n" + doc["test"]
    
    def get_entry_point(self, doc):
        return doc["entry_point"]
    
    def get_declaration(self, doc):
        return doc["declaration"]

    def process_evaluation_logs(self, baseline_logs_path, variant_logs_path, task_idx, variant_ids):
        """Process HumanEval evaluation logs for comparison"""
        # Load logs
        with open(baseline_logs_path, 'r') as f:
            baseline_logs = json.load(f)
        with open(variant_logs_path, 'r') as f:
            variant_logs = json.load(f)
        
        # Process results
        results = []

        for i, idx in enumerate(task_idx):
            baseline_log = baseline_logs[f'{idx}'][0][1]
            variant_log = variant_logs[f'{i}'][0][1]

            result = {
                'task_id': variant_ids[i],
                'baseline_passed': baseline_log['passed'],
                'new_passed': variant_log['passed'],
            }

            if baseline_log['passed'] and not variant_log['passed']:
                result['new_result'] = variant_log['result']
            elif not baseline_log['passed'] and variant_log['passed']:
                result['baseline_result'] = baseline_log['result']

            results.append(result)

        return results

class HumanEvalPackGenerative(HumanEvalPack):
    """Parent class for all HumanEvalPack tasks except describing code"""
    def check_fn(self, code):
        """
        Checks whether the generated code is finished.
        Problem: Models rarely split their code into multiple functions, but this stops the model after the 1st function.
        Inspiration: https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L115
        """
        if any([w in code for w in self.stop_words]): return True

        # The heuristics below do not hold for diff generation
        if (self.prompt.startswith("diff")): return False

        # Extract base language for processing
        base_language = self.DATASET_NAME.split('-')[0] if '-' in self.DATASET_NAME else self.DATASET_NAME
        
        if base_language == "python":
            for line in code.split("\n"):
                if len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t':
                    return True
        else:
            open_brackets = 2 if base_language == "java" else 1
            if code.count("{") + open_brackets == code.count("}"):
                return True
        return False 

    def remove_last_block(self, code):
        """
        Adapted from https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L151
        """
        for w in self.stop_words:
            if w in code:
                code = code[:code.find(w)]

        # Extract base language for processing
        base_language = self.DATASET_NAME.split('-')[0] if '-' in self.DATASET_NAME else self.DATASET_NAME

        ### Find the first occassion where a chain of { } is closed
        if base_language == "python":
            for i, line in enumerate(code.split("\n")):
                if len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t':
                    return "\n".join(code.split("\n")[:i])
        elif base_language in ["java", "js", "go", "cpp", "rust"]:
            open_brackets = 2 if base_language == "java" else 1
            cut = False
            for i, c in enumerate(code):
                if c == '{':
                    open_brackets += 1
                elif c == '}':
                    open_brackets -= 1
                if open_brackets == 0:
                    code = code[:i+1]
                    cut = True
                    break
            if not cut:
                if base_language == "java":
                    main_pos = code.find("public static void main")
                    if main_pos != -1:
                        code = code[:main_pos] + '}'
                    if '}' in code:
                        code = code[:code.rfind('}')] + '}'
                    if code.count('{') - 1 == code.count('}'):
                        code += "\n}"
                elif '}' in code:
                    code = code[:code.rfind('}')] + '}'
        return code

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        doc = self.get_dataset()[idx]
        prompt = self.get_prompt(doc)
        gen = self.remove_last_block(generation[len(prompt):].rstrip())
        # Strip to maintain same behavior as with get_prompt
        declaration = doc["declaration"] if "modified_declaration" not in doc or doc["modified_declaration"] is None or doc["modified_declaration"] == "" else doc["modified_declaration"]
        return declaration.rstrip() + gen
        
    def process_results(self, generations, references, logs_dir):
        """Takes the list of LM generations and evaluates them against ground truth references.

        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        code_metric = load("Muennighoff/code_eval_octopack")
        timeout = LANGUAGE_TO_TIMEOUT[self.DATASET_NAME]
        num_workers = LANGUAGE_TO_NUM_WORKERS[self.DATASET_NAME]
        
        # Extract base language for processing, but handle js special case
        base_language = self.DATASET_NAME.split('-')[0] if '-' in self.DATASET_NAME else self.DATASET_NAME
        language = base_language if base_language != "js" else "javascript"

        ### CUSTOM MUTATE METHOD CHANGES ###
        if self.prompt == "diff":
            # Requires:
            # !wget https://raw.githubusercontent.com/google/diff-match-patch/master/python3/diff_match_patch.py
            from diff_match_patch import diff_match_patch
            dmp = diff_match_patch()
            ds = self.get_dataset().select(range(len(generations)))
            for gen, doc in zip(generations, ds):
                prompt_base = self.get_prompt_base(doc)
                old_code = prompt_base + doc["buggy_solution"]
                for i, diff in enumerate(gen): 
                    try:
                        # Strip away anything to the left such as \n
                        patches = dmp.patch_fromText(diff.lstrip())
                        fixed_code, _ = dmp.patch_apply(patches, old_code)
                    except Exception as e:
                        print(f"Failed with {e} when applying patch to buggy code: {diff}")
                        fixed_code = ""
                    gen[i] = fixed_code
        elif self.prompt == "diff-carper":
            from bigcode_eval.tasks.custom_metrics.diff_eval import apply_diff
            ds = self.get_dataset().select(range(len(generations)))
            for gen, doc in zip(generations, ds):
                prompt_base = self.get_prompt_base(doc)
                old_code = prompt_base + doc["buggy_solution"]
                for i, diff_hunk in enumerate(gen):
                    if not(diff_hunk):
                        gen[i] = ""
                        continue
                    res: str = apply_diff(old_code, diff_hunk)        
                    gen[i] = res

        ### CUSTOM PROG LANGUAGE CHANGES ###
        # Inspiration: https://github.com/THUDM/CodeGeeX/blob/ebeb850f227a90c79de39f7e26b1302f374f3240/codegeex/benchmark/evaluate_humaneval_x.py
        if base_language == "python":
            python_imports = "\n".join(IMPORT_HELPER["python"])
            generations = [
                [(python_imports + "\n" + g).strip() for g in gen] for gen in generations
            ]
        elif base_language == "cpp":
            cpp_imports = "\n".join(IMPORT_HELPER["cpp"])
            # Remove main in case present
            generations = [
                [(cpp_imports + "\n" + g.split("int main")[0]).strip() for g in gen] for gen in generations
            ]
        elif base_language == "java":
            generations = [
                [g.replace("public class Main {\n    }", "").strip() for g in gen] for gen in generations
            ]
        elif base_language == "go":
            ds = self.get_dataset().select(range(len(generations)))
            for gen, ref, doc in zip(generations, references, ds):
                for line in doc["import"].split("\n"):
                    line = line.replace("import", "").replace("(", "").replace(")", "").replace('"', "").strip()
                    if line: assert line in IMPORT_HELPER["go"], doc["import"] # Will be added later
                test_setup_str = doc["test_setup"] + "\n"
                for i, g in enumerate(gen):
                    for line in test_setup_str.split("\n"):
                        line = line.replace("import", "").replace("(", "").replace(")", "").strip()
                        if line.startswith('"') and line in g:
                            test_setup_str = test_setup_str.replace(line, "")
                    g = test_setup_str + g + "\n" + ref
                    other_pkgs = set()
                    for pkg in IMPORT_HELPER["go"]:
                        if ('"' + pkg + '"' not in g):
                            p = pkg.split("/")[-1]
                            # Check if the package is used
                            if (p + "." in g):
                                # The problem is that it could appear in a comment
                                # E.g. in problem 158, the docstring is:
                                # // ... a list of strings.
                                # but the "strings" pkg is never used
                                # Golang throws an error if the pkg is not used
                                # Thus search for the package & make sure it's not in a commented line
                                lines = g.split("\n")
                                for line in lines:
                                    if (p + "." in line) and not(line.strip().startswith("//")):
                                        other_pkgs.add('"' + p + '"')
                                        break
                    other_pkgs_str = ""
                    if other_pkgs:
                        other_pkgs_str = "import (\n" + "\n".join(["    " + p for p in other_pkgs]) + "\n)\n"
                    if ("package main" in gen[i]) and ("package main" in test_setup_str):
                        gen[i] = gen[i].replace("package main", "")
                    gen[i] = test_setup_str + other_pkgs_str + gen[i]
        elif base_language == "rust":
            ds = self.get_dataset().select(range(len(generations)))
            main = "fn main(){}\n"
            for gen, doc in zip(generations, ds):
                declaration = doc["declaration"]
                for i, g in enumerate(gen):
                    new_gen = ""
                    if "fn main()" not in g:
                        new_gen += main
                    for line in declaration.split("\n"):
                        if line.strip() not in g:
                            # Skip if the function is already present
                            if line.strip().startswith("fn") and (line.strip().split("(")[0]) in g:
                                continue
                            new_gen += line.strip() + "\n"
                    # If fn main() is present twice, cut off before the second one
                    g = "fn main()".join(g.split("fn main()")[0:2])
                    new_gen += g
                    gen[i] = new_gen

        ### EVALUATION ###
        results, logs = code_metric.compute(
            references=references,
            predictions=generations,
            language=language,
            timeout=timeout,
            num_workers=num_workers,
        )
        
        logs_path = os.path.join(logs_dir, "logs.json")
        # Write logs to json
        with open(logs_path, "w") as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)

        """Debugging help
        for i, (gen, ref) in enumerate(zip(generations, references)):
            import time
            starttime = time.time()            
            results, log = code_metric.compute(
                references=[ref],
                predictions=[gen],
                language=language,
                timeout=timeout,
            )
            print("Took: ", time.time() - starttime)
            with open("errors.txt", "a") as f:
                f.write(log[0][0][1]["result"] + "\n")
            if ("compilation error" in log[0][0][1]["result"]):
                print("Result")
                print(results)
                print("Log")
                print(log)
                print("Gen")
                print(gen[0])
                print("Ref")
                print(ref)
        """
        return results


class HumanEvalFixBase(HumanEvalPackGenerative):
    def get_filename_with_extension(self, input_file):
        """Returns the synthetic filename for different datasets"""
        file_name = input_file if input_file is not None else "solution"
        return file_name + "." + LANGUAGE_TO_EXTENSION[self.DATASET_NAME]
        
    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        prompt_base = self.get_prompt_base(doc)
        entry_point = doc["entry_point"] if "modified_entry_point" not in doc or doc["modified_entry_point"] is None or doc["modified_entry_point"] == "" else doc["modified_entry_point"]
        instruction = f'Fix bugs in {entry_point}.'
        base_language = self.DATASET_NAME.split('-')[0] if '-' in self.DATASET_NAME else self.DATASET_NAME
        # Context is modified_context if it exists, otherwise it is prompt_base + buggy_solution
        if "modified_context" in doc and doc["modified_context"] is not None and doc["modified_context"] != "":
            # context = "```" + base_language + "\n" + doc["modified_context"] + "\n```"
            context = doc["modified_context"]
        else:
            # context = "```" + base_language + "\n" + prompt_base + doc["buggy_solution"] + "\n```"
            context = prompt_base + doc["buggy_solution"]
        
        
        
        if self.with_docs is False: # Add tests as source of ground truth
            if "modified_test" in doc and doc["modified_test"] is not None and doc["modified_test"] != "":
                # context += "\n" + "```" + base_language + "\n" + doc["modified_test"] + "\n```"
                context += "\n" + doc["modified_test"]
            else:
                # context += "\n" + "```" + base_language + "\n" + doc["test"] + "\n```"
                context += "\n" + doc["test"]
        
        if base_language == "python":
            context = "```python\n" + context + "\n```"

        if self.prompt == "file":
            file_name = self.get_filename_with_extension(input_file=doc["entry_point"])
            prompt = f"<file_name>\n{file_name}\n<commit_before>\n{context}\n<commit_msg>\n{instruction}<commit_after>\n{prompt_base}"
        elif self.prompt == "starcodercommit":
            prompt = f"<commit_before>{context}<commit_msg>{instruction}<commit_after>{prompt_base}"
        elif self.prompt == "diff":
            prompt = f"<commit_before>{context}<commit_msg>{instruction}<commit_after>"
        elif self.prompt == "diff-carper":
            prompt = f"<NME> {self.get_filename_with_extension(input_file=doc['entry_point'])}\n"
            prompt += f"<BEF> {context}\n<MSG> {instruction}\n<DFF>"
        elif self.prompt == "issue":
            prompt = f"<issue_start>username_0: {instruction}\n\n```{context}```\nUpvotes: 100<issue_comment>username_1: Sure, here is the fixed code.\n\n```{prompt_base}"
        else:
            prompt = super().get_prompt(prompt_base, instruction, context)
        return prompt.strip()
    
    # Add a new method to get the context only
    def get_context_only(self, doc):
        """Returns the context only of the prompt"""
        prompt_base = self.get_prompt_base(doc)
        if "modified_context" in doc and doc["modified_context"] is not None and doc["modified_context"] != "":
            context = doc["modified_context"]
        else:
            context = prompt_base + doc["buggy_solution"]
        return context
    
    # Get the task id
    def get_task_id(self, doc):
        return doc["task_id"]
    
    # Get the immutable identifiers in the context
    def get_immutable_identifiers(self, doc):
        immutable_identifiers = set()
        base_language = self.DATASET_NAME.split('-')[0] if '-' in self.DATASET_NAME else self.DATASET_NAME
        if base_language == "java":
            immutable_identifiers.add("Solution")
        return immutable_identifiers

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        doc = self.get_dataset()[idx]
        prompt = self.get_prompt(doc)

        if self.prompt == "diff-carper":
            # Only remove final stopwords like <MSG>
            generation = self.remove_last_block(generation[len(prompt):].rstrip())
            generation = prompt + generation
            from bigcode_eval.tasks.custom_metrics.diff_eval import split_diff
            # From https://github.com/CarperAI/OpenELM/blob/e6402a0696096011572152334ccbe049f89c332e/src/openelm/benchmarks/benchmark_bugs.py#L93
            end_of_diff = re.compile("\n[^ +-@]+")
            parsed: dict = split_diff(generation)
            if parsed and all(
                (s in parsed for s in ["name", "file", "message", "diff"])
            ):
                # truncate diff hunk at the first line not starting with " ", "+", "-", or "@"
                diff_hunk: str = end_of_diff.split(parsed["diff"])[0]
                # We apply diff patch loosely:
                #   1. it ignores the line numbers;
                #   2. it ignores invalid lines (not starting with " ",
                #   "+" or "-" and not being "@@ ... @@").
                # https://github.com/CarperAI/OpenELM/blob/e6402a0696096011572152334ccbe049f89c332e/src/openelm/benchmarks/benchmark_bugs.py#L162
                nme_idx: int = diff_hunk.find("<NME>")
                if nme_idx != -1:
                    diff_hunk = diff_hunk[:nme_idx]
                return diff_hunk
        else:
            gen = self.remove_last_block(generation[len(prompt):].rstrip())
            if self.prompt.startswith("diff"):
                return gen
            else:
                # Strip on the right to maintain same behavior as with get_prompt
                prompt_base = self.get_prompt_base(doc)
                return prompt_base.rstrip() + gen


class HumanEvalExplainDescribeBase(HumanEvalPack):
    def get_prompt_encoder(self, doc):
        """Encoder input for models with Enc-Dec architecture like CodeT5"""
        assert self.prompt == "instructcodet5p", "Enc-Dec is only tested for InstructCodeT5+"
        prompt_base = self.get_prompt_base(doc)
        instruction = f"Provide a concise natural language description of the code using at most {len(doc['docstring'])} characters."
        context = prompt_base + doc["canonical_solution"]

        return super().get_prompt("", instruction, context) # No prompt base as not generating
    
    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        prompt_base = self.get_prompt_base(doc)
        instruction = f"Provide a concise natural language description of the code using at most {len(doc['docstring'])} characters."
        # Context is modified_context if it exists, otherwise it is prompt_base + canonical_solution
        if "modified_context" in doc and doc["modified_context"] is not None and doc["modified_context"] != "":
            context = doc["modified_context"]
        else:
            context = prompt_base + doc["canonical_solution"]
        
        base_language = self.DATASET_NAME.split('-')[0] if '-' in self.DATASET_NAME else self.DATASET_NAME
        if base_language == "python":
            context = "```python\n" + context + "\n```"
        
        return super().get_prompt("", instruction, context)

    def remove_last_block(self, text):
        for w in self.stop_words:
            if w in text:
                text = text[:text.find(w)]
        return text

    def remove_code(self, text, canonical_solution):
        for line in canonical_solution.split("\n"):
            line = line.strip()
            if len(line) > 20 and line in text:
                text = text.replace(line, "")
        return text
    
    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        doc = self.get_dataset()[idx]
        prompt = self.get_prompt(doc)
        docstring_len = len(doc["docstring"])
        gen = self.remove_last_block(generation[len(prompt):].strip()[:docstring_len]).rstrip()
        if "modified_context" not in doc or doc["modified_context"] is None or doc["modified_context"] == "":
            gen = self.remove_code(gen, doc["canonical_solution"])
        else:
            if "modified_declaration" not in doc or doc["modified_declaration"] is None or doc["modified_declaration"] == "":
                canonical_solution = doc["modified_context"][len(doc["declaration"]):]
            else:
                canonical_solution = doc["modified_context"][len(doc["modified_declaration"]):]
            gen = self.remove_code(gen, canonical_solution)
        return gen

    def get_reference(self, doc, get_solution=False):
        return None

    def process_results(self, generations, references, logs_dir):
        raise ValueError("""ExplainDescribe should be run with `--generation_only`.
        Once generations are done run ExplainSynthesize with `--load_data_path path/to/generations.json`
        It will load the explanations, generate from them and evaluate.""")

    # Add a new method to get the context only
    def get_context_only(self, doc):
        """Returns the context only of the prompt"""
        if "modified_context" in doc and doc["modified_context"] is not None and doc["modified_context"] != "":
            context = doc["modified_context"]
        else:
            prompt_base = self.get_prompt_base(doc)
            context = prompt_base + doc["canonical_solution"]
        return context
    
    # Get the task id
    def get_task_id(self, doc):
        return doc["task_id"]
    
    # Get the immutable identifiers in the context
    def get_immutable_identifiers(self, doc):
        immutable_identifiers = set()
        base_language = self.DATASET_NAME.split('-')[0] if '-' in self.DATASET_NAME else self.DATASET_NAME
        if base_language == "java":
            immutable_identifiers.add("Solution")
        return immutable_identifiers
                

class HumanEvalExplainSynthesizeBase(HumanEvalPackGenerative):
    def __init__(self, load_data_path=None, **kwargs):
        assert load_data_path is not None, "load_data_path must be specified to load the descriptions."
        if load_data_path != "none":
            with open(load_data_path) as fp:
                self.descriptions = json.load(fp)
                print(f"{len(self.descriptions)} descriptions with {len(self.descriptions[0])} description candidates loaded.")    
        else:
            self.descriptions = None

        super().__init__(**kwargs)

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        dataset = []
        if self.descriptions is None:
            return self.dataset["test"]
        for description, sample in zip(self.descriptions, self.dataset["test"]):
            for description_candidate in description:
                dataset.append({"description": description_candidate} | sample)
        return dataset

    def get_prompt_encoder(self, doc):
        """Encoder input for models with Enc-Dec architecture like CodeT5"""
        assert self.prompt == "instructcodet5p", "Enc-Dec is only tested for InstructCodeT5+"
        prompt_base = "" # No prompt base as not generating
        instruction = f"Write functional code in {LANGUAGE_TO_NAME[self.DATASET_NAME]} according to the description."
        context = doc["description"]

        return super().get_prompt(prompt_base, instruction, context)
    
    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        prompt_base = self.get_prompt_base(doc)
        instruction = f"Write functional code in {LANGUAGE_TO_NAME[self.DATASET_NAME]} according to the description."
        context = doc["description"]

        return super().get_prompt(prompt_base, instruction, context)


class HumanEvalSynthesizeBase(HumanEvalPackGenerative):
    def get_prompt_encoder(self, doc):
        """Encoder input for models with Enc-Dec architecture like CodeT5"""
        assert self.prompt == "instructcodet5p", "Enc-Dec is only tested for InstructCodeT5+"
        prompt_base = "" # No prompt base as not generating
        instruction = doc["instruction"].strip()

        return super().get_prompt(prompt_base, instruction)
        
    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        prompt_base = self.get_prompt_base(doc)
        instruction = doc["instruction"].strip()

        return super().get_prompt(prompt_base, instruction)
