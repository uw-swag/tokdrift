import json
import re
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, List

from evaluate import load
from ..base import Task

LANGUAGE_TO_NAME = {
    "python": "Python",
    "cpp": "C++", 
    "js": "JavaScript",
    "java": "Java",
    "go": "Go",
    "rust": "Rust",
}

LANGUAGE_TO_EXTENSION = {
    "python": "py",
    "cpp": "cpp",
    "js": "js", 
    "java": "java",
    "go": "go",
    "rust": "rs",
}

# Code execution timeouts per language
LANGUAGE_TO_TIMEOUT = {
    "python": 10,
    "cpp": 60,
    "js": 10,
    "java": 10,
    "go": 20,
    "rust": 300,
}

# Source languages supported for translation
SOURCE_LANGUAGES = ["python", "java"]

# Target languages for translation
TARGET_LANGUAGES = ["python", "java", "cpp"]

# Naming convention variants
COMBINED_TOKEN_VARIANTS = ["lparentheses_name", "lparentheses_rparentheses", "rparentheses_colon", "rparentheses_rparentheses", "lsquarebracket_name", "period_name", "rsquarebracket_rparentheses", "rparentheses_semicolon", "period_asterisk", "double_plus_rparentheses", "rparentheses_period", "op_lsquarebracket", "op_rsquarebracket", "op_semicolon", "op_dash", "op_lparentheses", "op_name", "op_all"]
NAMING_CONVENTION_VARIANTS = ["snake_case", "pascal_case", "camel_case", "screaming_snake_case"]

ALL_VARIANTS = NAMING_CONVENTION_VARIANTS + COMBINED_TOKEN_VARIANTS

# 
TARGET_LANGUAGES = TARGET_LANGUAGES + [f"{target}-{variant}" for target in TARGET_LANGUAGES for variant in ALL_VARIANTS]


# Simple code extraction regex for markdown code blocks
CODE_EXTRACTION_REGEX = re.compile(
    r"```(?:python|cpp|c\+\+|javascript|js|java|go|rust|c|script)?\s*(.*?)```", 
    re.DOTALL | re.IGNORECASE
)


def create_all_tasks():
    translate = {f"avatartranslate-{source}2{target}": create_task("translate", source, target) for source in SOURCE_LANGUAGES for target in TARGET_LANGUAGES}

    return {**translate}

def create_task(name, source_lang, target_lang):
    class AvatarTranslation(AvatarTranslationBase):
        def __init__(self, source_lang=source_lang, target_lang=target_lang, prompt="instruct", model_name=None):
            super().__init__(source_lang=source_lang, target_lang=target_lang, prompt=prompt, model_name=model_name)
    
    if name == "translate":
        return AvatarTranslation
    else:
        raise ValueError(f"Invalid task name: {name}")


class Avatar(Task):
    """Base class for Avatar tasks"""

    DATASET_PATH = "./datasets/avatar"
    DATASET_NAME = None

    # FIXME: Not every task has target_lang
    def __init__(self, source_lang, target_lang, prompt="instruct", model_name=None):
        self.source_lang = source_lang
        self.prompt = prompt
        self.model_name = model_name
        
        if '-' in target_lang:
            # For token changed tasks
            self.target_lang = target_lang.split('-')[0]
            self.variant = target_lang.split('-')[1]

            self.DATASET_PATH = f"./{self.DATASET_PATH}/var"
            self.DATASET_NAME = f"{self.source_lang}-{self.variant}"
        else:
            # For original tasks
            self.target_lang = target_lang
            self.variant = None

            self.DATASET_PATH = f"./{self.DATASET_PATH}/base"
            self.DATASET_NAME = self.source_lang

        stop_words = ["<|endoftext|>", "```", "\n```"]
        # if target_lang == "python":
        #     # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L164
        #     stop_words.extend(["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\nassert"])
        
        super().__init__(stop_words=stop_words, requires_execution=True)
    
    def get_dataset(self):
        return self.dataset["test"]

    def get_reference(self, doc):
        """Get test cases for Avatar dataset"""
        return {
            "test_inputs": doc["test_IO"]["input"],
            "expected_outputs": doc["test_IO"]["output"]
        }


class AvatarGenerative(Avatar):
    """Base class for Avatar generative tasks"""

    def process_results(self, generations, references, logs_dir):
        """Evaluate translations using execution-based testing"""
        total_samples = len(generations)
        correct_translations = 0
        test_results = []
        
        for i, (gen_list, ref) in enumerate(zip(generations, references)):
            print(f"Processing {i} of {total_samples} samples")
            
            # Take first generation for each sample
            if not gen_list:
                test_results.append({"passed": False, "error": "No generation"})
                continue
                
            generated_code = gen_list[0] if isinstance(gen_list, list) else gen_list
            
            # Test the generated code against all test cases
            test_passed = self._test_code_execution(
                i,
                generated_code, 
                ref["test_inputs"], 
                ref["expected_outputs"]
            )
            
            test_results.append(test_passed)
            if test_passed["passed"]:
                correct_translations += 1

        # Calculate Computational Accuracy
        ca_at_1 = correct_translations / total_samples if total_samples > 0 else 0.0
        
        results = {
            "ca@1": ca_at_1,
            "total_samples": total_samples,
            "correct_translations": correct_translations,
        }
        
        # Write detailed logs
        logs_path = os.path.join(logs_dir, "logs.json")
        with open(logs_path, "w") as f:
            json.dump(test_results, f, indent=4, ensure_ascii=False)
            
        return results
    
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
            baseline_log = baseline_logs[idx]
            variant_log = variant_logs[i]

            result = {
                'idx': f'{self.source_lang}2{self.target_lang}/{idx}',
                'task_id': variant_ids[i],
                'baseline_passed': baseline_log['passed'],
                'baseline_result': f'{baseline_log["passed_tests"]}/{baseline_log["total_tests"]}',
                'new_passed': variant_log['passed'],
                'new_result': f'{variant_log["passed_tests"]}/{variant_log["total_tests"]}',
            }

            if baseline_log['passed'] and not variant_log['passed']:
                result['new_errors'] = variant_log['errors']
            elif not baseline_log['passed'] and variant_log['passed']:
                result['baseline_errors'] = baseline_log['errors']

            results.append(result)

        return results

    def _test_code_execution(self, idx, code, test_inputs, expected_outputs):
        """Test generated code against input/output test cases"""
        if not code.strip():
            return {"idx": idx, "passed": False, "error": "Empty code generation"}
            
        passed_tests = 0
        total_tests = len(test_inputs)
        errors = []
        
        for i, (test_input, expected_output) in enumerate(zip(test_inputs, expected_outputs)):
            try:
                actual_output = self._execute_code(code, test_input)
                if actual_output.strip() == expected_output.strip():
                    passed_tests += 1
                else:
                    errors.append(f"Test {i+1}: Expected '{expected_output.strip()}', got '{actual_output.strip()}'")
            except Exception as e:
                errors.append(f"Test {i+1}: Execution error: {str(e)}")
                
        return {
            "idx": idx,
            "passed": passed_tests == total_tests and total_tests > 0,
            "passed_tests": passed_tests,
            "total_tests": total_tests, 
            "errors": errors
        }

    def _execute_code(self, code, stdin_input):
        """Execute code with given input and return output"""
        timeout = LANGUAGE_TO_TIMEOUT[self.target_lang]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            if self.target_lang == "python":
                return self._execute_python(code, stdin_input, temp_dir, timeout)
            elif self.target_lang == "cpp":
                return self._execute_cpp(code, stdin_input, temp_dir, timeout)
            elif self.target_lang == "js":
                return self._execute_js(code, stdin_input, temp_dir, timeout)
            elif self.target_lang == "java":
                return self._execute_java(code, stdin_input, temp_dir, timeout)
            elif self.target_lang == "go":
                return self._execute_go(code, stdin_input, temp_dir, timeout)
            elif self.target_lang == "rust":
                return self._execute_rust(code, stdin_input, temp_dir, timeout)
            else:
                raise ValueError(f"Unsupported target language: {self.target_lang}")

    def _execute_python(self, code, stdin_input, temp_dir, timeout):
        """Execute Python code"""

        # Using uv virtual environment
        venv_path = Path("./venv") / "python3_8"
        python_executable = venv_path / "bin" / "python"

        code_file = Path(temp_dir) / "solution.py"
        with open(code_file, 'w') as f:
            f.write(code)
        
        result = subprocess.run(
            [str(python_executable), str(code_file)],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Python execution failed: {result.stderr}")
        return result.stdout

    def _execute_cpp(self, code, stdin_input, temp_dir, timeout):
        """Execute C++ code"""
        code_file = Path(temp_dir) / "solution.cpp"
        exe_file = Path(temp_dir) / "solution"
        
        with open(code_file, 'w') as f:
            f.write(code)
            
        # Compile
        compile_result = subprocess.run(
            ["g++", "-o", str(exe_file), str(code_file)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if compile_result.returncode != 0:
            raise RuntimeError(f"C++ compilation failed: {compile_result.stderr}")
            
        # Execute
        result = subprocess.run(
            [str(exe_file)],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"C++ execution failed: {result.stderr}")
        return result.stdout

    def _execute_js(self, code, stdin_input, temp_dir, timeout):
        """Execute JavaScript code"""
        js_wrapper = f"""
const readline = require('readline');
const fs = require('fs');

const input = `{stdin_input}`;
const lines = input.trim().split('\\n');
let lineIndex = 0;

function readline() {{
    if (lineIndex < lines.length) {{
        return lines[lineIndex++];
    }}
    return null;
}}

{code}
"""
        
        code_file = Path(temp_dir) / "solution.js"
        with open(code_file, 'w') as f:
            f.write(js_wrapper)
            
        result = subprocess.run(
            ["node", str(code_file)],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"JavaScript execution failed: {result.stderr}")
        return result.stdout

    def _execute_java(self, code, stdin_input, temp_dir, timeout):
        """Execute Java code"""

        # source_lang == target_lang for data verification
        if self.source_lang == self.target_lang:
            class_name = "SampleSolution"
        else:
            class_name = "Main"
        
        code_file = Path(temp_dir) / f"{class_name}.java"
        with open(code_file, 'w') as f:
            f.write(code)
            
        # Compile
        compile_result = subprocess.run(
            ["javac", str(code_file)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if compile_result.returncode != 0:
            raise RuntimeError(f"Java compilation failed: {compile_result.stderr}")
            
        # Execute
        result = subprocess.run(
            ["java", "-cp", str(temp_dir), class_name],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Java execution failed: {result.stderr}")
        return result.stdout

    def _execute_go(self, code, stdin_input, temp_dir, timeout):
        """Execute Go code"""
        code_file = Path(temp_dir) / "main.go"
        with open(code_file, 'w') as f:
            f.write(code)
            
        result = subprocess.run(
            ["go", "run", str(code_file)],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=temp_dir
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Go execution failed: {result.stderr}")
        return result.stdout

    def _execute_rust(self, code, stdin_input, temp_dir, timeout):
        """Execute Rust code"""
        code_file = Path(temp_dir) / "main.rs"
        with open(code_file, 'w') as f:
            f.write(code)
            
        result = subprocess.run(
            ["rustc", str(code_file), "-o", str(Path(temp_dir) / "main")],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Rust compilation failed: {result.stderr}")
            
        exe_result = subprocess.run(
            [str(Path(temp_dir) / "main")],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if exe_result.returncode != 0:
            raise RuntimeError(f"Rust execution failed: {exe_result.stderr}")
        return exe_result.stdout



class AvatarTranslationBase(AvatarGenerative):
    """Base class for Avatar translation tasks"""

    def get_prompt(self, doc):
        source_lang_name = LANGUAGE_TO_NAME[self.source_lang]
        target_lang_name = LANGUAGE_TO_NAME[self.target_lang]

        instruction = (f"You are a skilled software developer proficient in multiple programming languages. "
                       "Your task is to re-write the input source code."
                      f"Below is the input source code written in {source_lang_name} that you should re-write into {target_lang_name} programming language. "
                      f"You must respond with the {target_lang_name} output code only, without any explanations.")
        
        if self.variant:
            source_code = doc["modified_context"]
        else:
            source_code = doc["code"]
        
        prompt_base = f"{source_lang_name} code:\n```{self.source_lang}\n{source_code}\n```\n\n{target_lang_name} code:\n```{self.target_lang}\n"
        
        if self.prompt == "instruct":
            prompt = f"{instruction}\n\n{prompt_base}"
        elif self.prompt == "deepseek":
            prompt = f"You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n### Instruction:\n{instruction}\n### Response:\n{prompt_base}"
        elif self.prompt == "codeqwen":
            prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{prompt_base}"
        else:
            raise ValueError(f"The --prompt argument {self.prompt} wasn't provided or isn't supported")
        
        return prompt.strip()

    def postprocess_generation(self, generation, idx):
        """Extract code from generation and clean it up"""
        doc = self.get_dataset()[idx]
        prompt = self.get_prompt(doc)
        
        # Remove prompt from generation
        gen_text = generation[len(prompt):].strip()
        
        extracted_code = gen_text
        for stop_word in self.stop_words:
            if stop_word in extracted_code:
                extracted_code = extracted_code[:extracted_code.find(stop_word)]
        
        return extracted_code.strip()
    
    def get_context_only(self, doc):
        if self.variant:
            return doc["modified_context"]
        else:
            return doc["code"]
        
    def get_task_id(self, doc):
        return doc["id"]
    
    def get_immutable_identifiers(self, doc):


        immutable_identifiers = set()
        if self.source_lang == "java":
            immutable_identifiers.add("SampleSolution")
        return immutable_identifiers