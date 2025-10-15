# !! Currently only for avatar dataset !!

import os
import json
import re
from datasets import load_dataset

dataset = load_dataset("./datasets/avatar")["train"]



def modify_code(lang: str, code_id: int) -> str:
    dataset_path = os.path.join("./data/input/datasets/avatar", lang)
    if lang == "java":
        file_path = os.path.join(dataset_path, f"{code_id}.java")
    elif lang == "python":
        file_path = os.path.join(dataset_path, f"{code_id}.py")
    
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise FileNotFoundError(f"File {file_path} not found")

def check_code(lang: str, code_id: int) -> tuple[str, bool]:
    if lang == "java":
        delete_list = [27, 74, 100, 103]
        modify_list = [14, 34, 80, 113, 117, 122, 134, 162, 169, 189, 211, 214, 229]
    elif lang == "python":
        delete_list = [13, 110, 140, 153, 159, 246]
        modify_list = [6, 25, 40, 62, 69, 91, 109, 129, 142, 163, 181, 212]
    
    if code_id in delete_list:
        return None, False
    elif code_id in modify_list:
        return modify_code(lang, code_id), True
    else:
        return None, True

def modify_input(lang: str, code_id: int) -> dict[int, str]:
    if lang == "java":
        modified_input = {
            196: {1: 'gggg'},
        }
    elif lang == "python":
        modified_input = {
            36: {1: 'gggg'},
        }
    if code_id in modified_input:
        return modified_input[code_id]
    else:
        return None

def modify_output(lang: str, code_id: int) -> dict[int, str]:
    if lang == "java":
        modified_output = {
            25: {1: '7 1 8', 3: '1 2 3'},
            81: {1: '0'},
            113: {2: '380000000.0000', 4: '1000000000.0000', 5: '48000.0000'},
            183: {22: '1'},
            190: {4: '133 550', 12: '164 826 826', 13: '4 5 5'},
        }
    elif lang == "python":
        modified_output = {
        }
    
    if code_id in modified_output:
        return modified_output[code_id]
    else:
        return None

def check_test(lang: str, code_id: int) -> tuple[dict[int, str], dict[int, str], list[int]]:
    if lang == "java":
        delete_test_list = {51: [22, 25], 95: [1], 107: [3, 9, 11, 13, 14], 110: [12, 13, 15], 123: [22], 149: [4, 6, 14, 18, 20, 23]}
        modify_test_list = [25, 81, 113, 183, 190, 196]
    elif lang == "python":
        delete_test_list = {66: [4, 6, 14, 18, 20, 23], 69: [2, 4], 91: [3, 6, 19, 22, 38, 63], 143: [1, 2, 4, 6, 7, 8, 9, 17, 20, 21, 22, 24, 25, 27, 28, 30, 36, 51, 52], 159: [4, 5, 12, 13, 14, 16, 22, 23, 24, 25, 30, 31, 32, 38, 39, 40, 50, 55, 56, 57, 59, 65, 66, 67, 74, 75], 215: [22, 25]}
        modify_test_list = [36]
    
    if code_id in modify_test_list:
        if code_id in delete_test_list:
            return modify_input(lang, code_id), modify_output(lang, code_id), delete_test_list[code_id]
        else:
            return modify_input(lang, code_id), modify_output(lang, code_id), None
    elif code_id in delete_test_list:
        return None, None, delete_test_list[code_id]
    else:
        return None, None, None

print("Length of dataset: ", len(dataset))

# Get unique languages in the dataset
language_set = set()
for sample in dataset:
    language_set.add(sample["language"])

print("Languages found:", language_set)

# Separate data by language and prepare for saving
python_data = []
java_data = []

# Process each sample
for sample in dataset:
    language = sample["language"].lower()
    
    if "python" in language:
        # Create a copy of the sample data
        python_sample = dict(sample)
        python_data.append(python_sample)
    elif "java" in language:
        # Create a copy of the sample data
        java_sample = dict(sample)
        java_data.append(java_sample)

print(f"Python samples: {len(python_data)}")
print(f"Java samples: {len(java_data)}")

# Update IDs and save Python data
if python_data:
    # Create directory structure
    python_dir = "./datasets/avatar/base/data/python/data"
    os.makedirs(python_dir, exist_ok=True)
    
    # Update IDs and save
    python_file_path = os.path.join(python_dir, "avatar.jsonl")
    with open(python_file_path, 'w', encoding='utf-8') as f:
        delete_count = 0
        for i, sample in enumerate(python_data):
            sample["id"] = f"Python/{i-delete_count}"
            
            # Check the sample code
            code, is_modified = check_code(sample["language"].lower(), i)
            if is_modified:
                if code is not None:
                    sample["code"] = code
            else:
                delete_count += 1
                continue
            
            # Check the sample test
            modified_input, modified_output, delete_test = check_test(sample["language"].lower(), i)

            if modified_input is not None:
                for test_id, input_data in modified_input.items():
                    sample["test_IO"][test_id-1]["input"] = input_data
            if modified_output is not None:
                for test_id, output_data in modified_output.items():
                    sample["test_IO"][test_id-1]["output"] = output_data
            if delete_test is not None:
                for i, test_id in enumerate(delete_test):
                    del sample["test_IO"][test_id-i-1]

            # remove the "dataset" and "language"
            del sample["dataset"]
            del sample["language"]
            f.write(json.dumps(sample) + '\n')
    
    print(f"Saved {len(python_data) - delete_count} Python samples to {python_file_path}")

# Update IDs and save Java data
if java_data:
    # Create directory structure
    java_dir = "./datasets/avatar/base/data/java/data"
    os.makedirs(java_dir, exist_ok=True)
    
    # Update IDs and save
    java_file_path = os.path.join(java_dir, "avatar.jsonl")
    with open(java_file_path, 'w', encoding='utf-8') as f:
        delete_count = 0
        for i, sample in enumerate(java_data):
            sample["id"] = f"Java/{i-delete_count}"

            # Check the sample code
            code, is_modified = check_code(sample["language"].lower(), i)
            if is_modified:
                if code is not None:
                    sample["code"] = code
                else:
                    code = sample["code"]
                    class_match = re.search(r'class\s+(\w+)', code)
                    class_name = class_match.group(1) if class_match else "Main"
                    if i == 117:
                        class_name = "atcoder_ABC137_D"
                    # Replace the class name with the new class name
                    count = code.count("SampleSolution")
                    if count > 0:
                        print(f"Solution found in code {i}")
                    sample["code"] = sample["code"].replace(f"{class_name}", "SampleSolution")
            else:
                delete_count += 1
                continue
            
            # Check the sample test
            modified_input, modified_output, delete_test = check_test(sample["language"].lower(), i)
            # Currently delete test and modify test are not supported at the same time
            if delete_test is not None:
                for i, test_id in enumerate(delete_test):
                    del sample["test_IO"][test_id-i-1]
            else:
                if modified_input is not None:
                    for test_id, input_data in modified_input.items():
                        sample["test_IO"][test_id-1]["input"] = input_data
                if modified_output is not None:
                    for test_id, output_data in modified_output.items():
                        sample["test_IO"][test_id-1]["output"] = output_data

            # remove the "dataset" and "language"
            del sample["dataset"]
            del sample["language"]
            f.write(json.dumps(sample) + '\n')
    
    print(f"Saved {len(java_data) - delete_count} Java samples to {java_file_path}")

print("Dataset processing complete!")