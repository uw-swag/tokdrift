import json

import datasets


_DESCRIPTION = """
This is a modified version of the HumanEvalPack dataset based on Llama-3 tokenizer. 
For code summarization and bug fixing tasks.
"""


def get_url(name):
    url = f"data/{name}/data/humanevalpack.jsonl"
    return url

def split_generator(dl_manager, name):
    downloaded_files = dl_manager.download(get_url(name))
    return [
        datasets.SplitGenerator(
            name=datasets.Split.TEST,
            gen_kwargs={
                "filepath": downloaded_files,
            },
        )
    ]

class HumanEvalPackConfig(datasets.BuilderConfig):
    """BuilderConfig """

    def __init__(self, name, description, features, **kwargs):
        super(HumanEvalPackConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.name = name
        self.description = description
        self.features = features


class HumanEvalPack(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        HumanEvalPackConfig(
            name="python",
            description="Python HumanEvalPack",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction"
            ]
        ),
        HumanEvalPackConfig(
            name="js",
            description="JavaScript HumanEvalPack",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction"
            ]
        ),        
        HumanEvalPackConfig(
            name="java",
            description="Java HumanEvalPack",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction"
            ]
        ),
        HumanEvalPackConfig(
            name="go",
            description="Go HumanEvalPack",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction"
            ]
        ),        
        HumanEvalPackConfig(
            name="cpp",
            description="C++ HumanEvalPack",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction"
            ]
        ),        
        HumanEvalPackConfig(
            name="rust",
            description="Rust HumanEvalPack",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction"
            ]
        ),
        # Java naming convention variants
        HumanEvalPackConfig(
            name="java-pascal_case",
            description="Java HumanEvalPack (Pascal Case)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed", "modified_test", "modified_entry_point", "modified_declaration"
            ]
        ),
        HumanEvalPackConfig(
            name="java-snake_case",
            description="Java HumanEvalPack (Snake Case)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed", "modified_test", "modified_entry_point", "modified_declaration"
            ]
        ),
        HumanEvalPackConfig(
            name="java-screaming_snake_case",
            description="Java HumanEvalPack (Screaming Snake Case)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed", "modified_test", "modified_entry_point", "modified_declaration"
            ]
        ),
        HumanEvalPackConfig(
            name="java-rparentheses_semicolon",
            description="Java HumanEvalPack (Right Parentheses + Semicolon)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="java-lparentheses_name",
            description="Java HumanEvalPack (Left Parentheses + Name)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="java-lparentheses_rparentheses",
            description="Java HumanEvalPack (Left Parentheses + Right Parentheses)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="java-rparentheses_rparentheses",
            description="Java HumanEvalPack (Right Parentheses + Right Parentheses)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="java-period_name",
            description="Java HumanEvalPack (Period + Name)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="java-period_asterisk",
            description="Java HumanEvalPack (Period + Asterisk)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="java-double_plus_rparentheses",
            description="Java HumanEvalPack (Double Plus + Right Parentheses)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="java-rparentheses_period",
            description="Java HumanEvalPack (Right Parentheses + Period)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="java-op_semicolon",
            description="Java HumanEvalPack (Operator + Semicolon)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="java-op_lparentheses",
            description="Java HumanEvalPack (Operator + Left Parentheses)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="java-op_name",
            description="Java HumanEvalPack (Operator + Name)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="java-op_all",
            description="Java HumanEvalPack (Operator + All)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        # Python naming convention variants
        HumanEvalPackConfig(
            name="python-pascal_case",
            description="Python HumanEvalPack (Pascal Case)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed", "modified_test", "modified_entry_point", "modified_declaration"
            ]
        ),
        HumanEvalPackConfig(
            name="python-camel_case",
            description="Python HumanEvalPack (Camel Case)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed", "modified_test", "modified_entry_point", "modified_declaration"
            ]
        ),
        HumanEvalPackConfig(
            name="python-screaming_snake_case",
            description="Python HumanEvalPack (Screaming Snake Case)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed", "modified_test", "modified_entry_point", "modified_declaration"
            ]
        ),
        # Combined token operator variants
        HumanEvalPackConfig(
            name="python-lparentheses_name",
            description="Python HumanEvalPack (Left Parentheses + Name)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="python-lparentheses_rparentheses",
            description="Python HumanEvalPack (Left Parentheses + Right Parentheses)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="python-rparentheses_colon",
            description="Python HumanEvalPack (Right Parentheses + Colon)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="python-rparentheses_rparentheses",
            description="Python HumanEvalPack (Right Parentheses + Right Parentheses)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="python-lsquarebracket_name",
            description="Python HumanEvalPack (Left Square Bracket + Name)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="python-period_name",
            description="Python HumanEvalPack (Period + Name)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="python-rsquarebracket_rparentheses",
            description="Python HumanEvalPack (Right Square Bracket + Right Parentheses)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="python-op_lsquarebracket",
            description="Python HumanEvalPack (Operator + Left Square Bracket)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="python-op_rsquarebracket",
            description="Python HumanEvalPack (Operator + Right Square Bracket)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="python-op_dash",
            description="Python HumanEvalPack (Operator + Dash)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="python-op_name",
            description="Python HumanEvalPack (Operator + Name)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="python-op_all",
            description="Python HumanEvalPack (Operator + All)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        # Java fix variants
        HumanEvalPackConfig(
            name="java-pascal_case-fix",
            description="Java HumanEvalPack (Pascal Case - Fix)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed", "modified_test", "modified_entry_point", "modified_declaration"
            ]
        ),
        HumanEvalPackConfig(
            name="java-snake_case-fix",
            description="Java HumanEvalPack (Snake Case - Fix)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed", "modified_test", "modified_entry_point", "modified_declaration"
            ]
        ),
        HumanEvalPackConfig(
            name="java-screaming_snake_case-fix",
            description="Java HumanEvalPack (Screaming Snake Case - Fix)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed", "modified_test", "modified_entry_point", "modified_declaration"
            ]
        ),
        HumanEvalPackConfig(
            name="java-rparentheses_semicolon-fix",
            description="Java HumanEvalPack (Right Parentheses + Semicolon)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="java-lparentheses_name-fix",
            description="Java HumanEvalPack (Left Parentheses + Name)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="java-lparentheses_rparentheses-fix",
            description="Java HumanEvalPack (Left Parentheses + Right Parentheses)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="java-rparentheses_rparentheses-fix",
            description="Java HumanEvalPack (Right Parentheses + Right Parentheses)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="java-period_name-fix",
            description="Java HumanEvalPack (Period + Name)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="java-period_asterisk-fix",
            description="Java HumanEvalPack (Period + Asterisk)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="java-double_plus_rparentheses-fix",
            description="Java HumanEvalPack (Double Plus + Right Parentheses)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="java-rparentheses_period-fix",
            description="Java HumanEvalPack (Right Parentheses + Period)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="java-op_semicolon-fix",
            description="Java HumanEvalPack (Operator + Semicolon)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="java-op_lparentheses-fix",
            description="Java HumanEvalPack (Operator + Left Parentheses)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="java-op_name-fix",
            description="Java HumanEvalPack (Operator + Name)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="java-op_all-fix",
            description="Java HumanEvalPack (Operator + All)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        # Python fix variants
        HumanEvalPackConfig(
            name="python-pascal_case-fix",
            description="Python HumanEvalPack (Pascal Case - Fix)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed", "modified_test", "modified_entry_point", "modified_declaration"
            ]
        ),
        HumanEvalPackConfig(
            name="python-camel_case-fix",
            description="Python HumanEvalPack (Camel Case - Fix)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed", "modified_test", "modified_entry_point", "modified_declaration"
            ]
        ),
        HumanEvalPackConfig(
            name="python-screaming_snake_case-fix",
            description="Python HumanEvalPack (Screaming Snake Case - Fix)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed", "modified_test", "modified_entry_point", "modified_declaration"
            ]
        ),
        HumanEvalPackConfig(
            name="python-lparentheses_name-fix",
            description="Python HumanEvalPack (Left Parentheses + Name)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="python-lparentheses_rparentheses-fix",
            description="Python HumanEvalPack (Left Parentheses + Right Parentheses)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="python-rparentheses_colon-fix",
            description="Python HumanEvalPack (Right Parentheses + Colon)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="python-rparentheses_rparentheses-fix",
            description="Python HumanEvalPack (Right Parentheses + Right Parentheses)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="python-lsquarebracket_name-fix",
            description="Python HumanEvalPack (Left Square Bracket + Name)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="python-period_name-fix",
            description="Python HumanEvalPack (Period + Name)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="python-rsquarebracket_rparentheses-fix",
            description="Python HumanEvalPack (Right Square Bracket + Right Parentheses)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="python-op_lsquarebracket-fix",
            description="Python HumanEvalPack (Operator + Left Square Bracket)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="python-op_rsquarebracket-fix",
            description="Python HumanEvalPack (Operator + Right Square Bracket)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="python-op_dash-fix",
            description="Python HumanEvalPack (Operator + Dash)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="python-op_name-fix",
            description="Python HumanEvalPack (Operator + Name)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
        HumanEvalPackConfig(
            name="python-op_all-fix",
            description="Python HumanEvalPack (Operator + All)",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction", "modified_context", "token_boundary_changed"
            ]
        ),
    ]
    DEFAULT_CONFIG_NAME = "python"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "task_id": datasets.Value("string"),
                    "prompt": datasets.Value("string"),                  
                    "declaration": datasets.Value("string"),
                    "canonical_solution": datasets.Value("string"),
                    "buggy_solution": datasets.Value("string"),
                    "bug_type": datasets.Value("string"),
                    "failure_symptoms": datasets.Value("string"),
                    "entry_point": datasets.Value("string"),
                    "import": datasets.Value("string"),  
                    "test_setup": datasets.Value("string"),
                    "test": datasets.Value("string"),
                    "example_test": datasets.Value("string"),
                    "signature": datasets.Value("string"),
                    "docstring": datasets.Value("string"),
                    "instruction": datasets.Value("string"),
                    "modified_context": datasets.Value("string"),
                    "token_boundary_changed": datasets.Value("bool"),
                    "modified_test": datasets.Value("string"),
                    "modified_entry_point": datasets.Value("string"),
                    "modified_declaration": datasets.Value("string"),
                }
            ),
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager):
        return split_generator(dl_manager, self.config.name)
           
    def _generate_examples(self, filepath):
        key = 0
        with open(filepath) as f:
            for line in f:
                row = json.loads(line)
                key += 1
                yield key, {
                    "task_id": row["task_id"],
                    "prompt": row["prompt"],
                    "declaration": row["declaration"],
                    "canonical_solution": row["canonical_solution"],
                    "buggy_solution": row["buggy_solution"],
                    "bug_type": row["bug_type"],
                    "failure_symptoms": row["failure_symptoms"],
                    "import": row.get("import", ""), # Only for Go                    
                    "test_setup": row.get("test_setup", ""), # Only for Go
                    "test": row["test"],
                    "example_test": row["example_test"],
                    "entry_point": row["entry_point"],
                    "signature": row["signature"],
                    "docstring": row["docstring"],
                    "instruction": row["instruction"],
                    "modified_context": row.get("modified_context", ""),
                    "token_boundary_changed": row.get("token_boundary_changed", False),
                    "modified_test": row.get("modified_test", ""),
                    "modified_entry_point": row.get("modified_entry_point", ""),
                    "modified_declaration": row.get("modified_declaration", ""),
                }  
                key += 1