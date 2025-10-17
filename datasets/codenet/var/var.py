import json

import datasets


_DESCRIPTION = """
Modified CodeNet dataset subset for code translation tasks, organized by source language.
This dataset contains code translation pairs across 2 programming languages.
Each language has 200 samples.
"""

_HOMEPAGE = ""

def get_url(name):
    url = f"data/{name}/data/codenet.jsonl"
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

class CodenetConfig(datasets.BuilderConfig):
    """BuilderConfig """

    def __init__(self, name, description, features, **kwargs):
        super(CodenetConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.name = name
        self.description = description
        self.features = features


class Codenet(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        CodenetConfig(
            name="python",
            description="Python CodeNet",
            features=[
                "id", "code", "test_IO"
            ]
        ),      
        CodenetConfig(
            name="java",
            description="Java CodeNet",
            features=[
                "id", "code", "test_IO"
            ]
        ),
        CodenetConfig(
            name="cpp",
            description="C++ CodeNet",
            features=[
                "id", "code", "test_IO"
            ]
        ),
        # Python naming convention variants
        CodenetConfig(
            name="python-camel_case",
            description="Python CodeNet with camel case",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="python-pascal_case",
            description="Python CodeNet with pascal case",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="python-screaming_snake_case",
            description="Python CodeNet with screaming snake case",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        # Python combined token operator variants
        CodenetConfig(
            name="python-lparentheses_name",
            description="Python CodeNet with left parentheses and name",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="python-lparentheses_rparentheses",
            description="Python CodeNet with left parentheses and right parentheses",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="python-rparentheses_colon",
            description="Python CodeNet with right parentheses and colon",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="python-rparentheses_rparentheses",
            description="Python CodeNet with right parentheses and right parentheses",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="python-lsquarebracket_name",
            description="Python CodeNet with left square bracket and name",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="python-period_name",
            description="Python CodeNet with period and name",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="python-rsquarebracket_rparentheses",
            description="Python CodeNet with right square bracket and right parentheses",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="python-op_lsquarebracket",
            description="Python CodeNet with operator and left square bracket",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="python-op_rsquarebracket",
            description="Python CodeNet with operator and right square bracket",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="python-op_dash",
            description="Python CodeNet with operator and dash",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="python-op_name",
            description="Python CodeNet with operator and name",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="python-op_all",
            description="Python CodeNet with operator and all",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        # Java naming convention variants
        CodenetConfig(
            name="java-snake_case",
            description="Java CodeNet with snake case",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="java-pascal_case",
            description="Java CodeNet with pascal case",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="java-screaming_snake_case",
            description="Java CodeNet with screaming snake case",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        # Java combined token operator variants
        CodenetConfig(
            name="java-rparentheses_semicolon",
            description="Java CodeNet with right parentheses and semicolon",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="java-lparentheses_name",
            description="Java CodeNet with left parentheses and name",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="java-lparentheses_rparentheses",
            description="Java CodeNet with left parentheses and right parentheses",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="java-rparentheses_rparentheses",
            description="Java CodeNet with right parentheses and right parentheses",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="java-period_name",
            description="Java CodeNet with period and name",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="java-period_asterisk",
            description="Java CodeNet with period and asterisk",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="java-double_plus_rparentheses",
            description="Java CodeNet with double plus and right parentheses",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="java-rparentheses_period",
            description="Java CodeNet with right parentheses and period",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="java-op_semicolon",
            description="Java CodeNet with operator and semicolon",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="java-op_lparentheses",
            description="Java CodeNet with operator and left parentheses",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="java-op_name",
            description="Java CodeNet with operator and name",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="java-op_all",
            description="Java CodeNet with operator and all",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        # C++ naming convention variants
        CodenetConfig(
            name="cpp-snake_case",
            description="C++ Avatar with snake case",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="cpp-camel_case",
            description="C++ Avatar with camel case",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="cpp-pascal_case",
            description="C++ Avatar with pascal case",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        CodenetConfig(
            name="cpp-screaming_snake_case",
            description="C++ Avatar with screaming snake case",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        # C++ combined token operator variants
        CodenetConfig(
            name="cpp-lparentheses_name",
            description="C++ CodeNet with left parentheses and name",
            features=[
                "id", "code", "test_IO", "modified_context"
            ]
        ),
        CodenetConfig(
            name="cpp-rparentheses_colon",
            description="C++ CodeNet with right parentheses and colon",
            features=[
                "id", "code", "test_IO", "modified_context"
            ]
        ),
        CodenetConfig(
            name="cpp-rparentheses_rparentheses",
            description="C++ CodeNet with right parentheses and right parentheses",
            features=[
                "id", "code", "test_IO", "modified_context"
            ]
        ),
        CodenetConfig(
            name="cpp-lsquarebracket_name",
            description="C++ CodeNet with left square bracket and name",
            features=[
                "id", "code", "test_IO", "modified_context"
            ]
        ),
        CodenetConfig(
            name="cpp-period_name",
            description="C++ CodeNet with period and name",
            features=[
                "id", "code", "test_IO", "modified_context"
            ]
        ),
        CodenetConfig(
            name="cpp-rsquarebracket_rparentheses",
            description="C++ CodeNet with right square bracket and right parentheses",
            features=[
                "id", "code", "test_IO", "modified_context"
            ]
        ),
        CodenetConfig(
            name="cpp-op_dash",
            description="C++ CodeNet with operator and dash",
            features=[
                "id", "code", "test_IO", "modified_context"
            ]
        ),
        CodenetConfig(
            name="cpp-op_name",
            description="C++ CodeNet with operator and name",
            features=[
                "id", "code", "test_IO", "modified_context"
            ]
        ),
        
    ]
    DEFAULT_CONFIG_NAME = "python"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "code": datasets.Value("string"),
                    "test_IO": datasets.Sequence(
                        {
                            "input": datasets.Value("string"),
                            "output": datasets.Value("string")
                        }
                    ),
                    "modified_context": datasets.Value("string"),
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
                    "id": row["id"],
                    "code": row["code"],
                    "test_IO": row["test_IO"],
                    "modified_context": row.get("modified_context", ""),
                }  
                key += 1