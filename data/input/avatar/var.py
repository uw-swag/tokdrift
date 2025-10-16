import json

import datasets


_DESCRIPTION = """
Modified Avatar dataset for code translation tasks.
"""

_HOMEPAGE = ""

def get_url(name):
    url = f"data/{name}/data/avatar.jsonl"
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

class AvatarConfig(datasets.BuilderConfig):
    """BuilderConfig """

    def __init__(self, name, description, features, **kwargs):
        super(AvatarConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.name = name
        self.description = description
        self.features = features


class Avatar(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        AvatarConfig(
            name="python",
            description="Python Avatar",
            features=[
                "id", "code", "test_IO"
            ]
        ),      
        AvatarConfig(
            name="java",
            description="Java Avatar",
            features=[
                "id", "code", "test_IO"
            ]
        ),
        AvatarConfig(
            name="cpp",
            description="C++ Avatar",
            features=[
                "id", "code", "test_IO"
            ]
        ),
        # Python naming convention variants
        AvatarConfig(
            name="python-camel_case",
            description="Python Avatar with camel case",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="python-pascal_case",
            description="Python Avatar with pascal case",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="python-screaming_snake_case",
            description="Python Avatar with screaming snake case",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        # Python combined token operator variants
        AvatarConfig(
            name="python-lparentheses_name",
            description="Python Avatar with left parentheses and name",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="python-lparentheses_rparentheses",
            description="Python Avatar with left parentheses and right parentheses",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="python-rparentheses_colon",
            description="Python Avatar with right parentheses and colon",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="python-rparentheses_rparentheses",
            description="Python Avatar with right parentheses and right parentheses",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="python-lsquarebracket_name",
            description="Python Avatar with left square bracket and name",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="python-period_name",
            description="Python Avatar with period and name",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="python-rsquarebracket_rparentheses",
            description="Python Avatar with right square bracket and right parentheses",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="python-op_lsquarebracket",
            description="Python Avatar with operator and left square bracket",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="python-op_rsquarebracket",
            description="Python Avatar with operator and right square bracket",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="python-op_dash",
            description="Python Avatar with operator and dash",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="python-op_name",
            description="Python Avatar with operator and name",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="python-op_all",
            description="Python Avatar with operator and all",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        # Java naming convention variants
        AvatarConfig(
            name="java-snake_case",
            description="Java Avatar with snake case",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="java-pascal_case",
            description="Java Avatar with pascal case",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="java-screaming_snake_case",
            description="Java Avatar with screaming snake case",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        # Java combined token operator variants
        AvatarConfig(
            name="java-rparentheses_semicolon",
            description="Java Avatar with right parentheses and semicolon",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="java-lparentheses_name",
            description="Java Avatar with left parentheses and name",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="java-lparentheses_rparentheses",
            description="Java Avatar with left parentheses and right parentheses",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="java-rparentheses_rparentheses",
            description="Java Avatar with right parentheses and right parentheses",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="java-period_name",
            description="Java Avatar with period and name",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="java-period_asterisk",
            description="Java Avatar with period and asterisk",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="java-double_plus_rparentheses",
            description="Java Avatar with double plus and right parentheses",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="java-rparentheses_period",
            description="Java Avatar with right parentheses and period",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="java-op_semicolon",
            description="Java Avatar with operator and semicolon",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="java-op_lparentheses",
            description="Java Avatar with operator and left parentheses",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="java-op_name",
            description="Java Avatar with operator and name",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="java-op_all",
            description="Java Avatar with operator and all",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        # C++ naming convention variants
        AvatarConfig(
            name="cpp-snake_case",
            description="C++ Avatar with snake case",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="cpp-camel_case",
            description="C++ Avatar with camel case",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="cpp-pascal_case",
            description="C++ Avatar with pascal case",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        AvatarConfig(
            name="cpp-screaming_snake_case",
            description="C++ Avatar with screaming snake case",
            features=[
                "id", "code", "test_IO", "modified_context", "token_boundary_changed"
            ]
        ),
        # C++ combined token operator variants
        AvatarConfig(
            name="cpp-lparentheses_name",
            description="C++ Avatar with left parentheses and name",
            features=[
                "id", "code", "test_IO", "modified_context"
            ]
        ),
        AvatarConfig(
            name="cpp-rparentheses_colon",
            description="C++ Avatar with right parentheses and colon",
            features=[
                "id", "code", "test_IO", "modified_context"
            ]
        ),
        AvatarConfig(
            name="cpp-rparentheses_rparentheses",
            description="C++ Avatar with right parentheses and right parentheses",
            features=[
                "id", "code", "test_IO", "modified_context"
            ]
        ),
        AvatarConfig(
            name="cpp-lsquarebracket_name",
            description="C++ Avatar with left square bracket and name",
            features=[
                "id", "code", "test_IO", "modified_context"
            ]
        ),
        AvatarConfig(
            name="cpp-period_name",
            description="C++ Avatar with period and name",
            features=[
                "id", "code", "test_IO", "modified_context"
            ]
        ),
        AvatarConfig(
            name="cpp-rsquarebracket_rparentheses",
            description="C++ Avatar with right square bracket and right parentheses",
            features=[
                "id", "code", "test_IO", "modified_context"
            ]
        ),
        AvatarConfig(
            name="cpp-op_dash",
            description="C++ Avatar with operator and dash",
            features=[
                "id", "code", "test_IO", "modified_context"
            ]
        ),
        AvatarConfig(
            name="cpp-op_name",
            description="C++ Avatar with operator and name",
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