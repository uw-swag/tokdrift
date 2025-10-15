import json
import datasets


_DESCRIPTION = """
CodeNet dataset subset for code translation tasks, organized by source language.
This dataset contains code samples across 3 programming languages and each language has 200 samples. 
"""

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
    """BuilderConfig for CodeNet dataset"""

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
            description="Python HumanEvalPack",
            features=[
                "id", "code", "test_IO"
            ]
        ),      
        CodenetConfig(
            name="java",
            description="Java Avatar",
            features=[
                "id", "code", "test_IO"
            ]
        ),
        CodenetConfig(
            name="cpp",
            description="C++ Avatar",
            features=[
                "id", "code", "test_IO"
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
                    )
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
                }  
                key += 1