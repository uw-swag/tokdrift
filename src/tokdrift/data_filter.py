import os

from .regex import RegexProcessor
from .config import Config


class TxtDataFilter:
    @staticmethod
    def filter_camel_case(word_list: list) -> list:
        return [word for word in word_list if RegexProcessor.is_camel_case(word)]
    
    @staticmethod
    def filter_pascal_case(word_list: list) -> list:
        return [word for word in word_list if RegexProcessor.is_pascal_case(word) and not RegexProcessor.is_screaming_snake_case(word)]
    
    @staticmethod
    def filter_snake_case(word_list: list) -> list:
        return [word for word in word_list if RegexProcessor.is_snake_case(word)]
    
    @staticmethod
    def filter_screaming_snake_case(word_list: list) -> list:
        return [word for word in word_list if RegexProcessor.is_screaming_snake_case(word)]
    

    @staticmethod
    def generate_filtered_word_list(regex_type: str, word_list: list):
        if regex_type == "camel_case":
            filtered_word_list = TxtDataFilter.filter_camel_case(word_list)
        elif regex_type == "pascal_case":
            filtered_word_list = TxtDataFilter.filter_pascal_case(word_list)
        elif regex_type == "snake_case":
            filtered_word_list = TxtDataFilter.filter_snake_case(word_list)
        elif regex_type == "screaming_snake_case":
            filtered_word_list = TxtDataFilter.filter_screaming_snake_case(word_list)
        
        return filtered_word_list

    @staticmethod
    def output_filtered_word_list(regex_type: str, filtered_word_list: list, output_dir: str):
        output_dir = output_dir.format(regex_type=regex_type)
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        with open(output_dir, "w") as f:
            for word in filtered_word_list:
                f.write(word + "\n")


if __name__ == "__main__":
    config = Config()
    data_filter = TxtDataFilter()
    word_list = open(config.output_identifiers_file, "r").read().splitlines()
    for regex_type in config.filter_types:
        filtered_word_list = data_filter.generate_filtered_word_list(regex_type, word_list)
        data_filter.output_filtered_word_list(regex_type, filtered_word_list, config.filter_output_dir)
        print(f"Filtered {regex_type} word list saved to {config.filter_output_dir}")
    
    
