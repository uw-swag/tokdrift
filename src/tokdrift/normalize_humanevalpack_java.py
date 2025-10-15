import re
import ast
import os
import json
from typing import List, Set, Dict, Any, Tuple
from collections import defaultdict
from io import BytesIO, StringIO
import tokenize
from antlr4 import *
from transformers import AutoTokenizer

from .grammars import *
from .regex import RegexProcessor
from .data_filter import TxtDataFilter
from .config import Config
from .immutable_identifiers_handler import ImmutableIdentifiersHandler


class DataExtractor:
    def __init__(self, config: Config):
        self.config = config
        data_path = f"./data/input/humanevalpack.jsonl"
        # Load json file
        with open(data_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)

    
    def _get_tokenize_tokens_and_immutable_identifiers(self, context: str, initial_immutable_identifiers: Set[str]):
        """Get the tokens and immutable identifiers"""
        try:
            # context_bytes = context.encode('utf-8')
            if self.config.lang == "java":
                # Use ANTLR4 to tokenize Java code
                input_stream = InputStream(context)
                lexer = JavaLexer(input_stream)
                stream = CommonTokenStream(lexer)
                parser = JavaParser(stream)
                # Get immutable identifiers
                immutable_identifiers = ImmutableIdentifiersHandler().get_immutable_identifiers(parser, context, initial_immutable_identifiers, self.config.lang)
                stream.fill()
                tokens = stream.tokens
                for token in tokens:
                    if token.type == Token.EOF:
                        continue
                    token.type = lexer.symbolicNames[token.type] if token.type < len(lexer.symbolicNames) else 'UNKNOWN'
            else:
                raise ValueError(f"Unsupported language: {self.config.lang}")
            
            return tokens, immutable_identifiers
        except Exception as e:
            print(f"Tokenize parsing failed: {e}. Skipping this example")
            return None
    
    def _get_test_tokens(self, test_case: str):
        """This is a method to get the tokens from the test case for HumanEvalPack python tasks"""
        test_tokens = []
        if self.config.lang == "python":
            tokens = tokenize.generate_tokens(StringIO(test_case).readline)
            pos_count = 0
            for token in tokens:
                if token.type == tokenize.INDENT or token.type == tokenize.DEDENT or token.type == tokenize.ENDMARKER or token.string == '':
                    continue
                while pos_count < len(test_case) and test_case[pos_count] == " ":
                    pos_count += 1
                test_tokens.append({
                    'token_name': token.string,
                    'identifier': token.type == tokenize.NAME,
                    'pos': f'({pos_count}, {pos_count+len(token.string)})'
                })
                pos_count += len(token.string)
        elif self.config.lang == "java":
            input_stream = InputStream(test_case)
            lexer = JavaLexer(input_stream)
            stream = CommonTokenStream(lexer)
            parser = JavaParser(stream)
            stream.fill()
            tokens = stream.tokens
            for token in tokens:
                if token.type == Token.EOF:
                    continue
                token.type = lexer.symbolicNames[token.type] if token.type < len(lexer.symbolicNames) else 'UNKNOWN'
                test_tokens.append({
                    'token_name': token.text,
                    'identifier': token.type == 'IDENTIFIER',
                    'pos': f'({token.start}, {token.stop + 1})'
                })
        else:
            raise ValueError(f"Unsupported language: {self.config.lang}")
        return test_tokens
    
    def _get_tokenize_tokens_details(self, context: str, initial_immutable_identifiers: Set[str]):
        """Get the tokens from the tokenize library with details"""
        try:
            # Get the tokens by the tokenize library
            tokens, immutable_identifiers = self._get_tokenize_tokens_and_immutable_identifiers(context, initial_immutable_identifiers)

            tokenize_tokens = []
            pos_count = 0
            whitespace_count = 0
            for idx, token in enumerate(tokens):
                # Process tokens from ANTLR4
                if token.type == Token.EOF:
                    continue

                token_text = token.text
                java_keywords = {'ABSTRACT', 'ASSERT', 'BOOLEAN', 'BREAK', 'BYTE', 'CASE', 'CATCH', 'CHAR',
                                    'CLASS', 'CONST', 'CONTINUE', 'DEFAULT', 'DO', 'DOUBLE', 'ELSE', 'ENUM',
                                    'EXTENDS', 'FINAL', 'FINALLY', 'FLOAT', 'FOR', 'GOTO', 'IF', 'IMPLEMENTS', 'IMPORT',
                                    'INSTANCEOF', 'INT', 'INTERFACE', 'LONG', 'NATIVE', 'NEW', 'PACKAGE', 'PRIVATE',
                                    'PROTECTED', 'PUBLIC', 'RETURN', 'SHORT', 'STATIC', 'STRICTFP', 'SUPER', 'SWITCH',
                                    'SYNCHRONIZED', 'THIS', 'THROW', 'THROWS', 'TRANSIENT', 'TRY', 'VOID', 'VOLATILE',
                                    'WHILE'}
                is_identifier = (token.type == 'IDENTIFIER' or token.type in java_keywords) and token_text not in immutable_identifiers
                operator_types = {'ASSIGN', 'GT', 'LT', 'BANG', 'TILDE', 'QUESTION', 'COLON',
                                    'EQUAL', 'LE', 'GE', 'NOTEQUAL', 'AND', 'OR', 'INC', 'DEC',
                                    'ADD', 'SUB', 'MUL', 'DIV', 'BITAND', 'BITOR', 'CARET', 'MOD',
                                    'ARROW', 'COLONCOLON', 'ADD_ASSIGN', 'SUB_ASSIGN', 'MUL_ASSIGN',
                                    'DIV_ASSIGN', 'AND_ASSIGN', 'OR_ASSIGN', 'XOR_ASSIGN', 'MOD_ASSIGN',
                                    'LSHIFT_ASSIGN', 'RSHIFT_ASSIGN', 'URSHIFT_ASSIGN',
                                    'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'LBRACK', 'RBRACK',
                                    'SEMI', 'DOT', 'COMMA'}
                is_operator = token.type in operator_types

                token.type = 'NAME' if is_identifier else 'OP' if is_operator else token.type

                tokenize_tokens.append({
                    'token_name': token_text,
                    'identifier': is_identifier,
                    'type': token.type,
                    'pos': f'({token.start}, {token.stop + 1})'
                })

            return tokenize_tokens
        except Exception as e:
            print(f"Tokenize parsing failed: {e}. Skipping this example")
            return None
    
    def _get_target_identifiers(self, buggy_tokenize_tokens, canonical_tokenize_tokens, declaration, test_case = None):
        """Get the multi-token identifiers from the context"""
        
        # Get the target names from the tokenize tokens
        buggy_target_names = set()
        canonical_target_names = set()
        dec_length = len(declaration)
        
        buggy_multi_token_identifiers = []
        canonical_multi_token_identifiers = []
        for idx, token in enumerate(buggy_tokenize_tokens):
            if token['identifier']:
                pos_start = int(token['pos'].split(",")[0].strip('('))
                pos_end = int(token['pos'].split(",")[1].strip(')'))
                
                tokenize_token = token['token_name']
                token_pos = f'({pos_start}, {pos_end})'
                is_in_declaration = pos_end < dec_length
                buggy_multi_token_identifiers.append({
                    'tokenize_token': tokenize_token, 
                    'pos': token_pos,
                    'is_in_declaration': is_in_declaration
                })
                buggy_target_names.add(tokenize_token)
        
        for idx, token in enumerate(canonical_tokenize_tokens):
            if token['identifier']:
                pos_start = int(token['pos'].split(",")[0].strip('('))
                pos_end = int(token['pos'].split(",")[1].strip(')'))
                tokenize_token = token['token_name']
                token_pos = f'({pos_start}, {pos_end})'
                is_in_declaration = pos_end < dec_length
                canonical_multi_token_identifiers.append({
                    'tokenize_token': tokenize_token, 
                    'pos': token_pos,
                    'is_in_declaration': is_in_declaration
                })
                canonical_target_names.add(tokenize_token)
        
        test_target_identifiers = []
        # Get the test tokens for HumanEvalPack python tasks
        if test_case is not None:
            test_tokens = self._get_test_tokens(test_case)
            for token in test_tokens:
                if token['identifier'] and token['token_name']in canonical_target_names:
                    test_target_identifiers.append(token)

        return buggy_multi_token_identifiers, canonical_multi_token_identifiers, test_target_identifiers
    
    def extract_new_identifiers(self):
        """Extract new multi-token identifiers"""
        all_buggy_multi_token_identifiers = []
        all_canonical_multi_token_identifiers = []
        all_test_identifiers = []
        # Process each sample in the dataset
        for idx, sample in enumerate(self.dataset):
            print(f"Processing example {idx} of {len(self.dataset)}...")
            
            # Get the context only from the dataset sample
            buggy_context = sample["declaration"] + sample["buggy_solution"]
            canonical_context = sample["declaration"] + sample["canonical_solution"]

            # Get the immutable identifiers from the context
            immutable_identifiers = set()
            immutable_identifiers.add("Solution")

            # Get the context tokens by tokenize library
            buggy_tokenize_tokens = self._get_tokenize_tokens_details(buggy_context, immutable_identifiers)
            canonical_tokenize_tokens = self._get_tokenize_tokens_details(canonical_context, immutable_identifiers)

            # Get the multi-token identifiers from the context
            test_case = "\n" + sample["test"]
            declaration = sample["declaration"]
            buggy_multi_token_identifiers, canonical_multi_token_identifiers, test_target_identifiers = self._get_target_identifiers(buggy_tokenize_tokens, canonical_tokenize_tokens, declaration, test_case)

            all_buggy_multi_token_identifiers.append(buggy_multi_token_identifiers)
            all_canonical_multi_token_identifiers.append(canonical_multi_token_identifiers)
            all_test_identifiers.append(test_target_identifiers)
        
        return all_buggy_multi_token_identifiers, all_canonical_multi_token_identifiers, all_test_identifiers


class DataGenerator:
    def __init__(self, config: Config):
        self.config = config
        data_path = f"./data/input/humanevalpack.jsonl"
        # self.output_path = f"./datasets/humanevalpack/base/data/java/data/humanevalpack.jsonl"
        self.output_path = f"./data/input/output.jsonl"
        # Load json file
        with open(data_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)

    def process_multi_token_identifiers(self, multi_token_identifiers, context, declaration):
        """Process multi-token identifiers"""
        filtered_identifiers = []
        new_context = context
        for identifier in multi_token_identifiers:
            if identifier['tokenize_token'] not in filtered_identifiers:
                filtered_identifiers.append(identifier['tokenize_token'])
        filtered_identifiers = TxtDataFilter.generate_filtered_word_list(self.config.filter_type, filtered_identifiers)

        total_difference = 0
        difference_declaration = 0
        for identifier in multi_token_identifiers:
            if identifier['tokenize_token'] in filtered_identifiers:
                new_target_identifier, _ = RegexProcessor.to_target_case(identifier['tokenize_token'], self.config.target_type)
                difference = len(new_target_identifier) - len(identifier['tokenize_token'])
                
                start_pos = int(identifier['pos'].split(",")[0].strip('('))
                end_pos = int(identifier['pos'].split(",")[1].strip(')'))

                new_context = new_context[:start_pos + total_difference] + new_target_identifier + new_context[end_pos + total_difference:]
                total_difference += difference
                if identifier['is_in_declaration']:
                    difference_declaration += difference
        
        new_declaration = new_context[:len(declaration) + difference_declaration]
        new_context = new_context[len(new_declaration):]

        return new_context, new_declaration

    def process_test_identifiers(self, test_case, test_target_identifiers):
        filtered_identifiers = []
        new_test_case = test_case
        for identifier in test_target_identifiers:
            if identifier['token_name'] not in filtered_identifiers:
                filtered_identifiers.append(identifier['token_name'])
        filtered_identifiers = TxtDataFilter.generate_filtered_word_list(self.config.filter_type, filtered_identifiers)

        total_difference = 0
        for identifier in test_target_identifiers:
            if identifier['token_name'] in filtered_identifiers:
                new_target_identifier, _ = RegexProcessor.to_target_case(identifier['token_name'], self.config.target_type)
                difference = len(new_target_identifier) - len(identifier['token_name'])
                
                start_pos = int(identifier['pos'].split(",")[0].strip('('))
                end_pos = int(identifier['pos'].split(",")[1].strip(')'))

                new_test_case = new_test_case[:start_pos + total_difference] + new_target_identifier + new_test_case[end_pos + total_difference:]
                total_difference += difference

        return new_test_case

    def process_all_multi_token_identifiers(self, all_buggy_multi_token_identifiers, all_canonical_multi_token_identifiers, all_test_identifiers):
        """Process all multi-token identifiers"""
        new_buggy_contexts = {} 
        new_canonical_contexts = {}
        new_declarations = {}
        all_modified_tests = {}
        for idx, canonical_multi_token_identifiers in enumerate(all_canonical_multi_token_identifiers):
            context = self.dataset[idx]["declaration"] + self.dataset[idx]["canonical_solution"]
            declaration = self.dataset[idx]["declaration"]
            test_case = "\n" + self.dataset[idx]["test"]
            new_canonical_context, new_declaration = self.process_multi_token_identifiers(canonical_multi_token_identifiers, context, declaration)
            if new_canonical_context:
                new_idx = self.dataset[idx]["task_id"]
                new_canonical_contexts[new_idx] = new_canonical_context
                new_declarations[new_idx] = new_declaration
                test_target_identifiers = all_test_identifiers[idx]
                new_test_case = self.process_test_identifiers(test_case, test_target_identifiers)
                all_modified_tests[new_idx] = new_test_case
        for idx, buggy_multi_token_identifiers in enumerate(all_buggy_multi_token_identifiers):
            context = self.dataset[idx]["declaration"] + self.dataset[idx]["buggy_solution"]
            declaration = self.dataset[idx]["declaration"]
            new_buggy_context, new_declaration = self.process_multi_token_identifiers(buggy_multi_token_identifiers, context, declaration)
            if new_buggy_context:
                new_idx = self.dataset[idx]["task_id"]
                # Check if the new buggy declaration is the same as the canonical declaration
                if new_declaration != new_declarations[new_idx]:
                    print(f"New buggy declaration: {new_declaration}")
                    print(f"Canonical declaration: {new_declarations[new_idx]}")
                    raise ValueError(f"New buggy declaration is different from the canonical declaration. Example {new_idx}")
                else:
                    print(f"New buggy declaration is the same in example {new_idx}")
                new_buggy_contexts[new_idx] = new_buggy_context

        return new_buggy_contexts, new_canonical_contexts, all_modified_tests, new_declarations
    
    
    def generate_new_dataset(self, new_buggy_contexts, new_canonical_contexts, all_modified_tests=None, new_declarations=None):
        """Generate new dataset with modified contexts (works for both tasks)"""
        new_dataset = self.dataset
        
        # Create a filtered dataset containing only samples with modified contexts
        filtered_dataset = new_dataset.filter(
            lambda sample: sample["task_id"] in new_canonical_contexts
        )
            
        # Update the samples with their modified contexts
        def add_modified_context(sample):
            task_id = sample["task_id"]
            sample['buggy_solution'] = new_buggy_contexts[task_id]
            sample['canonical_solution'] = new_canonical_contexts[task_id]
            sample['test'] = all_modified_tests[task_id]
            sample['declaration'] = new_declarations[task_id]
            return sample
        
        filtered_dataset = filtered_dataset.map(add_modified_context)
        
        # Save the new dataset as a JSONL file
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for sample in filtered_dataset:
                f.write(json.dumps(sample) + '\n')
        
        print(f"Saved {len(filtered_dataset)} samples to {self.output_path}")
    
    # Keep old method name for backwards compatibility
    def generate_new_multi_token_dataset(self, new_contexts):
        """Legacy method name - now calls generate_new_dataset"""
        return self.generate_new_dataset(new_contexts)
        
        
if __name__ == "__main__":
    
    # Initialize config
    config = Config()

    config.target_type = "camel_case"
    config.filter_type = "snake_case"

    data_extractor = DataExtractor(config)

    data_generator = DataGenerator(config)

    # Extract identifiers
    print(f"Extracting identifiers...")
    all_buggy_multi_token_identifiers, all_canonical_multi_token_identifiers, all_test_identifiers = data_extractor.extract_new_identifiers()

    # Generate new multi-token dataset
    print(f"Generating new multi-token dataset...")
    new_buggy_contexts, new_canonical_contexts, all_modified_tests, new_declarations = data_generator.process_all_multi_token_identifiers(all_buggy_multi_token_identifiers, all_canonical_multi_token_identifiers, all_test_identifiers)

    # Generate dataset file
    data_generator.generate_new_dataset(new_buggy_contexts, new_canonical_contexts, all_modified_tests, new_declarations)