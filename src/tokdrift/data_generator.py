import re
import ast
import os
import json
from typing import List, Set, Dict, Any, Tuple
from collections import defaultdict
from io import BytesIO, StringIO
import tokenize
from antlr4 import *
from datasets import load_dataset
from transformers import AutoTokenizer

from .grammars import *
from .regex import RegexProcessor
from .data_filter import TxtDataFilter
from .config import Config
from .immutable_identifiers_handler import ImmutableIdentifiersHandler
from . import tasks


class DataExtractor:
    def __init__(self, tokenizer: AutoTokenizer, config: Config):
        self.tokenizer = tokenizer
        self.config = config
        self.task = tasks.get_task(self.config.task, data_preprocessing=True, model=self.config.model)
        self.dataset = self.task.get_dataset()
    
    def _get_tokenize_tokens_and_immutable_identifiers(self, context: str, initial_immutable_identifiers: Set[str]):
        """Get the tokens and immutable identifiers"""
        try:
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
            elif self.config.lang == "python":
                input_stream = InputStream(context)
                lexer = Python3Lexer(input_stream)
                stream = CommonTokenStream(lexer)
                parser = Python3Parser(stream)
                # Get immutable identifiers
                immutable_identifiers = ImmutableIdentifiersHandler().get_immutable_identifiers(parser, context, initial_immutable_identifiers, self.config.lang)
                tokens = tokenize.generate_tokens(StringIO(context).readline)
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

            # Empty the immutable_identifiers if config.processing_mode is combined_token_operators
            if self.config.processing_mode == "combined_token_operators":
                immutable_identifiers = set()

            tokenize_tokens = []
            pos_count = 0
            whitespace_count = 0
            for idx, token in enumerate(tokens):
                if self.config.lang == "python":
                    whitespace_before = False
                    if token.type == tokenize.INDENT or token.type == tokenize.DEDENT or token.type == tokenize.ENDMARKER or token.string == '':
                        continue

                    # Check if current character is the whitespace 
                    while pos_count < len(context) and context[pos_count] == " ":
                        whitespace_count += 1
                        pos_count += 1
                        whitespace_before = True
                    
                    if whitespace_before:
                        whitespace_string = context[pos_count-whitespace_count:pos_count]
                        pos = f'({pos_count-whitespace_count}, {pos_count})'
                        tokenize_tokens.append({
                            'token_name': whitespace_string, 
                            'identifier': False, 
                            'operator': False, 
                            'type': 'WHITESPACE',
                            'pos': pos
                        })
                        whitespace_count = 0
                    
                    # Determine token type
                    is_operator = token.type == tokenize.OP
                    is_identifier = token.type == tokenize.NAME and token.string not in immutable_identifiers
                    
                    tokenize_tokens.append({
                        'token_name': token.string, 
                        'identifier': is_identifier, 
                        'operator': is_operator, 
                        'type': tokenize.tok_name[token.type],
                        'pos': f'({pos_count}, {pos_count+len(token.string)})'
                    })
                    pos_count += len(token.string)
                else:
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
                        'operator': is_operator,
                        'type': token.type,
                        'pos': f'({token.start}, {token.stop + 1})'
                    })

            return tokenize_tokens
        except Exception as e:
            print(f"Tokenize parsing failed: {e}. Skipping this example")
            return None
    
    def _get_LLM_tokens(self, context: str):
        """Get the tokens from the LLM"""
        try:
            tokens = self.tokenizer.tokenize(context)
            # This is a hard code for Qwen/CodeQwen1.5-7B-Chat
            if self.config.model == "Qwen/CodeQwen1.5-7B-Chat":
                tokens = [token.replace("<0x0A>", "\n") for token in tokens]
            return tokens
        except Exception as e:
            print(f"LLM tokenization failed: {e}. Skipping this example")
            return None
    
    def _get_target_identifiers(self, context, tokenize_tokens, LLM_tokens, test_case = None):
        """Get the multi-token identifiers from the context"""
        LLM_positions = [0]
        for token in LLM_tokens:
            LLM_positions.append(LLM_positions[-1] + len(token))
        
        # Get the target names from the tokenize tokens
        start_idx = 0
        end_idx = 1
        target_names = set()
        for token in tokenize_tokens:
            if token['identifier']:
                pos_start = int(token['pos'].split(',')[0].strip('('))
                pos_end = int(token['pos'].split(',')[1].strip(')')) - 1
                
                # Find corresponding LLM token range
                while start_idx < len(LLM_positions) - 1 and LLM_positions[start_idx + 1] <= pos_start:
                    start_idx += 1
                end_idx = start_idx + 1
                while end_idx < len(LLM_positions) and LLM_positions[end_idx] - 1 < pos_end:
                    end_idx += 1

                if end_idx - start_idx > 1:
                    target_names.add(token['token_name'])
        
        start_idx = 0
        end_idx = 1
        multi_token_identifiers = []
        for idx, token in enumerate(tokenize_tokens):
            if token['identifier']:
                # pos is in the format of (start, end)
                pos_start = int(token['pos'].split(',')[0].strip('('))
                pos_end = int(token['pos'].split(',')[1].strip(')')) - 1

                # Find corresponding LLM token range
                while start_idx < len(LLM_positions) - 1 and LLM_positions[start_idx + 1] <= pos_start:
                    start_idx += 1
                end_idx = start_idx + 1
                while end_idx < len(LLM_positions) and LLM_positions[end_idx] - 1 < pos_end:
                    end_idx += 1
                
                target_identifier = context[LLM_positions[start_idx]:LLM_positions[end_idx]]
                selected_LLM_tokens = LLM_tokens[start_idx:end_idx]
                selected_LLM_tokens_idx = f"({start_idx}, {end_idx})"
                tokenize_token = token['token_name']
                token_pos = f'({LLM_positions[start_idx]}, {LLM_positions[end_idx]})'
                
                # Get the multi-token identifier or single-token identifier but multi-token in other forms (e.g., with whitespace)
                if end_idx - start_idx > 1 or tokenize_token in target_names:
                    multi_token_identifiers.append({
                        'multi_token_identifier': target_identifier, 
                        'LLM_tokens': selected_LLM_tokens, 
                        'tokenize_token': tokenize_token, 
                        'pos': token_pos,
                        'LLM_tokens_idx': selected_LLM_tokens_idx
                    })
        
        test_target_identifiers = []
        # Get the test tokens for HumanEvalPack tasks
        if test_case is not None:
            test_tokens = self._get_test_tokens(test_case)
            for token in test_tokens:
                if token['identifier'] and token['token_name']in target_names:
                    test_target_identifiers.append(token)

        return multi_token_identifiers, test_target_identifiers
    
    def extract_new_identifiers(self):
        """Extract new multi-token identifiers"""
        all_multi_token_identifiers = []
        all_test_identifiers = []
        # Process each sample in the dataset
        for idx, sample in enumerate(self.dataset):
            if idx % 50 == 0:
                print(f"Processing example {idx} of {len(self.dataset)}...")
            # print(f"Processing example {idx} of {len(self.dataset)}...")

            # Get the context only from the dataset sample
            context = self.task.get_context_only(sample)

            # Get the immutable identifiers from the context
            immutable_identifiers = self.task.get_immutable_identifiers(sample)

            # Get the context tokens by tokenize library
            tokenize_tokens = self._get_tokenize_tokens_details(context, immutable_identifiers)

            # Get the LLM tokens by the LLM tokenizer
            LLM_tokens = self._get_LLM_tokens(context)

            # Get the multi-token identifiers from the context
            test_case = self.task.get_test_case(sample) if self.config.output_jsonl_file_name == "humanevalpack" else None
            multi_token_identifiers, test_target_identifiers = self._get_target_identifiers(context, tokenize_tokens, LLM_tokens, test_case)

            all_multi_token_identifiers.append(multi_token_identifiers)
            all_test_identifiers.append(test_target_identifiers)
        
        return all_multi_token_identifiers, all_test_identifiers

    def _get_combined_token_operators(self, context, tokenize_tokens, LLM_tokens):
        """Find combined token operators that match target combinations"""
        # Get the immutable identifiers from the context
        immutable_identifiers = set()  # No immutable identifiers for operators
        
        # Get LLM token positions
        LLM_positions = [0]
        for token in LLM_tokens:
            LLM_positions.append(LLM_positions[-1] + len(token))

        combined_token_operators = []
        start_idx = 0
        end_idx = 1
        following_start_idx = 0
        following_end_idx = 1

        for idx, token in enumerate(tokenize_tokens):
            if token['operator'] and token['token_name'] != ' ':
                # Get position of this operator token
                pos_start = int(token['pos'].split(',')[0].strip('('))
                pos_end = int(token['pos'].split(',')[1].strip(')')) - 1
                
                # Find corresponding LLM token range
                while start_idx < len(LLM_positions) - 1 and LLM_positions[start_idx + 1] <= pos_start:
                    start_idx += 1
                end_idx = start_idx + 1
                while end_idx < len(LLM_positions) and LLM_positions[end_idx] - 1 < pos_end:
                    end_idx += 1
                
                # Check if operator is combined with other tokens
                target_text = context[LLM_positions[start_idx]:LLM_positions[end_idx]]
                selected_LLM_tokens = LLM_tokens[start_idx:end_idx]
                selected_LLM_tokens_idx = f"({start_idx}, {end_idx})"
                
                # Look for following token
                following_token = None
                if idx + 1 < len(tokenize_tokens):
                    next_token = tokenize_tokens[idx + 1]
                    if next_token['type'] != 'WHITESPACE':
                        following_token = next_token
                
                # Check if this matches our target combinations
                operator_char = token['token_name']
                following_type = following_token['type'] if following_token else None
                following_name = following_token['token_name'] if following_token else None
                following_pos = following_token['pos'] if following_token else None
                
                
                matches_target = False
                for target_op, target_follow in self.config.target_combinations:
                    if target_op == 'OP':  # Any operator
                        if target_follow == 'NAME' and following_type == 'NAME':
                            matches_target = True
                            break
                        elif target_follow == 'ALL':
                            if following_type == 'NAME' or following_type == 'OP':
                                matches_target = True
                                break
                        elif following_name == target_follow:
                            matches_target = True
                            break
                    elif operator_char == target_op:
                        if target_follow == following_type or target_follow == following_name:
                            matches_target = True
                            break
                
                if matches_target:

                    following_pos_start = int(following_pos.split(',')[0].strip('('))
                    following_pos_end = int(following_pos.split(',')[1].strip(')')) - 1

                    while following_start_idx < len(LLM_positions) - 1 and LLM_positions[following_start_idx + 1] <= following_pos_start:
                        following_start_idx += 1
                    following_end_idx = following_start_idx + 1
                    while following_end_idx < len(LLM_positions) and LLM_positions[following_end_idx] - 1 < following_pos_end:
                        following_end_idx += 1

                    following_LLM_tokens = LLM_tokens[following_start_idx:following_end_idx]
                    following_LLM_tokens_idx = f"({following_start_idx}, {following_end_idx})"


                    is_combined_token = True
                    if following_pos_start > LLM_positions[end_idx] - 1:
                        is_combined_token = False

                    combined_token_operators.append({
                        'operator': operator_char,
                        'following_token': following_token,
                        'combined_text': target_text,
                        'LLM_tokens': selected_LLM_tokens,
                        'operator_pos': token['pos'],
                        'LLM_token_range': f'({LLM_positions[start_idx]}, {LLM_positions[end_idx]})',
                        'LLM_tokens_idx': selected_LLM_tokens_idx,
                        'following_pos': following_pos,
                        'following_LLM_tokens': following_LLM_tokens,
                        'following_LLM_tokens_idx': following_LLM_tokens_idx,
                        'is_combined_token': is_combined_token
                    })

        return combined_token_operators
    
    def extract_combined_token_operators(self):
        """Extract combined token operators"""
        all_combined_token_operators = []
        for idx, sample in enumerate(self.dataset):
            if idx % 50 == 0:
                print(f"Processing example {idx} of {len(self.dataset)}...")
            
            # Get the context only from the dataset sample
            context = self.task.get_context_only(sample)

            # Get the immutable identifiers from the context
            immutable_identifiers = self.task.get_immutable_identifiers(sample)

            # Get the context tokens by tokenize library
            tokenize_tokens = self._get_tokenize_tokens_details(context, immutable_identifiers)

            # Get the LLM tokens by the LLM tokenizer
            LLM_tokens = self._get_LLM_tokens(context)

            combined_token_operators = self._get_combined_token_operators(context, tokenize_tokens, LLM_tokens)

            all_combined_token_operators.append(combined_token_operators)
            
        return all_combined_token_operators


    

class DataGenerator:
    def __init__(self, tokenizer: AutoTokenizer, config: Config):
        self.config = config
        self.tokenizer = tokenizer
        self.task = tasks.get_task(config.task, data_preprocessing=True, model=config.model)
        self.dataset = self.task.get_dataset()

    def _get_LLM_tokens(self, context: str):
        """Get the tokens from the LLM"""
        try:
            tokens = self.tokenizer.tokenize(context)
            # This is a hard code for Qwen/CodeQwen1.5-7B-Chat
            if self.config.model == "Qwen/CodeQwen1.5-7B-Chat":
                tokens = [token.replace("<0x0A>", "\n") for token in tokens]
            return tokens
        except Exception as e:
            print(f"LLM tokenization failed: {e}. Skipping this example")
            return None

    def _get_LLM_token_positions(self, context):
        """Get the positions of the LLM tokens in the context"""
        LLM_tokens = self._get_LLM_tokens(context)
        LLM_positions = [0]
        for token in LLM_tokens:
            LLM_positions.append(LLM_positions[-1] + len(token))
        return LLM_positions

    def modify_LLM_positions(self, LLM_positions, new_LLM_positions, token_pos, difference_sign, split_positions):
        """Modify the LLM positions"""
        start_pos = int(token_pos.split(",")[0].strip('('))
        for idx in range(len(new_LLM_positions)):
            pos = LLM_positions[idx]
            for split_pos in split_positions:
                if pos > start_pos + split_pos:
                    new_LLM_positions[idx] += difference_sign
        return new_LLM_positions

    def process_multi_token_identifiers(self, multi_token_identifiers, context, declaration=None):
        """Process the multi-token identifiers"""
        new_context = context
        LLM_positions = self._get_LLM_token_positions(context)
        new_LLM_positions = LLM_positions.copy()

        # Get the unique multi-token identifiers from the multi-token identifiers
        unique_multi_token_identifiers = []
        for identifier in multi_token_identifiers:
            if identifier['tokenize_token'] not in unique_multi_token_identifiers:
                unique_multi_token_identifiers.append(identifier['tokenize_token'])
        
        # Filter the unique multi-token identifiers by the filter type (e.g., snake_case)
        unique_multi_token_identifiers = TxtDataFilter.generate_filtered_word_list(self.config.filter_type, unique_multi_token_identifiers)

        # Check if the unique multi-token identifiers are in the context
        selected_multi_token_identifiers = [] # List of selected multi-token identifiers that may influence the LLM tokenization
        total_difference = 0
        difference_declaration = 0
        for identifier in multi_token_identifiers:
            if identifier['tokenize_token'] in unique_multi_token_identifiers:
                new_target_identifier, split_positions = RegexProcessor.to_target_case(identifier['multi_token_identifier'], self.config.target_type)
                new_tokenize_token , _ = RegexProcessor.to_target_case(identifier['tokenize_token'], self.config.target_type)
                difference = len(new_target_identifier) - len(identifier['multi_token_identifier'])
                difference_sign = 1 if difference > 0 else (-1 if difference < 0 else 0)

                start_pos = int(identifier['pos'].split(",")[0].strip('('))
                end_pos = int(identifier['pos'].split(",")[1].strip(')'))

                # Modification for new LLM positions
                new_LLM_positions = self.modify_LLM_positions(LLM_positions, new_LLM_positions, identifier['pos'], difference_sign, split_positions)

                selected_multi_token_identifiers.append({'target_identifier': identifier['multi_token_identifier'], 'new_target_identifier': new_target_identifier, 'tokenize_token': identifier['tokenize_token'], 'new_tokenize_token': new_tokenize_token, 'pos': identifier['pos'], 'new_pos': f'({(start_pos + total_difference)}, {(end_pos + difference + total_difference)})', 'LLM_tokens': identifier['LLM_tokens'], 'LLM_tokens_idx': identifier['LLM_tokens_idx']})
                
                new_context = new_context[:start_pos + total_difference] + new_target_identifier + new_context[end_pos + total_difference:]
                total_difference += difference

                if declaration is not None:
                    if end_pos <= len(declaration):
                        difference_declaration += difference
        
        new_type_LLM_tokens = self._get_LLM_tokens(new_context)
        new_type_LLM_positions = self._get_LLM_token_positions(new_context)

        if len(selected_multi_token_identifiers) == 0:
            return None, None, False, None
        
        token_boundary_changed = False
        
        # Get the LLM tokens
        start_idx = 0
        end_idx = 0
        for identifier in selected_multi_token_identifiers:
            start_pos = int(identifier['new_pos'].split(",")[0].strip('('))
            end_pos = int(identifier['new_pos'].split(",")[1].strip(')'))

            for idx in range(end_idx, len(new_type_LLM_positions) - 1):
                if new_type_LLM_positions[idx + 1] > start_pos:
                    start_idx = idx
                    while new_type_LLM_positions[end_idx] < end_pos:
                        end_idx += 1         
                    
                    identifier['new_LLM_tokens'] = new_type_LLM_tokens[start_idx:end_idx]
                    identifier['new_LLM_tokens_idx'] = f"({start_idx}, {end_idx})"
                    break
            
            identifier['transformation'] = f"{identifier['LLM_tokens']} -> {identifier['new_LLM_tokens']}"

            # Delete '_' and make everything lowercase (both LLM tokens and new LLM tokens)
            LLM_tokens = [token.lstrip('_').lower() for token in identifier['LLM_tokens'] if token != '_']
            new_LLM_tokens = [token.lstrip('_').lower() for token in identifier['new_LLM_tokens'] if token != '_']
            identifier['length_difference'] = len(new_LLM_tokens) - len(LLM_tokens)

            is_fragment_changed_token = False
            if LLM_tokens != new_LLM_tokens:
                token_boundary_changed = True
                is_fragment_changed_token = True
            
            identifier['is_fragment_changed_token'] = is_fragment_changed_token
                

        new_declaration = new_context[:len(declaration) + difference_declaration] if declaration is not None else None

        return new_context, selected_multi_token_identifiers, token_boundary_changed, new_declaration
        

    def process_combined_token_operators(self, combined_token_operators, context):
        """Process combined token operators"""

        # Get the LLM tokens by the LLM tokenizer
        LLM_tokens = self._get_LLM_tokens(context)
        if not LLM_tokens:
            return None, None, None, None

        if not combined_token_operators:
            return None, None, None, None

        # Insert whitespace after operators
        new_context = context
        total_offset = 0
        processed_operators = []
        insertion_points = []
        has_combined_token = False

        LLM_positions = self._get_LLM_token_positions(context)

        for cto in combined_token_operators:
            operator_pos_start = int(cto['operator_pos'].split(',')[0].strip('('))
            operator_pos_end = int(cto['operator_pos'].split(',')[1].strip(')'))
            
            # Insert whitespace after the operator
            insertion_point = operator_pos_end + total_offset
            new_context = new_context[:insertion_point] + ' ' + new_context[insertion_point:]
            total_offset += 1

            # Modification for LLM positions
            for idx in range(len(LLM_positions)):
                if LLM_positions[idx] > insertion_point:
                    LLM_positions[idx] += 1

            if cto['is_combined_token']:
                insertion_points.append(insertion_point)
            
            processed_operators.append({
                'operator': cto['operator'],
                'original_pos': cto['operator_pos'],
                'insertion_point': insertion_point,
                'following_token': cto['following_token']['token_name'] if cto['following_token'] else None,
                'original_following_pos': cto['following_pos'],
                'is_combined_token': cto['is_combined_token'],
                'target_LLM_tokens_idx': cto['LLM_tokens_idx'],
                'LLM_tokens_idx': cto['following_LLM_tokens_idx']
            })

        # Check if tokenization changed
        new_LLM_tokens = self._get_LLM_tokens(new_context)
        new_LLM_positions = self._get_LLM_token_positions(new_context)

        start_idx = 0
        end_idx = 0
        difference = 0
        for cto in processed_operators:
            pos_start = int(cto['original_pos'].split(",")[0].strip('(')) + difference
            pos_end = int(cto['original_pos'].split(",")[1].strip(')')) - 1 + difference
            difference += 1
            while start_idx < len(new_LLM_positions) - 1 and new_LLM_positions[start_idx + 1] <= pos_start:
                start_idx += 1
            end_idx = start_idx + 1
            while end_idx < len(new_LLM_positions) and new_LLM_positions[end_idx] - 1 < pos_end:
                end_idx += 1
            
            cto['new_target_LLM_tokens_idx'] = f"({start_idx}, {end_idx})"

            following_pos_start = int(cto['original_following_pos'].split(",")[0].strip('(')) + difference
            following_pos_end = int(cto['original_following_pos'].split(",")[1].strip(')')) - 1 + difference
            while start_idx < len(new_LLM_positions) - 1 and new_LLM_positions[start_idx + 1] <= following_pos_start:
                start_idx += 1
            end_idx = start_idx + 1
            while end_idx < len(new_LLM_positions) and new_LLM_positions[end_idx] - 1 < following_pos_end:
                end_idx += 1
            
            cto['new_LLM_tokens_idx'] = f"({start_idx}, {end_idx})"

            # Get the LLM tokens indices
            baseline_LLM_tokens_idx = cto['LLM_tokens_idx']
            variant_LLM_tokens_idx = cto['new_LLM_tokens_idx']
            baseline_prev_LLM_tokens_idx = cto['target_LLM_tokens_idx']
            variant_prev_LLM_tokens_idx = cto['new_target_LLM_tokens_idx']

            baseline_LLM_tokens_idx_end = int(baseline_LLM_tokens_idx.split(",")[1].strip(')'))
            variant_LLM_tokens_idx_end = int(variant_LLM_tokens_idx.split(",")[1].strip(')'))
            baseline_prev_LLM_tokens_idx_start = int(baseline_prev_LLM_tokens_idx.split(",")[0].strip('('))
            variant_prev_LLM_tokens_idx_start = int(variant_prev_LLM_tokens_idx.split(",")[0].strip('('))
            variant_prev_LLM_tokens_idx_end = int(variant_prev_LLM_tokens_idx.split(",")[1].strip(')'))

            baseline_LLM_tokens = LLM_tokens[baseline_prev_LLM_tokens_idx_start:baseline_LLM_tokens_idx_end]
            variant_LLM_tokens = new_LLM_tokens[variant_prev_LLM_tokens_idx_start:variant_LLM_tokens_idx_end]
            cto['transformation'] = f"{baseline_LLM_tokens} -> {variant_LLM_tokens}"

            baseline_prev_LLM_tokens = LLM_tokens[baseline_prev_LLM_tokens_idx_start:baseline_LLM_tokens_idx_end]
            variant_prev_LLM_tokens = new_LLM_tokens[variant_prev_LLM_tokens_idx_start:variant_prev_LLM_tokens_idx_end]
            cto['transformation_prev_tokens'] = f"{baseline_prev_LLM_tokens} -> {variant_prev_LLM_tokens}"

            cto['length_difference'] = len(variant_LLM_tokens) - len(baseline_LLM_tokens)

            is_fragment_changed_token = False
            if LLM_positions[baseline_prev_LLM_tokens_idx_start:baseline_LLM_tokens_idx_end] != new_LLM_positions[variant_prev_LLM_tokens_idx_start:variant_LLM_tokens_idx_end]:
                is_fragment_changed_token = True
            cto['is_fragment_changed_token'] = is_fragment_changed_token

        fragment_changed_type = "remained"
        
        new_LLM_positions_modified = [idx for idx in new_LLM_positions if idx not in insertion_points]
        merged_positions = [idx for idx in LLM_positions if idx not in new_LLM_positions_modified]
        split_positions = [idx for idx in new_LLM_positions_modified if idx not in LLM_positions]

        if len(merged_positions) > 0 and len(split_positions) == 0:
            fragment_changed_type = "merged"
        elif len(merged_positions) == 0 and len(split_positions) > 0:
            fragment_changed_type = "split"
        elif len(merged_positions) > 0 and len(split_positions) > 0:
            fragment_changed_type = "mixed"

        if fragment_changed_type != "remained":
            token_boundary_changed = True
        else:
            token_boundary_changed = False

        return new_context, processed_operators, token_boundary_changed, fragment_changed_type

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
    
    def process_entry_point(self, entry_point):
        filtered_identifiers = []
        filtered_identifiers.append(entry_point)
        filtered_identifiers = TxtDataFilter.generate_filtered_word_list(self.config.filter_type, filtered_identifiers)
        if len(filtered_identifiers) == 0:
            return entry_point
        
        new_entry_point, _ = RegexProcessor.to_target_case(entry_point, self.config.target_type)
        return new_entry_point

    def process_all_multi_token_identifiers(self, all_multi_token_identifiers, all_test_identifiers):
        """Process all multi-token identifiers"""
        new_contexts = {} 
        all_selected_multi_token_identifiers = {}      
        all_token_boundary_changed = {}
        all_modified_tests = {} if self.config.output_jsonl_file_name == "humanevalpack" else None
        all_modified_entry_points = {} if self.config.output_jsonl_file_name == "humanevalpack" else None
        all_modified_declarations = {} if self.config.output_jsonl_file_name == "humanevalpack" else None
        for idx, multi_token_identifiers in enumerate(all_multi_token_identifiers):
            context = self.task.get_context_only(self.dataset[idx])
            test_case = self.task.get_test_case(self.dataset[idx]) if self.config.output_jsonl_file_name == "humanevalpack" else None
            entry_point = self.task.get_entry_point(self.dataset[idx]) if self.config.output_jsonl_file_name == "humanevalpack" else None
            declaration = self.task.get_declaration(self.dataset[idx]) if self.config.output_jsonl_file_name == "humanevalpack" else None
            if multi_token_identifiers:
                new_context, selected_multi_token_identifiers, token_boundary_changed, new_declaration = self.process_multi_token_identifiers(multi_token_identifiers, context, declaration)
                if new_context:
                    new_idx = self.task.get_task_id(self.dataset[idx])
                    new_contexts[new_idx] = new_context

                    all_selected_multi_token_identifiers[new_idx] = selected_multi_token_identifiers
                    all_token_boundary_changed[new_idx] = token_boundary_changed

                    if test_case:
                        test_target_identifiers = all_test_identifiers[idx]
                        new_test_case = self.process_test_identifiers(test_case, test_target_identifiers)
                        all_modified_tests[new_idx] = new_test_case

                    if entry_point:
                        new_entry_point = self.process_entry_point(entry_point)
                        all_modified_entry_points[new_idx] = new_entry_point

                    if declaration:
                        all_modified_declarations[new_idx] = new_declaration

        return new_contexts, all_selected_multi_token_identifiers, all_token_boundary_changed, all_modified_tests, all_modified_entry_points, all_modified_declarations
    
    def process_all_combined_token_operators(self, all_combined_token_operators):
        """Process all combined token operators"""
        new_contexts = {}
        all_processed_operators = {}
        all_token_boundary_changed = {}
        all_fragment_changed_types = {}

        for idx, combined_token_operators in enumerate(all_combined_token_operators):
            
            context = self.task.get_context_only(self.dataset[idx])
            new_context, processed_operators, token_boundary_changed, fragment_changed_type = self.process_combined_token_operators(combined_token_operators, context)

            if new_context:
                task_id = self.task.get_task_id(self.dataset[idx])
                new_contexts[task_id] = new_context
                all_processed_operators[task_id] = processed_operators
                all_fragment_changed_types[task_id] = fragment_changed_type
                all_token_boundary_changed[task_id] = token_boundary_changed

        return new_contexts, all_processed_operators, all_token_boundary_changed, all_fragment_changed_types


    def generate_new_dataset(self, new_contexts, all_modified_tests=None, all_modified_entry_points=None, all_modified_declarations=None):
        """Generate new dataset with modified contexts"""
        new_dataset = self.dataset
        
        # Check if dataset is a list or a datasets.Dataset object
        if isinstance(new_dataset, list):
            # Handle list case
            filtered_data = []
            for sample in new_dataset:
                task_id = self.task.get_task_id(sample)
                if task_id in new_contexts:
                    # Create a copy of the sample and add modified context
                    modified_sample = sample.copy() if isinstance(sample, dict) else sample
                    if isinstance(modified_sample, dict):
                        modified_sample['modified_context'] = new_contexts[task_id]
                        if all_modified_tests:
                            modified_sample['modified_test'] = all_modified_tests[task_id]
                        if all_modified_entry_points:
                            modified_sample['modified_entry_point'] = all_modified_entry_points[task_id]
                        if all_modified_declarations:
                            modified_sample['modified_declaration'] = all_modified_declarations[task_id]
                    else:
                        # If sample is not a dict, we need to handle it differently
                        # This shouldn't happen in typical cases, but let's be safe
                        modified_sample = {
                            'original_sample': sample,
                            'modified_context': new_contexts[task_id]
                        }
                        if all_modified_tests:
                            modified_sample['modified_test'] = all_modified_tests[task_id]
                        if all_modified_entry_points:
                            modified_sample['modified_entry_point'] = all_modified_entry_points[task_id]
                        if all_modified_declarations:
                            modified_sample['modified_declaration'] = all_modified_declarations[task_id]
                    filtered_data.append(modified_sample)
            
            # Save the new dataset as a JSONL file
            os.makedirs(os.path.dirname(self.config.data_generator_output_dataset), exist_ok=True)
            
            with open(self.config.data_generator_output_dataset, 'w', encoding='utf-8') as f:
                for sample in filtered_data:
                    f.write(json.dumps(sample) + '\n')
            
            print(f"Saved {len(filtered_data)} samples to {self.config.data_generator_output_dataset}")
            
        else:
            # Handle Dataset object case
            # Create a filtered dataset containing only samples with modified contexts
            filtered_dataset = new_dataset.filter(
                lambda sample: self.task.get_task_id(sample) in new_contexts
            )
            
            # Update the samples with their modified contexts
            def add_modified_context(sample):
                task_id = self.task.get_task_id(sample)
                sample['modified_context'] = new_contexts[task_id]
                if all_modified_tests:
                    sample['modified_test'] = all_modified_tests[task_id]
                if all_modified_entry_points:
                    sample['modified_entry_point'] = all_modified_entry_points[task_id]
                if all_modified_declarations:
                    sample['modified_declaration'] = all_modified_declarations[task_id]
                return sample
            
            filtered_dataset = filtered_dataset.map(add_modified_context)
            
            # Save the new dataset as a JSONL file
            os.makedirs(os.path.dirname(self.config.data_generator_output_dataset), exist_ok=True)
            
            with open(self.config.data_generator_output_dataset, 'w', encoding='utf-8') as f:
                for sample in filtered_dataset:
                    f.write(json.dumps(sample) + '\n')
            
            print(f"Saved {len(filtered_dataset)} samples to {self.config.data_generator_output_dataset}")


        
if __name__ == "__main__":
    import argparse
    from .config_iterator import ConfigIterator

    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", default=False)
    parser.add_argument("--all_tasks", action="store_true", default=False)
    parser.add_argument("--all_multi_token_identifiers", action="store_true", default=False)
    parser.add_argument("--all_target_combinations", action="store_true", default=False)
    parser.add_argument("--process_avatar", action="store_true", default=False)
    args = parser.parse_args()

    # Initialize config
    config = Config()

    if args.all:
        args.all_tasks = True
        args.all_multi_token_identifiers = True
        args.all_target_combinations = True

    if args.process_avatar:
        config.all_tasks = ["avatartranslate-python2java", "avatartranslate-java2python"]
        args.all_tasks = True
        args.all_multi_token_identifiers = True
        args.all_target_combinations = True

    # Use ConfigIterator for main processing loop
    iterator = ConfigIterator(config)

    for task, model, processing_mode, tokenizer in iterator.iterate_all(
        all_tasks=args.all_tasks,
        all_models=False,
        all_multi_token_identifiers=args.all_multi_token_identifiers,
        all_combined_token_operators=args.all_target_combinations,
        generate_dataset=True
    ):
        # Initialize data processor and generator
        data_extractor = DataExtractor(tokenizer, config)
        data_generator = DataGenerator(tokenizer, config)

        # Naming rewrites
        if processing_mode == "multi_token_identifiers":
            # Extract identifiers
            print(f"Extracting identifiers...")
            all_multi_token_identifiers, all_test_identifiers = data_extractor.extract_new_identifiers()

            # Generate new multi-token dataset
            print(f"Generating new multi-token dataset...")
            new_contexts, all_selected_multi_token_identifiers, all_token_boundary_changed, all_modified_tests, all_modified_entry_points, all_modified_declarations = data_generator.process_all_multi_token_identifiers(all_multi_token_identifiers, all_test_identifiers)

            # Generate dataset file
            if new_contexts:
                data_generator.generate_new_dataset(new_contexts, all_modified_tests, all_modified_entry_points, all_modified_declarations)
            else:
                print("No contexts were modified. No data generated.")

        # Spacing rewrites
        elif processing_mode == "combined_token_operators":
            # Extract combined token operators
            print(f"Extracting combined token operators...")
            all_combined_token_operators = data_extractor.extract_combined_token_operators()

            # Process combined token operators
            print(f"Finding and processing combined token operators...")
            new_contexts, all_processed_operators, all_token_boundary_changed, all_fragment_changed_types = data_generator.process_all_combined_token_operators(all_combined_token_operators)

            # Generate dataset file
            if new_contexts:
                data_generator.generate_new_dataset(new_contexts)
            else:
                print("No contexts were modified. No data generated.")