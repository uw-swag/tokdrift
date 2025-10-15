from antlr4.tree.Tree import TerminalNode
from .grammars import JavaLexer, JavaParser
from .grammars import Python3Lexer, Python3Parser
from antlr4 import InputStream, CommonTokenStream


class ImmutableIdentifiersHandler:
    """
    A class to handle immutable identifiers detection for different programming languages.
    Uses language-specific parsers and logical rules to identify identifiers that should
    remain immutable during code transformations.
    """
    
    def __init__(self):
        """Initialize the handler."""
        pass
    
    def get_immutable_identifiers(self, parser, context, initial_immutable_identifiers, lang):
        """
        Entry point method to get immutable identifiers based on language.
        
        Args:
            parser: The ANTLR4 parser
            context: The source code as string
            initial_immutable_identifiers: Set of identifiers that are known to be immutable
            lang: Programming language ('java', 'cpp', 'python')
            
        Returns:
            tuple: (immutable_identifiers_set, declarations_set)
        """
        lang = lang.lower()
        
        if lang == 'java':
            immutable_identifiers, declarations = self.get_java_immutable_identifiers(parser, context)
            immutable_identifiers.difference_update(declarations)
            immutable_identifiers.update(initial_immutable_identifiers)
        elif lang == 'python':
            immutable_identifiers, declarations = self.get_python_immutable_identifiers(parser, context)
            immutable_identifiers.difference_update(declarations)
            immutable_identifiers.update(initial_immutable_identifiers)
        else:
            raise ValueError(f"Unsupported language: {lang}")

        return immutable_identifiers
    
    def get_java_immutable_identifiers(self, parser, context):
        """
        Use rules based on syntactic context to identify immutable identifiers in Java code.
        
        Args:
            parser: The ANTLR4 parser
            context: The source code as string
            
        Returns:
            tuple: (immutable_identifiers_set, declarations_set)
        """
        
        unchanging = set()
        declarations = set()
        
        def has_override_annotation_simple(method_node, grandparent_node, grandgrandparent_node, greatgrandparent_node):
            """Simple text-based check for @Override annotation"""
            # Simple approach: check if @Override appears in the source code before this method
            try:
                if hasattr(method_node, 'start') and method_node.start:
                    method_start = method_node.start.start
                    # Look backwards in the source for @Override
                    search_start = max(0, method_start - 50)
                    text_before = context[search_start:method_start]
                    return '@Override' in text_before
                return False
            except:
                return False
        
        def analyze_node(node, parent=None, grandparent=None, grandgrandparent=None):
            """Recursively analyze parse tree nodes"""
            if isinstance(node, TerminalNode):
                token = node.getSymbol()
                if token.type == JavaLexer.IDENTIFIER:
                    identifier = token.text
                    
                    # Get the grammatical context
                    node_rule = parent.getRuleIndex() if parent and hasattr(parent, 'getRuleIndex') else None
                    parent_rule = grandparent.getRuleIndex() if grandparent and hasattr(grandparent, 'getRuleIndex') else None
                    grandparent_rule = grandgrandparent.getRuleIndex() if grandgrandparent and hasattr(grandgrandparent, 'getRuleIndex') else None

                    parent_rule_name = parser.ruleNames[parent_rule] if parent_rule is not None else None
                    grandparent_rule_name = parser.ruleNames[grandparent_rule] if grandparent_rule is not None else None
                    
                    # Rule 1: Import statements - all identifiers in imports are unchanging
                    if parent_rule_name in ['importDeclaration', 'qualifiedName'] or grandparent_rule_name == 'importDeclaration':
                        unchanging.add(identifier)
                    
                    # Rule 2: Type references in variable declarations, method parameters, return types
                    elif parent_rule_name in ['typeType', 'classOrInterfaceType', 'primitiveType']:
                        unchanging.add(identifier)
                    
                    # Rule 3: Method calls - check if it's a method being called
                    elif parent_rule_name == 'methodCall':
                        # If this identifier is the method name in a method call, it might be unchanging
                        # We need to check if it's being called on a standard type
                        unchanging.add(identifier)
                    
                    # Rule 4: Qualified names (package.Class, object.method)
                    elif parent_rule_name == 'qualifiedName':
                        # In qualified names, both parts are often unchanging
                        unchanging.add(identifier)
                    
                    # Rule 5: Constructor calls (new ClassName())
                    elif parent_rule_name == 'creator' or grandparent_rule_name == 'creator':
                        unchanging.add(identifier)
                    
                    # Rule 6: Annotation names
                    elif parent_rule_name in ['annotation', 'qualifiedName'] and grandparent_rule_name == 'annotation':
                        unchanging.add(identifier)

                    # Rule 7: Expression
                    elif parent_rule_name == 'expression':
                        unchanging.add(identifier)
                    
                    # Rule 8: Explicit generic invocation suffix
                    elif parent_rule_name == 'explicitGenericInvocationSuffix':
                        unchanging.add(identifier)
                    
                    # Rule 9: Class declarations
                    elif parent_rule_name == 'classDeclaration':
                        declarations.add(identifier)
                    
                    # Rule 10: Method declarations
                    elif parent_rule_name == 'methodDeclaration':
                        # Special case: if method has @Override annotation, it should be unchanging
                        has_override = has_override_annotation_simple(parent, grandparent, grandgrandparent, None)
                        if has_override:
                            unchanging.add(identifier)
                        else:
                            declarations.add(identifier)
            
            # Recursively process children
            if hasattr(node, 'getChildCount'):
                for i in range(node.getChildCount()):
                    child = node.getChild(i)
                    analyze_node(child, parent=node, grandparent=parent, grandgrandparent=grandparent)
            
        # Start the analysis
        analyze_node(parser.compilationUnit())
        
        return unchanging, declarations
    
    def get_python_immutable_identifiers(self, parser, context):
        """
        Use rules based on syntactic context to identify immutable identifiers in Python code.
        
        Args:
            parser: The ANTLR4 parser
            context: The source code as string
            
        Returns:
            tuple: (immutable_identifiers_set, declarations_set)
        """
        
        unchanging = set()
        declarations = set()
        
        # Python built-in functions and keywords that should be unchanging
        python_builtins = {
            'print', 'len', 'range', 'str', 'int', 'float', 'bool', 'list', 'dict', 'tuple', 'set',
            'open', 'input', 'type', 'isinstance', 'hasattr', 'getattr', 'setattr', 'delattr',
            'min', 'max', 'sum', 'abs', 'round', 'sorted', 'reversed', 'enumerate', 'zip',
            'map', 'filter', 'any', 'all', 'iter', 'next', 'repr', 'ord', 'chr', 'hex', 'oct',
            'bin', 'id', 'hash', 'eval', 'exec', 'compile', 'format', 'globals', 'locals',
            'vars', 'dir', 'help', 'callable', 'classmethod', 'staticmethod', 'property',
            'super', 'object', 'Exception', 'BaseException'
        }
        
        def analyze_node(node, parent=None, grandparent=None, grandgrandparent=None):
            """Recursively analyze parse tree nodes"""
            if isinstance(node, TerminalNode):
                token = node.getSymbol()
                if token.type == Python3Lexer.NAME:
                    identifier = token.text
                    
                    # Get the grammatical context
                    node_rule = parent.getRuleIndex() if parent and hasattr(parent, 'getRuleIndex') else None
                    parent_rule = grandparent.getRuleIndex() if grandparent and hasattr(grandparent, 'getRuleIndex') else None
                    grandparent_rule = grandgrandparent.getRuleIndex() if grandgrandparent and hasattr(grandgrandparent, 'getRuleIndex') else None
                    
                    parent_rule_name = parser.ruleNames[parent_rule] if parent_rule is not None else None
                    grandparent_rule_name = parser.ruleNames[grandparent_rule] if grandparent_rule is not None else None
                    
                    # Rule 1: Built-in functions and types
                    if identifier in python_builtins:
                        unchanging.add(identifier)
                    
                    # Rule 2: Import statements - all identifiers in imports are unchanging
                    elif parent_rule_name in ['import_as_name', 'import_as_names', 'import_from', 'dotted_as_names', 'dotted_name']:
                        unchanging.add(identifier)
                    
                    # Rule 3: Attribute access - method calls on objects
                    elif parent_rule_name in ['trailer']:
                        # This could be a method call like obj.method()
                        unchanging.add(identifier)
                    
                    # Rule 4: Exception types in except clauses
                    elif parent_rule_name == 'except_clause':
                        unchanging.add(identifier)
                    
                    # Rule 5: Decorator names
                    elif parent_rule_name == 'decorator':
                        unchanging.add(identifier)
                    
                    # Rule 6: Class definitions
                    elif parent_rule_name == 'classdef':
                        declarations.add(identifier)
                    
                    # Rule 7: Function definitions
                    elif parent_rule_name == 'funcdef':
                        declarations.add(identifier)
            
            # Recursively process children
            if hasattr(node, 'getChildCount'):
                for i in range(node.getChildCount()):
                    child = node.getChild(i)
                    analyze_node(child, parent=node, grandparent=parent, grandgrandparent=grandparent)
        
        # Start the analysis
        analyze_node(parser.file_input())
        
        return unchanging, declarations 