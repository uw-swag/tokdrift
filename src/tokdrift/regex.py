import re

class RegexProcessor:
    @staticmethod
    def is_camel_case(text: str) -> str:
        """Check is a string is in camelCase format"""
        # pattern = r'[a-z]+((\d)|([A-Z0-9][a-z0-9]+))*([A-Z])?'
        pattern = r'[a-z]+(?:[A-Z]+[A-Za-z0-9]+[A-Za-z0-9]*)+'
        return bool(re.match(pattern, text))

    @staticmethod
    def is_pascal_case(text: str) -> str:
        """
        Check is a string is in PascalCase format
        For example: XmlHttpRequest, NewCustomerId, InnerStopwatch, SupportsIpv6OnIos, YouTubeImporter, YoutubeImporter, Affine3D
        """
        # This pattern matches strings that start with an uppercase letter, contain only letters and numbers, and contain at least one lowercase letter and at least one other uppercase letter.
        pattern = r'[A-Z]([A-Z0-9]*[a-z][a-z0-9]*[A-Z]|[a-z0-9]*[A-Z][A-Z0-9]*[a-z])[A-Za-z0-9]*'
        return bool(re.match(pattern, text))

    @staticmethod
    def is_snake_case(text: str) -> str:
        """Check is a string is in snake_case format"""
        pattern = r'[a-z0-9]+(?:_[A-Za-z0-9]+)+'
        return bool(re.match(pattern, text))
    
    @staticmethod
    def is_screaming_snake_case(text: str) -> str:
        """Check is a string is in screaming_snake_case format"""
        pattern = r'[A-Z0-9]+(?:_[A-Z0-9]+)+'
        return bool(re.match(pattern, text))

    @staticmethod
    def separate_case(text: str) -> tuple[list, list]:
        """
        Separate a string into parts based on case boundaries.
        Examples:
        - paren_string -> ['paren', 'string']
        - parenString -> ['paren', 'string']
        - HTTPConnection -> ['HTTP', 'Connection']
        """
        if not text:
            return []
        
        # The regex matches positions where we should split
        pattern = r'(?<=[_$])(?!$)|(?<!^)(?=[_$])|(?<=[a-z])(?=[A-Z0-9])|(?<=[A-Z])(?=[0-9]|[A-Z][a-z0-9])|(?<=[0-9])(?=[a-zA-Z])'
        
        # Use re.finditer to find all positions
        split_positions = [match.start() for match in re.finditer(pattern, text)]
        
        # Use these positions to split the string
        parts = []
        prev_pos = 0
        for pos in split_positions:
            parts.append(text[prev_pos:pos])
            prev_pos = pos
        parts.append(text[prev_pos:])
        
        # Filter out any special characters
        parts = [part for part in parts if part and part not in ['_', '$']]

        # Check if the text is starting with a character, if so, remove the first character
        if not text[0].isalpha():
            text = text[1:]

        if RegexProcessor.is_screaming_snake_case(text) or RegexProcessor.is_snake_case(text):
            target_split_positions = [split_positions[i] for i in range(len(split_positions)) if i % 2 == 0]
        else:
            target_split_positions = split_positions
        
        return parts, target_split_positions

    @staticmethod
    def to_camel_case(text: str) -> tuple[str, list]:
        """
        Convert any case format to camelCase.
        Example: 'paren_string' -> 'parenString'
        """
        if not text:
            return ''
        parts, split_positions = RegexProcessor.separate_case(text)
        if not parts:
            return ''
        return parts[0].lower() + ''.join(part.capitalize() for part in parts[1:]), split_positions

    @staticmethod
    def to_pascal_case(text: str) -> tuple[str, list]:
        """
        Convert any case format to PascalCase.
        Example: 'paren_string' -> 'ParenString'
        """
        if not text:
            return ''
        parts, split_positions = RegexProcessor.separate_case(text)
        # Check if the first character of the first part is a whitespace, if so, remove it
        if not parts[0][0].isalpha():
            parts[0] = parts[0][0] + parts[0][1].upper() + parts[0][2:]
            return ''.join(parts[0]) + ''.join(part.capitalize() for part in parts[1:]), split_positions
        return ''.join(part.capitalize() for part in parts), split_positions

    @staticmethod
    def to_snake_case(text: str) -> tuple[str, list]:
        """
        Convert any case format to snake_case.
        Example: 'parenString' -> 'paren_string'
        """
        if not text:
            return ''
        parts, split_positions = RegexProcessor.separate_case(text)
        return '_'.join(part.lower() for part in parts), split_positions

    @staticmethod
    def to_screaming_snake_case(text: str) -> tuple[str, list]:
        """
        Convert any case format to SCREAMING_SNAKE_CASE.
        Example: 'parenString' -> 'PAREN_STRING'
        """
        if not text:
            return ''
        parts, split_positions = RegexProcessor.separate_case(text)
        return '_'.join(part.upper() for part in parts), split_positions
    
    @staticmethod
    def to_target_case(text: str, target_type: str) -> tuple[str, list]:
        """
        Convert any case format to the target case.
        Example: 'parenString' -> 'paren_string'
        """
        if not text:
            return ''
        if target_type == "camel_case":
            return RegexProcessor.to_camel_case(text)
        elif target_type == "pascal_case":
            return RegexProcessor.to_pascal_case(text)
        elif target_type == "snake_case":
            return RegexProcessor.to_snake_case(text)
        elif target_type == "screaming_snake_case":
            return RegexProcessor.to_screaming_snake_case(text)
        else:
            raise ValueError(f"Invalid target type: {target_type}")


if __name__ == "__main__":
    # print(RegexProcessor.is_camel_case("(isPalindrome"))
    # print(Regex.is_pascal_case("HTTPConnection"))
    # print(Regex.is_screaming_snake_case("UPPER"))
    # print(RegexProcessor.is_snake_case(" is_palindrome"))
    # print(RegexProcessor.to_snake_case("HTTPConnection"))
    # print(RegexProcessor.to_camel_case("current_string"))
    print(RegexProcessor.to_pascal_case(" is_palindrome"))
    # print(RegexProcessor.to_camel_case(" is_palindrome"))
    # print(RegexProcessor.to_snake_case("getCurrentString"))