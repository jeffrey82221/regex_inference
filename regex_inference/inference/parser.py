import re
from typing import List

class Parser:
    @staticmethod
    def convert_patterns_to_prompt(patterns: List[str]) -> str:
        return '\n'.join(map(lambda x: f'"{x}"', patterns))
    
    @staticmethod
    def select_regex_from_result(result: str) -> str:
        for regex in result.split('\n'):
            if len(regex.strip()) > 0:
                re.compile(regex)
                return regex.strip()