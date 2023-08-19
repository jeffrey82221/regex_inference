import re
from typing import List
from langchain import PromptTemplate
from langchain import LLMChain
from .parser import Parser

__all__ = ['Chain']


class Chain:
    def __init__(self, llm):
        self._llm = llm
        self._setup_lang_chains()

    def _setup_lang_chains(self):
        self._inference_regex = LLMChain(
            prompt=self.new_inference_prompt,
            llm=self._llm
        )
        self._explain_regex = LLMChain(
            prompt=self.explain_regex_prompt,
            llm=self._llm
        )
        self._fix_regex = LLMChain(
            prompt=self.fix_regex_prompt,
            llm=self._llm
        )

    def fix(self, regex: str, correct_patterns: List[str], incorrect_patterns: List[str]):

        fact_0_str = f"""

A regex to be fix is double quoted and shown as the follows:

"{regex}"
    """
        fact_1_str = f"""

It correctly match the patterns double quoted and shown as follows:
{Parser.convert_patterns_to_prompt(correct_patterns)}
However, it mistakenly match the patterns double quoted and shown as follows:
{Parser.convert_patterns_to_prompt(incorrect_patterns)}
        """

        
        result = self._fix_regex.run(
            facts=f"""
{fact_0_str}

{fact_1_str}
        """
        )
        return result

    def inference(self, patterns):
        strings = Parser.convert_patterns_to_prompt(patterns)
        result = self._inference_regex.run(strings)
        regex = Parser.select_regex_from_result(result)
        return regex
    
    def explain(self, regex: str) -> None:
        result = self._explain_regex.run(regex)
        print(result)

    @property
    def new_inference_prompt(self) -> PromptTemplate:
        template = """Question: Show me the best and shortest regex that can fully match the strings that I provide to you.
Note that:
*. The regex should be as short as possible.
*. Match sure the resulting regex does not have syntax error.
*. The regex should full match as many strings as possible.
*. The regex should not match strings that is not provided.
*. The number of string combinations matching the resulting regex should be as smaller than the number of target strings provided.
Now, each instance of the strings that should be fully matched is provided line-by-line and wrapped by double quotes as follows:
{strings}

Note that:
1. The double quote is not part of the string instance. Ignore the double quote during inferencing the regex.
2. Provide the resulting regex without wrapping it in quote
3. Do not provide any other text besides the regex in the first line. After the first line, there should be only lines with space characters.

The resulting regex is: """
        prompt = PromptTemplate(
            template=template,
            input_variables=['strings']
        )
        return prompt

    @property
    def explain_regex_prompt(self) -> PromptTemplate:
        template = """Question: Explain the regex "{regex}" such that
1. The role of each character in the regex is elaberated.
2. Provide 5 most interpretive example strings that fullmatch the regex.

The explaination is: """
        prompt = PromptTemplate(
            template=template,
            input_variables=['regex']
        )
        return prompt

    @property
    def fix_regex_prompt(self) -> PromptTemplate:
        template = """
{facts}
Question: What is the altered regex that meet the following criteria?
1. It correctly matches the patterns that is correctly match.
2. It excludes the pattern mistakenly matched. That is, those mistakenly match patterns should not be matched.

Example Answer:

The regex: 

[2-9]

Answer:
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=['facts']
        )
        return prompt
