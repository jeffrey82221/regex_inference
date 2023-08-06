from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain import LLMChain
from typing import List, Optional
import re
import os


class Engine:
    def __init__(self, openai_api_key: Optional[str] = None, temparature: float = 0.8,
                 mismatch_tolerance: float = 0.1, max_iteration: int = 3):
        if openai_api_key is None:
            openai_api_key = os.environ["OPENAI_API_KEY"]
        self._openai_llm = OpenAI(
            openai_api_key=openai_api_key,
            temperature=temparature,
            model='text-davinci-003',  # https://platform.openai.com/docs/models/gpt-3-5
            client='regex_inference'
        )
        self._mismatch_tolerance = mismatch_tolerance
        self._max_iteration = max_iteration
        self._setup_lang_chains()

    def _setup_lang_chains(self):
        self._regex_alter_chain = LLMChain(
            prompt=self.alter_regex_prompt,
            llm=self._openai_llm
        )
        self._regex_revise_chain = LLMChain(
            prompt=self.revise_regex_prompt,
            llm=self._openai_llm
        )
        self._new_inference_chain = LLMChain(
            prompt=self.new_inference_prompt,
            llm=self._openai_llm
        )
        self._regex_simplify_chain = LLMChain(
            prompt=self.simplify_regex_prompt,
            llm=self._openai_llm
        )

    def run(self, patterns: List[str], regex: Optional[str] = None) -> str:
        if regex is None:
            regex = self._run(patterns)
        regex = self._run_simplify_regex(regex, patterns)
        failed_patterns = self._get_failed_patterns(patterns, regex)
        while failed_patterns:
            failed_regex = self._run(failed_patterns)
            regex = f'{regex}|{failed_regex}'
            regex = self._run_simplify_regex(regex, patterns)
            failed_patterns = self._get_failed_patterns(patterns, regex)
        return regex

    def _run(self, patterns: List[str], regex: Optional[str] = None) -> str:
        if regex:
            regex_result = self._run_alter_regex(regex, patterns)
        else:
            regex_result = self._run_new_inference(patterns)
        regex_result = self._revise_regex(patterns, regex_result)
        return regex_result

    def _run_alter_regex(self, regex: str, patterns: List[str]) -> str:
        for _ in range(self._max_iteration):
            result = self._regex_alter_chain.run(
                regex=regex,
                strings='\n'.join(patterns)).strip()
            try:
                re.compile(result)
                break
            except BaseException:
                pass
        return result

    def _run_simplify_regex(self, regex: str, patterns: List[str]) -> str:
        for _ in range(self._max_iteration):
            result = self._regex_simplify_chain.run(
                regex=regex,
                strings='\n'.join(patterns)).strip()
            try:
                re.compile(result)
                break
            except BaseException:
                pass
        return result

    def _run_new_inference(self, patterns: List[str]) -> str:
        for _ in range(self._max_iteration):
            result = self._new_inference_chain.run('\n'.join(patterns)).strip()
            try:
                re.compile(result)
                break
            except BaseException:
                pass
        return result

    def _run_revise_regex(self, regex: str, patterns: List[str]) -> str:
        for _ in range(self._max_iteration):
            result = self._regex_revise_chain.run(
                regex=regex,
                strings='\n'.join(patterns)).strip()
            try:
                re.compile(result)
                break
            except BaseException:
                pass
        return result

    def _revise_regex(self, patterns: List[str], regex: str) -> str:
        mismtach_patterns = self._get_failed_patterns(patterns, regex)
        iter_count = 0
        while len(mismtach_patterns) / \
                len(patterns) >= self._mismatch_tolerance:
            regex = self._run_revise_regex(regex, patterns)
            mismtach_patterns = self._get_failed_patterns(patterns, regex)
            iter_count += 1
            if iter_count > self._max_iteration:
                break
        return regex

    def _get_failed_patterns(
            self, patterns: List[str], result_regex: str) -> List[str]:
        try:
            re_com = re.compile(result_regex)
        except BaseException as e:
            print('syntax error in result_regex:', result_regex)
            raise e
        result = list(filter(lambda x: re_com.fullmatch(x) is None, patterns))
        return result

    @property
    def new_inference_prompt(self) -> PromptTemplate:
        template = """Question: Show me the best and shortest regex that can fully match the strings that I provide to you.
Note that:
*. The regex should be made more generalized (e.g., use \\d to represent digit rather than using [0-9]) and shorter than the original regex.
*. Match sure the resulting regex does not have syntax error.
*. The character count of the resulting regex should not be larger than 30.
*. Use \\d to replace [0-9].
*. Try to focus more on the global pattern rather than the local patterns.
Now, the patterns should be fully matched is provided line-by-line as follows:
{strings}

Note: Provide the resulting regex without wrapping it in quote
The resulting regex is: """
        prompt = PromptTemplate(
            template=template,
            input_variables=['strings']
        )
        return prompt

    @property
    def alter_regex_prompt(self) -> PromptTemplate:
        template = """Question: Alter the regex "{regex}" such that the following requirements is matched:
*. The pattern fully match the regex still fully match the regex.
*. The strings that I provide to you also fully match.
*. The character count of the resulting regex should not be larger than 30.
*. The regex should be made more generalized (e.g., use \\d to represent digit rather than using [0-9]) and shorter than the original regex.
Each string of the strings is provided line-by-line as follows:
{strings}

Note: Provide the resulting regex without wrapping it in quote
The resulting altered regex is: """
        prompt = PromptTemplate(
            template=template,
            input_variables=['regex', 'strings']
        )
        return prompt

    @property
    def simplify_regex_prompt(self) -> PromptTemplate:
        template = """
Please revise the regex "{regex}"
such that
*. It becomes as short as possible.
*. It still fully match all the strings full matched the original regex
*. It still fully match each of the following strings:

{strings}

Note: Provide the resulting regex without wrapping it in quote
The resulting revise regex is:
"""
        prompt = PromptTemplate(
            template=template,
            input_variables=['regex', 'strings']
        )
        return prompt

    @property
    def revise_regex_prompt(self) -> PromptTemplate:
        template = """Question: Revise the regex "{regex}" such that the following requirements is matched:
*. The patterns fully match the regex still fully match the revised regex.
*. The regex should be made more generalized (e.g., use \\d to represent digit rather than using [0-9]) and shorter than the original regex.
*. Match sure the resulting regex does not have syntax error.
*. Use \\d to replace [0-9].
*. Try to focus more on the global pattern rather than the local patterns.
*. The character count of the resulting regex should not be larger than 30.
*. The regex should be revised such that the provided mis-matched strings should be fully matched.

Each string of the mis-matched strings is provided line-by-line as follows:
{strings}

Note: Provide the resulting regex without wrapping it in quote
The resulting altered regex is: """
        prompt = PromptTemplate(
            template=template,
            input_variables=['regex', 'strings']
        )
        return prompt

    @property
    def explain_regex_prompt(self) -> PromptTemplate:
        template = """Question: Explain the regex "{regex}" such that the role of each character in the regex is elaberated

The explaination is: """
        prompt = PromptTemplate(
            template=template,
            input_variables=['regex']
        )
        return prompt
