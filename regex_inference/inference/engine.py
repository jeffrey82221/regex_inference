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

    def run(self, patterns: List[str], regex: Optional[str] = None) -> str:
        assert len(
            patterns) > 0, '`patterns` input to `run` should no be an empty list'
        if regex is None:
            regex = self._run(patterns)
        regex = self._run_simplify_regex(regex, patterns)
        failed_patterns = self._get_mismatching_patterns(patterns, regex)
        while failed_patterns:
            failed_regex = self._run(failed_patterns)
            regex = f'{regex}|{failed_regex}'
            regex = self._run_simplify_regex(regex, patterns)
            failed_patterns = self._get_mismatching_patterns(patterns, regex)
        return regex

    def _run(self, patterns: List[str], regex: Optional[str] = None) -> str:
        if regex:
            regex_result = self._run_alter_regex(regex, patterns)
        else:
            regex_result = self._run_new_inference(patterns)
        return regex_result

    @staticmethod
    def _convert_patterns_to_prompt(patterns: List[str]) -> str:
        return '\n'.join(map(lambda x: f'"{x}"', patterns))

    def _run_alter_regex(self, regex: str, patterns: List[str]) -> str:
        for _ in range(self._max_iteration):
            result = self._regex_alter_chain.run(
                regex=regex,
                strings=Engine._convert_patterns_to_prompt(patterns)
            ).strip()
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
                strings=Engine._convert_patterns_to_prompt(patterns)
            ).strip()
            try:
                re.compile(result)
                break
            except BaseException:
                pass
        return result

    def _run_new_inference(self, patterns: List[str]) -> str:
        for _ in range(self._max_iteration):
            result = self._new_inference_chain.run(
                Engine._convert_patterns_to_prompt(patterns)
            ).strip()
            try:
                re.compile(result)
                break
            except BaseException:
                pass
        return result

    def explain(self, regex: str) -> None:
        result = self._regex_explain_chain.run(regex)
        print(result)

    def _get_mismatching_patterns(
            self, patterns: List[str], result_regex: str) -> List[str]:
        try:
            re_com = re.compile(result_regex)
        except BaseException as e:
            print('syntax error in result_regex:', result_regex)
            raise e
        result = list(filter(lambda x: re_com.fullmatch(x) is None, patterns))
        return result

    def _setup_lang_chains(self):
        self._regex_alter_chain = LLMChain(
            prompt=self.alter_regex_prompt,
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
        self._regex_explain_chain = LLMChain(
            prompt=self.explain_regex_prompt,
            llm=self._openai_llm
        )

    @property
    def new_inference_prompt(self) -> PromptTemplate:
        template = """Question: Show me the best and shortest regex that can fully match the strings that I provide to you.
Note that:
*. The regex should be made more generalized (e.g., use \\d to represent digit rather than using [0-9]) and shorter than the original regex.
*. Match sure the resulting regex does not have syntax error.
*. The character count of the resulting regex should not be larger than 30.
*. Use \\d to replace [0-9].
*. Try to focus more on the global pattern rather than the local patterns.
Now, each instance of the strings that should be fully matched is provided line-by-line and wrapped by double quotes as follows:
{strings}

Note that:
1. The double quote is not part of the string instance. Ignore the double quote during inferencing the regex.
2. Provide the resulting regex without wrapping it in quote

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
Now, each instance of the strings is provided line-by-line and wrapped by double quotes as follows:
{strings}

Note that:
1. The double quote is not part of the string instance. Ignore the double quote during inferencing the regex.
2. Provide the resulting regex without wrapping it in quote

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
*. It still fully match each of the strings I provided to you.
Now, each instance of the strings is provided line-by-line and wrapped by double quotes as follows:
{strings}


Note that:
1. The double quote is not part of the string instance. Ignore the double quote during inferencing the regex.
2. Provide the resulting regex without wrapping it in quote

The resulting revise regex is:
"""
        prompt = PromptTemplate(
            template=template,
            input_variables=['regex', 'strings']
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
