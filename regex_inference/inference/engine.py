"""
TODO:
- [ ] Consider continual inferencing mode: statistics should evaluate on the future cases.
- [ ] Add LLMChain to fix the regex with low F1 scores.
"""
import typing
from typing import List, Optional, Callable, Any, Dict
import re
import os
import exrex
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain import LLMChain


def make_verbose(func: Callable) -> Callable:
    def warp(*args: Any, **kwargs: Any) -> Any:
        args_str = str(args)
        kwargs_str = str(kwargs)
        if len(args_str) > 30:
            args_str = args_str[:10] + '...' + args_str[-10:]
        if len(kwargs_str) > 30:
            kwargs_str = kwargs_str[:10] + '...' + kwargs_str[-10:]
        print(
            f'[{func.__name__}]',
            f'START with input -- args: {args_str}; kwargs: {kwargs_str}')
        result = func(*args, **kwargs)
        print(f'END [{func.__name__}]')
        return result
    return warp


class Engine:
    def __init__(self, openai_api_key: Optional[str] = None, temparature: float = 0.8,
                 mismatch_tolerance: float = 0.1, max_iteration: int = 3, simpify_regex: bool = True, verbose: bool = False):
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
        self._simpify_regex = simpify_regex
        if verbose:
            self._make_verbose()
        self._setup_lang_chains()

    @typing.no_type_check
    def _make_verbose(self):
        self.run = make_verbose(self.run)
        self._run_simplify_regex = make_verbose(
            self._run_simplify_regex)
        self._run_alter_regex = make_verbose(
            self._run_alter_regex)
        self._run_new_inference = make_verbose(
            self._run_new_inference)

    def run(self, patterns: List[str]) -> str:
        regex_list = self.get_regex_sequence(patterns)
        return Engine.merge_regex_sequence(regex_list)

    @staticmethod
    @typing.no_type_check
    def get_statistics(
            patterns: List[str], regex_list: List[str]) -> List[Dict]:
        """
        Args:
            patterns: list of strings to be inferenced
            regex_list: the regex sequenced inferenced by ChatGPT.
        Returns:
            results (list of dict):
                result (dict with fields):
                    - regex
                    - n_sim_patterns (number of strings matching the regex)
                    - n_matched_patterns
        """
        total_cnt = len(patterns)
        results: List[Dict] = []
        previous_matched: List[str] = []
        for i in range(len(regex_list)):
            result = dict()
            result['regex'] = regex_list[i]
            result['n_sim_patterns'] = exrex.count(result['regex'])
            matched = Engine.filter_match(result['regex'], patterns)
            result['n_matched_patterns'] = len(
                set(matched) - set(previous_matched))
            result['n_target_matching'] = total_cnt
            total_cnt -= result['n_matched_patterns']
            previous_matched.extend(matched)
            previous_matched = list(set(previous_matched))
            result['precision'] = result['n_matched_patterns'] / \
                result['n_sim_patterns']
            result['recall'] = result['n_matched_patterns'] / \
                result['n_target_matching']
            if result['precision'] == 0. or result['recall'] == 0.:
                result['f1'] = 0.
            else:
                result['f1'] = 2. / \
                    (1. / result['precision'] + 1. / result['recall'])
            results.append(result)
        return results

    def get_regex_sequence(self, patterns: List[str]) -> List[str]:
        assert len(
            patterns) > 0, '`patterns` input to `run` should no be an empty list'
        regex_list = [self._run_new_inference(patterns)]
        mismatched_patterns = Engine.filter_mismatch(
            Engine.merge_regex_sequence(regex_list),
            patterns
        )
        while mismatched_patterns:
            regex = self._run_new_inference(mismatched_patterns)
            regex_list.append(regex)
            mismatched_patterns = Engine.filter_mismatch(
                Engine.merge_regex_sequence(regex_list), patterns)
        return regex_list

    @staticmethod
    def merge_regex_sequence(regex_list: List[str]) -> str:
        return '|'.join(map(lambda x: f'({x})', regex_list))

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

    @staticmethod
    def filter_match(regex: str, patterns: List[str]) -> List[str]:
        try:
            re_com = re.compile(regex)
        except BaseException as e:
            print('syntax error in result_regex:', regex)
            raise e
        result = list(
            filter(
                lambda x: re_com.fullmatch(x) is not None,
                patterns))
        return result

    @staticmethod
    def filter_mismatch(
            regex: str, patterns: List[str]) -> List[str]:
        try:
            re_com = re.compile(regex)
        except BaseException as e:
            print('syntax error in result_regex:', regex)
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
*. The regex should full match as many strings provided as possible.
*. The regex should be as short as possible.
*. The regex should not match strings that is not provided except for those full match the original regex.
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
such that the following constraint start with *. can be met:
*. The original regex consists of multiple regex seperated by "|". Try to combine the similar regex.
*. After combine, the resulting regex should be as short as possible.
*. The revised regex should still fully match all the strings full matched the original regex
*. The revised regex should still fully match each of the strings I provided to you.
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
