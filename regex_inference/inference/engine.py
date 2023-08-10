"""
TODO:
- [ ] Evaluation Metrics Design
    - [ ] F1-Score / Precision / Recall should work on whole Regex
        - [ ] Precision's base is random generated strings.
    - [ ] Using Accuracy to quantize the non-overlapping explainability of sub-regex.
        - [ ] ((# correctly match of positive patterns) + (# correctly mismatch of negative patterns)) / (# all patterns)
- [ ] Consider continual inferencing mode: statistics should evaluate on the future cases.
- [ ] Add LLMChain to fix the regex with low F1 scores.
"""
import typing
from typing import List, Optional, Callable, Any
import re
from .filter import Filter
from .chain import Chain
from ..utils import make_verbose


class Engine:
    def __init__(self, openai_api_key: Optional[str] = None, temperature: float = 0.8,
                 mismatch_tolerance: float = 0.1, max_iteration: int = 3, simpify_regex: bool = True, verbose: bool = False):
        self._chain = Chain(
            openai_api_key=openai_api_key,
            temperature=temperature)
        self._mismatch_tolerance = mismatch_tolerance
        self._max_iteration = max_iteration
        self._simpify_regex = simpify_regex
        if verbose:
            self._make_verbose()

    @typing.no_type_check
    def _make_verbose(self):
        self.run = make_verbose(self.run)
        self._run_new_inference = make_verbose(self._run_new_inference)

    def run(self, patterns: List[str]) -> str:
        regex_list = self.get_regex_sequence(patterns)
        return Engine.merge_regex_sequence(regex_list)

    def get_regex_sequence(self, patterns: List[str]) -> List[str]:
        assert len(
            patterns) > 0, '`patterns` input to `run` should no be an empty list'
        regex_list = [self._run_new_inference(patterns)]
        mismatched_patterns = Filter.mismatch(
            Engine.merge_regex_sequence(regex_list),
            patterns
        )
        while mismatched_patterns:
            regex = self._run_new_inference(mismatched_patterns)
            regex_list.append(regex)
            mismatched_patterns = Filter.mismatch(
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
            result = self._chain.alter_regex.run(
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
            result = self._chain.simplify_regex.run(
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
            result = self._chain.inference_regex.run(
                Engine._convert_patterns_to_prompt(patterns)
            ).strip()
            try:
                re.compile(result)
                break
            except BaseException:
                pass
        return result

    def explain(self, regex: str) -> None:
        result = self._chain.explain_regex.run(regex)
        print(result)
