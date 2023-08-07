from typing import List, Tuple
from .inference import Filter
from .inference import Engine


class Evaluator:
    @staticmethod
    def evaluate_regex_list(
            regex_list: List[str], patterns: List[str]) -> Tuple[float, float, float]:
        regex = Engine.merge_regex_sequence(regex_list)
        divided_patterns = Evaluator._divide_patterns(regex_list, patterns)
        recall = Evaluator.recall(regex, patterns)
        precisions = []
        for i in range(len(divided_patterns)):
            negative_patterns = []
            for not_i in [j for j in range(len(divided_patterns)) if j != i]:
                negative_patterns.extend(divided_patterns[not_i])
            precision = Evaluator.precision(
                regex_list[i], divided_patterns[i], negative_patterns)
            precisions.append(precision)
        precision = sum(precisions) / len(precisions)
        f1 = 2. / (1. / precision + 1. / recall)
        return precision, recall, f1

    @staticmethod
    def _divide_patterns(regex_list: List[str],
                         patterns: List[str]) -> List[List[str]]:
        results = []
        for regex in regex_list:
            results.append(Filter.match(regex, patterns))
            patterns = Filter.mismatch(regex, patterns)
        return results

    @staticmethod
    def recall(regex: str, patterns: List[str]) -> float:
        """
        Recall evaluate how well the regex capture the patterns presented.

        Args:
            - regex: whole regex consists of multiple sub-regex
            - patterns: the patterns in the future or not presented but should be captured by the regex.
        """
        return len(Filter.match(regex, patterns)) / len(patterns)

    @staticmethod
    def precision(
            sub_regex: str, positive_patterns: List[str], negative_patterns: List[str]) -> float:
        """
        Precision evaluate how precise or explainable is the regex on the target patterns.

        Because my goal is that each sub-regex should exactly match its target patterns,
        the positive patterns and negative patterns for the sub-regex is defined as follows:

        * positive_patterns: pattern presented previously and matched by the sub-regex
        * negative_patterns: pattern not hosted by the sub-regex.
        """
        if positive_patterns:
            return len(Filter.match(sub_regex, positive_patterns)) / \
                len(Filter.match(sub_regex,
                                 positive_patterns + negative_patterns))
        else:
            return 0.0
