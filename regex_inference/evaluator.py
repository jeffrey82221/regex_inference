from typing import List, Tuple
import numpy as np
from .inference import Filter
from .inference import Engine


class Evaluator:
    """
    Evaluate a regex against the ground truth patterns
    """
    @staticmethod
    def evaluate(
            regex: str, patterns: List[str]) -> Tuple[float, float, float]:
        """
        Evaluate the performance of a regex against a list of patterns.

        Args:
            - regex: regex to be evaluated
            - patterns: patterns to be matched by the regex
        Returns:
            - precision: describe how well the regex describe the patterns
            - recall: describe how well the regex captures the patterns
            - f1: combined score for precision and recall
        """
        return Evaluator.evaluate_regex_list([regex], patterns)

    @staticmethod
    def evaluate_regex_list(
            regex_list: List[str], patterns: List[str]) -> Tuple[float, float, float]:
        """
        Evaluate the performance of a regex (represented by a regex_list) against a list of patterns.

        Args:
            - regex_list: regex to be evaluated
            - patterns: patterns to be matched by the regex_list
        Returns:
            - precision: describe how well each regex in the regex list describe the patterns
            - recall: describe how well the entire regex list match the patterns
            - f1: combined score for precision and recall
        """

        recall = Evaluator.recall(regex_list, patterns)
        precision = Evaluator.precision(regex_list, patterns)
        if recall == 0. or precision == 0.:
            f1 = 0.
        else:
            f1 = 2. / (1. / precision + 1. / recall)
        return precision, recall, f1

    @staticmethod
    def precision(regex_list: List[str], patterns: List[str]) -> float:
        """
        Precision evaluate how precise or explainable is the regex_list on the target patterns.
        The better each sub-regex uniquely matchs its corresponding pattern,
        the higher is the precision.

        Args:
            - regex_list: regex to be evaluated
            - patterns: patterns to be matched by the regex_list
        Returns:
            - precision
        """
        divided_patterns = Engine.divide_patterns(regex_list, patterns)
        precisions = []
        for i in range(len(divided_patterns)):
            negative_patterns = Evaluator._collect_negative_patterns(
                i, divided_patterns)
            precision = Evaluator._regex_precision(
                regex_list[i], divided_patterns[i], negative_patterns)
            precisions.append(precision)
        precision = sum(precisions) / len(precisions)
        return precision

    @staticmethod
    def recall(regex_list: List[str], patterns: List[str]) -> float:
        """
        Recall evaluate how well the regex capture the patterns presented.

        Args:
            - regex: whole regex consists of multiple sub-regex
            - patterns: the patterns in the future or not presented but should be captured by the regex.
        Returns:
            - recall
        """
        regex = Engine.merge_regex_sequence(regex_list)
        return len(Filter.match(regex, patterns)) / len(patterns)

    @staticmethod
    def _collect_negative_patterns(
            target_regex_index: int, divided_patterns: List[List[str]]) -> List[str]:
        negative_patterns = []
        for not_i in [j for j in range(
                len(divided_patterns)) if j != target_regex_index]:
            negative_patterns.extend(divided_patterns[not_i])
        return negative_patterns

    @staticmethod
    def _regex_precision(
            sub_regex: str, positive_patterns: List[str], negative_patterns: List[str]) -> float:
        """
        Args:
           - sub_regex: a regex within the regex_list
           - positive_patterns: pattern presented and matched by the sub-regex
           - negative_patterns: pattern not matched by the sub-regex.
        """
        if positive_patterns:
            return len(Filter.match(sub_regex, positive_patterns)) / \
                len(Filter.match(sub_regex,
                                 positive_patterns + negative_patterns))
        else:
            return 0.0

    @staticmethod
    def get_variation_score(
            patterns: List[str], regex_list_group: List[List[str]]) -> List[float]:
        """
        Calculate the performance variation for a list of patterns against an ensemble of regex.

        Args:
            - patterns: a list of patterns.
            - regex_list_group: an ensemble of regex list for calculating the variation score.
        """
        results = []
        for pattern in patterns:
            scores = [
                Evaluator.evaluate_regex_list(
                    regex_list, [pattern]
                ) for regex_list in regex_list_group]
            precisions = [score[0] for score in scores]
            recalls = [score[1] for score in scores]
            results.append(np.std(precisions) + np.std(recalls))
        return results

    @staticmethod
    def get_performance_score(
            patterns: List[str], regex_list_group: List[List[str]]) -> List[float]:
        """
        Calculate the average performance score for a list of patterns against an ensemble of regex.

        Args:
            - patterns: a list of patterns.
            - regex_list_group: an ensemble of regex list for calculating the average performance score.
        """
        results = []
        for pattern in patterns:
            f1_scores = [
                Evaluator.evaluate_regex_list(
                    regex_list, [pattern]
                )[-1] for regex_list in regex_list_group]
            results.append(np.mean(f1_scores))
        return results
