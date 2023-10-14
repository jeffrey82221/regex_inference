from typing import List
from more_itertools import chunked
import random
from .engine import Engine
from .fado import FAdoEngine, FAdoAIEngine
from .candidate import Candidate, CandidateRecords

__all__ = ['Inference']


class Inference:
    def __init__(self, *args, **kwargs):
        if 'engine' in kwargs:
            if kwargs['engine'] == 'fado+ai':
                del kwargs['engine']
                self._engine = FAdoAIEngine(*args, **kwargs)
            elif kwargs['engine'] == 'ai':
                del kwargs['engine']
                self._engine = Engine(*args, **kwargs)
            elif kwargs['engine'] == 'fado':
                del kwargs['engine']
                self._engine = FAdoEngine(*args, **kwargs)
        else:
            self._engine = FAdoAIEngine(*args, **kwargs)
        if 'verbose' in kwargs:
            self._verbose = kwargs['verbose']
        else:
            self._verbose = False

    def run(self, train_patterns: List[str],
            val_patterns: List[str] = [], n_fold: int = 10, train_rate: float = 1.) -> str:
        """
        Args:
            - train_patterns: The patterns to be infered from.
            - val_patterns: The validation patterns for selecting the best regex.
                If not provided, validation data would be randomly and non-repetively selected from training data.
            - n_fold: repeating validation count.
            - train_rate: Ratio of training data selected for each fold (used only when val_patterns is not provided.)
        Return:
            - regex: The infered regex
        """
        if val_patterns:
            regex = self._infer_with_fix_val_patterns(
                train_patterns, val_patterns, n_fold=n_fold)
        else:
            regex = self._infer_with_cross_val_patterns(
                train_patterns, n_fold=n_fold, total_train_rate=train_rate * n_fold)
        return regex

    def _infer_with_fix_val_patterns(
            self, train_patterns: List[str], val_patterns: List[str], n_fold: int) -> str:
        candidates = []
        for _ in range(n_fold):
            candidate = Candidate(
                self._engine,
                train_patterns,
                val_patterns)
            candidates.append(candidate)
        return CandidateRecords(candidates).get_best()

    def _infer_with_cross_val_patterns(
            self, train_patterns: List[str], n_fold: int, total_train_rate: float) -> str:
        selected_train_count = int(len(train_patterns) * total_train_rate)
        train_buckets = Inference._get_train_buckets(
            train_patterns, selected_train_count, n_fold)
        candidates = []
        for i in range(n_fold):
            val_bucket = list(set(train_patterns) - set(train_buckets[i]))
            candidate = Candidate(
                self._engine,
                train_buckets[i],
                val_bucket)
            candidates.append(candidate)
        return CandidateRecords(candidates).get_best()

    @staticmethod
    def _get_train_buckets(
            train_patterns: List[str], bucket_size: int, n_fold: int) -> List[List[str]]:
        if bucket_size <= len(train_patterns):
            train_selected = random.sample(
                train_patterns, bucket_size)
        else:
            train_selected = random.choices(
                train_patterns, k=bucket_size)
        train_buckets = list(
            chunked(
                train_selected,
                bucket_size //
                n_fold))
        return train_buckets
