from typing import List
from threading import Thread
from more_itertools import chunked
import random
from .engine import Engine
from .fado import FAdoEngine, FAdoAIEngine
from ..evaluator import Evaluator


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
        class GetThread(Thread):
            def __init__(self, engine):
                Thread.__init__(self)
                self.value = None
                self._engine = engine

            def run(self):
                regex_list = self._engine.get_regex_sequence(train_patterns)
                _, _, f1 = Evaluator.evaluate_regex_list(
                    regex_list, val_patterns)
                self.value = (f1, regex_list)
        threads = []
        for _ in range(n_fold):
            thread = GetThread(self._engine)
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        regex_list = [thread.value for thread in threads]
        if self._verbose:
            print('regex_list candidates:', regex_list)
        sorted_results = sorted(regex_list, key=lambda x: x[0], reverse=True)
        regex_list = sorted_results[0][1]
        return Engine.merge_regex_sequence(regex_list)

    def _infer_with_cross_val_patterns(
            self, train_patterns: List[str], n_fold: int, total_train_rate: float) -> str:
        class GetThread(Thread):
            def __init__(self, engine, train_patterns, val_patterns):
                Thread.__init__(self)
                self.value = None
                self._engine = engine
                self._train_patterns = train_patterns
                self._val_patterns = val_patterns

            def run(self):
                regex_list = self._engine.get_regex_sequence(
                    self._train_patterns)
                _, _, f1 = Evaluator.evaluate_regex_list(
                    regex_list, self._val_patterns)
                self.value = (f1, regex_list)

        selected_train_count = int(len(train_patterns) * total_train_rate)
        if selected_train_count <= len(train_patterns):
            train_selected = random.sample(
                train_patterns, selected_train_count)
        else:
            train_selected = random.choices(
                train_patterns, k=selected_train_count)
        train_buckets = list(
            chunked(
                train_selected,
                selected_train_count //
                n_fold))
        threads = []
        for i in range(n_fold):
            val_bucket = list(set(train_patterns) - set(train_buckets[i]))
            thread = GetThread(self._engine, train_buckets[i], val_bucket)
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        regex_list = [thread.value for thread in threads]
        if self._verbose:
            print('regex_list candidates:', regex_list)
        sorted_results = sorted(regex_list, key=lambda x: x[0], reverse=True)
        regex_list = sorted_results[0][1]
        return Engine.merge_regex_sequence(regex_list)
