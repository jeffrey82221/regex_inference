from typing import List, Tuple, Iterator
from threading import Thread
# from multiprocessing import Process as Thread
from multiprocessing import Queue
from more_itertools import chunked
from langchain.callbacks import get_openai_callback
import random
from .engine import Engine
from .fado import FAdoEngine, FAdoAIEngine
from ..evaluator import Evaluator

__all__ = ['Inference']


class Candidate(Thread):
    def __init__(self, engine, train_patterns, val_patterns, queue):
        Thread.__init__(self)
        self.value = None
        self._engine = engine
        self._train_patterns = train_patterns
        self._val_patterns = val_patterns
        self._q = queue

    def run(self):
        regex_list = self._engine.get_regex_sequence(
            self._train_patterns)
        _, _, f1 = Evaluator.evaluate_regex_list(
            regex_list, self._val_patterns)
        self.value = (f1, regex_list)
        self._q.put((f1, regex_list))

    @staticmethod
    def get_best(candidates: List['Candidate']) -> str:
        for worker in candidates:
            worker.start()
        for worker in candidates:
            worker.join()
        regex_list = [Candidate.get_value(worker) for worker in candidates]
        sorted_results = sorted(regex_list, key=lambda x: x[0], reverse=True)
        return Engine.merge_regex_sequence(sorted_results[0][1])

    @staticmethod
    def get_value(candidate) -> Tuple[float, List[str]]:
        value = candidate.value
        if value is None:
            value = candidate._q.get()
        return value


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
        self._regex_candidate = []

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
    
    def process_batch(self, train_batch: List[str], n_fold: int = 10, train_rate: float = 1.):
        """
        TODO:
        - [ ] Refactor: build class `CandidateRecords` for holding candidates related data with following methods:
            - [ ] `get_best_regex` (with metrics for ordering as variables)
            - [ ] `get_regex_candidates` (get regex list from the hold candidates)
            - [ ] `join` enable combine of two set of candidates
            - [ ] `drop_bad_regex` (with number and metric variables guiding the drop)
        - [ ] Feature: add with get_openai_callback() as cb: for recording the price statistics. 
            - [ ] get_chatgpt_summary: https://python.langchain.com/docs/modules/model_io/models/llms/token_usage_tracking
                cb.total_tokens
                cb.prompt_tokens
                cb.completion_tokens
                cb.total_cost
        - [ ] Experiment: compare price usage and f1 result between `ai` and `fado+ai` approach. 
        - [ ] Feature: Build new class `ContinuousInference` with `run` procedure: 
            - [ ] Step1: select `train_rate` percentage of instances from train_batch.
                - [ ] If len(candidates) == 0: select randomly
                - [ ] Else: select by low f1 with high f1-std
            - [ ] Step2: Do _run_new_inference `n_fold` times (procedure similar to self._infer_with_cross_val_patterns), 
                    keep the resulting candidates.  
            - [ ] Step3: Get all combination of regex concate from the old candidates and new candidates.
            - [ ] Step4: Calculate the F1 scores of all regex concates, retain the top `n_fold` onces as new candidates and drop the rest.
        """
        

    def _infer_with_fix_val_patterns(
            self, train_patterns: List[str], val_patterns: List[str], n_fold: int) -> str:
        candidates = []
        for _ in range(n_fold):
            candidate = Candidate(
                self._engine,
                train_patterns,
                val_patterns,
                Queue())
            candidates.append(candidate)
        result = Candidate.get_best(candidates)
        self._regex_candidate = [Candidate.get_value(e)[1] for e in candidates]
        return result

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
                val_bucket,
                Queue())
            candidates.append(candidate)
        result = Candidate.get_best(candidates)
        self._regex_candidate = [Candidate.get_value(e)[1] for e in candidates]
        return result

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
