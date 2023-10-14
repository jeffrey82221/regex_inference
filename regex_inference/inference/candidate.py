from typing import List, Optional
from threading import Thread
from scipy import stats
import numpy as np
# from multiprocessing import Process as Thread
from multiprocessing import Queue
from ..evaluator import Evaluator
from .engine import Engine


class Candidate(Thread):
    """
    Generate a candidate using inference engine
    """

    def __init__(self, engine: Optional[Engine], train_patterns: List[str], val_patterns: List[str], queue: Queue = Queue(
    ), value: Optional[List[str]] = None, score: Optional[float] = None):
        Thread.__init__(self)
        self._engine = engine
        self._train_patterns = train_patterns
        self._val_patterns = val_patterns
        self._q = queue
        self._value = value
        self._score = score

    def run(self):
        regex_list = self._engine.get_regex_sequence(
            self._train_patterns)
        _, _, f1 = Evaluator.evaluate_regex_list(
            regex_list, self._val_patterns)
        self._value = regex_list
        self._score = f1
        self._q.put((f1, regex_list))

    @property
    def value(self) -> List[str]:
        value = self._value
        if value is None:
            value = self._q.get()[1]
        return value

    @property
    def score(self) -> float:
        value = self._score
        if value is None:
            value = self._q.get()[0]
        return value

    def get_score(self) -> float:
        if self._score is not None:
            return self._score
        else:
            _, _, f1 = Evaluator.evaluate_regex_list(
                self.value, self._val_patterns)
            self._score = f1
            return self._score

    def hash(self):
        return hash(self.value)

    def __repr__(self):
        return self.value


class CandidateRecords:
    """
    Holder of regex candidates
    """

    def __init__(self, candidates: List[Candidate], run=True):
        self._candidates = candidates
        if run:
            self._run_inference()
        worker_list = [(worker, worker.get_score())
                       for worker in self._candidates]
        self._candidates = [
            e[0] for e in sorted(
                worker_list,
                key=lambda x: x[1],
                reverse=True)]

    def _run_inference(self):
        for worker in self._candidates:
            worker.start()
        for worker in self._candidates:
            worker.join()

    def get_best(self) -> str:
        return Engine.merge_regex_sequence(self._candidates[0].value)

    @property
    def candidates(self) -> List[List[str]]:
        return [c.value for c in self._candidates]

    @property
    def scores(self) -> List[float]:
        return [c.score for c in self._candidates]

    def drop_bad(self, n_drop: int):
        assert n_drop < len(
            self._candidates), 'You can not drop too many candidates (at least one candidate should remain in the record).'
        retain_cnt = len(self._candidates) - n_drop
        self._candidates = self._candidates[:retain_cnt]

    def __or__(self, other: 'CandidateRecords') -> 'CandidateRecords':
        new_obj = CandidateRecords(self._candidates, run=False)
        new_obj._candidates = list(
            set(new_obj._candidates) | set(other._candidates))
        worker_list = [(worker, worker.score)
                       for worker in new_obj._candidates]
        new_obj._candidates = [
            e[0] for e in sorted(
                worker_list,
                key=lambda x: x[1],
                reverse=True)]
        return new_obj

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CandidateRecords):
            return NotImplemented
        return self.candidates == other.candidates

    def __add__(self, other):
        new_candidates = [
            Candidate(
                None,
                c1._train_patterns + c2._train_patterns,
                list(set(c1._val_patterns + c2._val_patterns) -
                     set(c1._train_patterns + c2._train_patterns)),
                value=c1._value + c2._value
            ) for c1 in self._candidates for c2 in other._candidates]
        for c in new_candidates:
            c._score = c.get_score()
        return CandidateRecords(new_candidates, run=False)

    def sort_by_benefit(self, patterns: List[str]) -> List[str]:
        """
        Inplace sort of training patterns by their benefit on
        continous training.

        Rule:
            - performances: smaller the better
            - variations: larger the better

        Args:
            - patterns: the patterns to be sorted
        Returns:
            - results: the sorted patterns
        """
        performances = Evaluator.get_performance_score(
            patterns, self.candidates)
        variations = Evaluator.get_variation_score(patterns, self.candidates)
        interest_score = stats.zscore(np.array(variations)) - stats.zscore(np.array(performances))
        results = list(map(
            lambda x: x[1],
            sorted(zip(interest_score, patterns),
                   reverse=True, key=lambda x: x[0])
        ))
        return results
