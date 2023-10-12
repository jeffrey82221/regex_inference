from typing import List, Optional
from threading import Thread
# from multiprocessing import Process as Thread
from multiprocessing import Queue
from ..evaluator import Evaluator
from .engine import Engine


class Candidate(Thread):
    """
    Generate a candidate using inference engine
    """
    def __init__(self, engine: Engine, train_patterns: List[str], val_patterns: List[str], queue: Queue = Queue()):
        Thread.__init__(self)
        self._engine = engine
        self._train_patterns = train_patterns
        self._val_patterns = val_patterns
        self._q = queue
        self._value: Optional[List[str]] = None
        self._score: Optional[float] = None

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
        worker_list = [(worker, worker.get_score()) for worker in self._candidates]
        self._candidates = [e[0] for e in sorted(worker_list, key=lambda x: x[1], reverse=True)]

    def _run_inference(self):
        for worker in self._candidates:
            worker.start()
        for worker in self._candidates:
            worker.join()
    
    def get_best(self) -> List[str]:
        return Engine.merge_regex_sequence(self._candidates[0].value)

    @property
    def candidates(self) -> List[List[str]]:
        return [c.value for c in self._candidates]
    
    @property
    def scores(self) -> List[float]:
        return [c.score for c in self._candidates]
    
    def drop_bad(self, n_drop: int):
        assert n_drop < len(self._candidates), 'You can not drop too many candidates (at least one candidate should remain in the record).'
        retain_cnt = len(self._candidates) - n_drop
        self._candidates = self._candidates[:retain_cnt]

    def __or__(self, other: 'CandidateRecords') -> 'CandidateRecords':
        new_obj = CandidateRecords(self._candidates, run=False)
        new_obj._candidates = list(set(new_obj._candidates) | set(other._candidates))
        worker_list = [(worker, worker.score) for worker in new_obj._candidates]
        new_obj._candidates = [e[0] for e in sorted(worker_list, key=lambda x: x[1], reverse=True)]
        return new_obj

    def __eq__(self, other: 'CandidateRecords') -> bool:
        return self.candidates == other.candidates
    
    def __add__(self, other: 'CandidateRecords') -> 'CandidateRecords':
        new_regex_candidates = [c1 + c2 for c1 in self.candidates for c2 in other.candidates]
        val_buckets = [c1._val_patterns + c2._val_patterns for c1 in self._candidates for c2 in other._candidates]
        new_candidates = []
        for regex_list, val_patterns in zip(new_regex_candidates, val_buckets):
            c = Candidate(self._candidates[0]._engine, [], val_patterns)
            c._value = regex_list
            c._score = c.get_score()
            new_candidates.append(c)
        return CandidateRecords(new_candidates, run=False)
