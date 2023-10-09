import pytest
import re
from regex_inference import Inference
from regex_inference import Evaluator
from tests.fixture import train_complex, versions_slim

@pytest.fixture
def train_abc():
    return ['a', 'b', 'c']

def test_run_by_pure_ai(train_abc, train_complex):
    for train in [train_abc, train_complex]:
        inf = Inference(temperature=0, engine='ai', max_iteration=1)
        regex = inf.run(train, val_patterns=train, n_fold=1)
        assert re.compile(regex)
        assert all([re.compile(regex).match(x) is not None for x in train])


def test_run_by_fado_ai(versions_slim):
    for _, train in enumerate([versions_slim]):
        inf = Inference(temperature=0.8, engine='fado+ai', max_iteration=3)
        regex = inf.run(train, n_fold=12, train_rate=0.2)
        assert re.compile(regex)
        _, _, f1 = Evaluator.evaluate(regex, train)
        assert f1 >= 0.5


def test_run_by_cross_validate(versions_slim):
    for i, train in enumerate([versions_slim]):
        inf = Inference(temperature=0.8, engine='ai', max_iteration=3)
        regex = inf.run(train, n_fold=3)
        assert re.compile(regex)
        _, _, f1 = Evaluator.evaluate(regex, train)
        assert f1 >= 0.7
