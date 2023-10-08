import pytest
from regex_inference import FAdoAIEngine, FAdoEngine, Evaluator


@pytest.fixture
def train_patterns():
    return [
        '0',
        '1',
        '3',
        '4',
        '8',
        '9'
    ]


@pytest.fixture
def test_patterns():
    return [
        '2',
        '5',
        '6',
        '7'
    ]


def test_evaluate(train_patterns, test_patterns):
    regex = FAdoEngine()._run_new_inference(train_patterns)
    assert Evaluator.evaluate(regex, train_patterns) == (1.0, 1.0, 1.0)
    assert Evaluator.evaluate(regex, test_patterns) == (0.0, 0.0, 0.0)
    regex = FAdoAIEngine(temperature=0.0)._run_new_inference(train_patterns)
    p, r, f = Evaluator.evaluate(regex, test_patterns)
    assert p > 0.5 and p <= 1
    assert r > 0.5 and r <= 1
    assert f == 2. / ((1. / p) + (1. / r))
    
