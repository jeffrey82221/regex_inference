import pytest
import re
from regex_inference import Inference
from regex_inference import Evaluator


@pytest.fixture
def train_abc():
    return ['a', 'b', 'c']


@pytest.fixture
def train_complex():
    return [
        "0",
        "9",
        "",
        "123",
        "apple",
        "",
        "@",
        "中華文化",
        "   "
    ]


@pytest.fixture
def train_versions():
    return [
        "0.0.1",
        "0.0.10",
        "0.0.3.2",
        "0.0.4",
        "0.0.5",
        "0.0.6",
        "0.0.7",
        "0.0.8",
        "0.1.14",
        "0.1.15",
        "0.1.16",
        "0.1.17",
        "0.1.18",
        "0.1.2",
        "0.1.2a0",
        "0.1.3",
        "0.1.4",
        "0.11",
        "0.12",
        "0.13",
        "0.14",
        "0.15",
        "0.16",
        "0.17",
        "0.18",
        "0.19",
        "0.2.0",
        "0.2.6",
        "0.20",
        "0.21",
        "0.3",
        "0.3.0",
        "0.3.1",
        "0.3.11",
        "0.3.5",
        "0.3.6",
        "0.3.7",
        "0.3.8",
        "0.3.9",
        "0.30",
        "0.4",
        "0.4.6",
        "1.0.2",
        "1.0.20",
        "1.0.20200721",
        "1.0.20200723",
        "1.0.20200812",
        "1.0.20200812.1",
        "1.0.20200812.2",
        "1.0.20200812.3",
        "1.0.20200820.post6",
        "1.0.20200820.post7",
        "1.0.20200821",
        "1.0.20200825.2",
        "1.0.20200825.3",
        "1.0.20200827",
        "1.1.44",
        "1.1.45",
        "1.1.46",
        "1.2.1",
        "1.2.2",
        "2016.6.0",
        "2017.1.0"
    ]


def test_run_by_pure_ai(train_abc, train_complex, train_versions):
    for train in [train_abc, train_complex, train_versions]:
        inf = Inference(temperature=0, engine='ai')
        regex = inf.run(train, val_patterns=train, n_fold=1)
        assert re.compile(regex)
        assert all([re.compile(regex).match(x) is not None for x in train])


def test_run_by_fado_ai(train_versions):
    for i, train in enumerate([train_versions]):
        inf = Inference(temperature=0.8, engine='fado+ai')
        regex = inf.run(train, n_fold=12, train_rate=0.2)
        assert re.compile(regex)
        precision, recall, f1 = Evaluator.evaluate(regex, train)
        assert f1 >= 0.8


def test_run_by_cross_validate(train_versions):
    for i, train in enumerate([train_versions]):
        inf = Inference(temperature=0.8, engine='ai')
        regex = inf.run(train, n_fold=10)
        assert re.compile(regex)
        assert all([re.compile(regex).match(
            x) is not None for x in train]), f'error on no.{i} train'
