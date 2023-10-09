
import pytest
import re
from tests.fixture import versions, addresses, train_complex
from regex_inference import FAdoAIEngine, FAdoEngine


@pytest.fixture
def patterns_v1():
    return [
        'apple',
        'apple2',
        'apple3',
        'apple4'
    ]


@pytest.fixture
def patterns_v3():
    return [
        "0",
        "9",
        ""
    ]


def test_run(versions, addresses):
    for patterns in [versions, addresses]:
        e = FAdoAIEngine(max_iteration=1)
        regex = e.run(patterns)
        re_com = re.compile(regex)
        for pattern in patterns:
            check = re_com.fullmatch(pattern)
            assert check is not None, f'{pattern} does not fullmatch {regex}'


def test_infer_by_fado(patterns_v1):
    e = FAdoEngine()
    regex = e.infer_by_fado(patterns_v1)
    assert regex == 'apple[2-4]?'
    re_com = re.compile(regex)
    for pattern in patterns_v1:
        check = re_com.fullmatch(pattern)
        assert check is not None, f'{pattern} does not fullmatch {regex}'


def test_infer_by_fado_v2(train_complex):
    e = FAdoEngine()
    regex = e.infer_by_fado(train_complex)
    assert regex == '([09@]?|中華文化|123|\\ \\ \\ |apple)'
    re_com = re.compile(regex)
    for pattern in train_complex:
        check = re_com.fullmatch(pattern)
        assert check is not None, f'{pattern} does not fullmatch {regex}'


def test_infer_by_fado_v3(patterns_v3):
    e = FAdoEngine()
    regex = e.infer_by_fado(patterns_v3)
    assert regex == '[09]?'
    re_com = re.compile(regex)
    for pattern in patterns_v3:
        check = re_com.fullmatch(pattern)
        assert check is not None, f'{pattern} does not fullmatch {regex}'
