
import pytest 
import re
from regex_inference import FAdoEngine

@pytest.fixture
def patterns():
    return [
        "0.0.1",
        "0.0.10",
        "0.0.12",
        "0.0.13",
        "0.0.2",
        "0.0.3",
        "0.0.3.2",
        "0.0.4",
        "0.0.5",
        "0.0.6",
        "0.0.7",
        "0.0.8",
        "0.0.9",
        "0.1",
        "0.1.0",
        "0.1.1",
        "0.1.10",
        "0.1.11",
        "0.1.12",
        "0.1.13",
        "0.1.14",
        "0.1.15",
        "0.1.16",
        "0.1.17",
        "0.1.18",
        "0.1.2",
        "0.1.2a0",
        "0.1.3",
        "0.1.4",
        "0.1.5",
        "0.1.6",
        "0.1.7",
        "0.1.8",
        "0.1.9",
        "0.10",
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
        "0.22",
        "0.23",
        "0.24",
        "0.25",
        "0.26",
        "0.27",
        "0.28",
        "0.29",
        "0.3",
        "0.3.0",
        "0.3.1",
        "0.3.11",
        "0.3.12",
        "0.3.13",
        "0.3.14",
        "0.3.15",
        "0.3.2",
        "0.3.33",
        "0.3.34",
        "0.3.36",
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
        "1.0.20200821.2",
        "1.0.20200821.3",
        "1.0.20200821.4",
        "1.0.20200824",
        "1.0.20200825",
        "1.0.20200825.10",
        "1.0.20200825.2",
        "1.0.20200825.3",
        "1.0.20200825.4",
        "1.0.20200825.5",
        "1.0.20200825.6",
        "1.0.20200825.7",
        "1.0.20200825.8",
        "1.0.20200825.9",
        "1.0.20200827",
        "1.1.44",
        "1.1.45",
        "1.1.46",
        "1.1.47",
        "1.1.48",
        "1.2.0",
        "1.2.1",
        "1.2.2",
        "2016.6.0",
        "2017.1.0",
    ]

@pytest.fixture
def patterns_v1():
    return [
        'apple', 
        'apple2', 
        'apple3', 
        'apple4'
    ]

@pytest.fixture
def patterns_v2():
    return [
        "0",
        "9",
        "123",
        "apple",
        "@",
        "中華文化",
        "   "
    ]

@pytest.fixture
def patterns_v3():
    return [
        "0",
        "9",
        ""
    ]


def test_run(patterns):
    e = FAdoEngine()
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


def test_infer_by_fado_v2(patterns_v2):
    e = FAdoEngine()
    regex = e.infer_by_fado(patterns_v2)
    assert regex == '([09@]|中華文化|\ \ \ |123|apple)'
    re_com = re.compile(regex)
    for pattern in patterns_v2:
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

        

