import re
import pytest
from regex_inference import Engine


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
        "0.4.7",
        "0.4.8",
        "0.5",
        "0.6",
        "0.7",
        "0.8",
        "1.0",
        "1.0.0",
        "1.0.1",
        "1.0.15",
        "1.0.16",
        "1.0.17",
        "1.0.18",
        "1.0.2",
        "1.0.20",
        "1.0.20200721",
        "1.0.20200723",
        "1.0.20200812",
        "1.0.20200812.1",
        "1.0.20200812.2",
        "1.0.20200812.3",
        "1.0.20200812.4",
        "1.0.20200812.5",
        "1.0.20200812.6",
        "1.0.20200817",
        "1.0.20200820.10",
        "1.0.20200820.8",
        "1.0.20200820.9",
        "1.0.20200820.post2",
        "1.0.20200820.post4",
        "1.0.20200820.post5",
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
        "1.0.20200827.2",
        "1.0.20200827.3",
        "1.0.20200827.4",
        "1.0.20200827.5",
        "1.0.20200827.6",
        "1.0.20200827.7",
        "1.0.20200827.8",
        "1.0.20200828",
        "1.0.20200828.2",
        "1.0.20200828.3",
        "1.0.20200828.4",
        "1.0.20200829",
        "1.0.20200829.1",
        "1.0.20200829.2",
        "1.0.20200829.3",
        "1.0.20200831",
        "1.0.20200831.2",
        "1.0.3",
        "1.0.4",
        "1.0.5",
        "1.0.6",
        "1.0.7",
        "1.0.8",
        "1.0.9",
        "1.1",
        "1.1.0",
        "1.1.1",
        "1.1.10",
        "1.1.11",
        "1.1.12",
        "1.1.13",
        "1.1.14",
        "1.1.15",
        "1.1.16",
        "1.1.17",
        "1.1.18",
        "1.1.19",
        "1.1.2",
        "1.1.20",
        "1.1.21",
        "1.1.22",
        "1.1.23",
        "1.1.24",
        "1.1.25",
        "1.1.26",
        "1.1.27",
        "1.1.28",
        "1.1.29",
        "1.1.3",
        "1.1.30",
        "1.1.31",
        "1.1.32",
        "1.1.33",
        "1.1.34",
        "1.1.35",
        "1.1.36",
        "1.1.37",
        "1.1.38",
        "1.1.39",
        "1.1.4",
        "1.1.40",
        "1.1.41",
        "1.1.42",
        "1.1.43",
        "1.1.44",
        "1.1.45",
        "1.1.46",
        "1.1.47",
        "1.1.48",
        "1.1.49",
        "1.1.5",
        "1.1.50",
        "1.1.55",
        "1.1.56",
        "1.1.57",
        "1.1.6",
        "1.1.7",
        "1.1.8",
        "1.1.9",
        "1.11",
        "1.12",
        "1.13",
        "1.14",
        "1.15",
        "1.16",
        "1.2.0",
        "1.2.1",
        "1.2.2",
        "1.2.3",
        "1.2.4",
        "1.2.5",
        "1.2.6",
        "1.3.0",
        "1.3.1",
        "1.3.2",
        "1.3.3",
        "1.3.4",
        "1.3.5",
        "1.4.0",
        "2.1.4",
        "2016.4.0",
        "2016.6.0",
        "2017.1.0",
    ]


def test_run(patterns):
    e = Engine()
    regex = e.run(patterns)
    re_com = re.compile(regex)
    for pattern in patterns:
        check = re_com.fullmatch(pattern)
        assert check is not None, f'{pattern} does not fullmatch {regex}'


@pytest.fixture
def patterns_with_empty_string():
    return [
        "0",
        "9",
        "",
        "123",
        "apple",
        ""
    ]


def test_run_with_empty_string(patterns_with_empty_string):
    e = Engine()
    regex = e.run(patterns_with_empty_string)
    re_com = re.compile(regex)
    for pattern in patterns_with_empty_string:
        check = re_com.fullmatch(pattern)
        assert check is not None, f'{pattern} does not fullmatch {regex}'


def test_emply_input():
    e = Engine()
    with pytest.raises(AssertionError):
        e.run([])
