import re
import pytest
from regex_inference import Engine
from tests.fixture import versions_more, addresses_more


def test_run(versions_more, addresses_more):
    for patterns in [versions_more, addresses_more]:
        e = Engine(max_iteration=1)
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
        "",
        "@",
        "中華文化",
        "   "
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


def test_fix_regex_list():
    e = Engine(temperature=0.0)
    result = e.fix_regex_list(
        ["[1-4]", "[3-5]", "[5-8]"],
        {
            '[1-4]': {
                'correct': ['1', '2', '3'],
                'incorrect': ['4']
            },
            '[3-5]': {
                'correct': ['4', '5'],
                'incorrect': ['3']
            },
            '[5-8]': {
                'correct': ['6', '7', '8'],
                'incorrect': ['5']
            }
        }
    )
    assert result[0] == '[1-3]'
    assert result[1] in ('[45]', '[4-5]')
    assert result[2] == '[6-8]'
