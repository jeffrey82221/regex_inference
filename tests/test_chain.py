import pytest
from regex_inference.inference.chain import Chain
from regex_inference import Engine


@pytest.fixture
def chain():
    return Chain(temperature=0.0)


def test_inference_regex(chain):
    assert chain.inference_regex.run(Engine._convert_patterns_to_prompt(
        [str(i) for i in range(10)])).strip() in ['\\d', '[0-9]']
    assert chain.inference_regex.run(Engine._convert_patterns_to_prompt(
        ['a' + str(i) for i in range(10)])).strip() in ['a\\d', 'a[0-9]', '^a[0-9]$']


def test_simplify_regex(chain):
    assert chain.simplify_regex.run(
        regex='[0-3]|[0-9]').strip() in ('[0-9]')
    assert chain.simplify_regex.run(
        regex='[0-3]|[3-9]').strip() in ('[0-9]')
