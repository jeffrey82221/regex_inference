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


def test_alter_regex(chain):
    assert chain.alter_regex.run(
        regex='[0-3]',
        strings=Engine._convert_patterns_to_prompt(
            [
                str(i) for i in range(10)])).strip() in (
        '[0-9]',
        '[0-3]|[0-9]')
    assert chain.alter_regex.run(
        regex='[0-4]',
        strings=Engine._convert_patterns_to_prompt(
            [
                str(i) for i in range(
                    3,
                    8)])).strip() == '[3-7]'


def test_simplify_regex(chain):
    assert chain.simplify_regex.run(
        regex='[0-3]|[0-9]',
        strings=Engine._convert_patterns_to_prompt(
            [
                str(i) for i in range(10)])).strip() in ('[0-9]')
    assert chain.simplify_regex.run(
        regex='[0-3]|[3-9]',
        strings=Engine._convert_patterns_to_prompt(
            [
                str(i) for i in range(10)])).strip() in ('[0-9]')