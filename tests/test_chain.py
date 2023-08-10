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


def test_fix_regex(chain):

    regex_data_status = [
        ("[1-4]", ["1", "2", "3"], ["4"]),
        ("[3-5]", ["4", "5"], ["3"]),
        ("[5-8]", ["6", "7", "8"], ["5"]),
    ]
    n = len(regex_data_status)
    fact_0_str = f"""
Fact 0:

A list of regex describing {n} type of patterns is double quoted and shown as the following bullet points:
    """
    regex_list_str = "\n".join(
        map(lambda x: f'{x[0]+1}. "{x[1][0]}"', enumerate(regex_data_status)))

    facts = "\n\n".join(map(lambda x: f"""
Fact {x[0]+1}

For regex number {x[0]+1}, it correctly match the patterns double quoted and shown as follows:

{Engine._convert_patterns_to_prompt(x[1][1])}

However, it mistakenly match the patterns double quoted and shown as follows:

{Engine._convert_patterns_to_prompt(x[1][2])}

""", enumerate(regex_data_status)))
    ans = chain.fix_regex.run(
        facts=f"""
{fact_0_str}

{regex_list_str}

Now, I will provide to you {n} facts.

{facts}
        """
    )
    parsed_result = list(map(eval, ans.strip().split()))
    assert len(parsed_result) == 3
    assert parsed_result[0][0] == '[1-4]'
    assert parsed_result[1][0] == '[3-5]'
    assert parsed_result[2][0] == '[5-8]'
    assert parsed_result[0][1] == '[1-3]'
    assert parsed_result[1][1] in ('[45]', '[4-5]')
    assert parsed_result[2][1] == '[6-8]'
