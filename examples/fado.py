"""
TODO:
- [X] More simplification:
    - [X] Convert `Set` to `Range`
    - [X] Use ChatGPT to simplify the exact regex infered using FAdo
    - [X] Evaluate the regex
"""
from regex_inference import Evaluator
from typing import List
from functools import reduce
import exrex
import random
from regex_inference import FAdoAIEngine
from regexfactory import escape
from FAdo.reex import CAtom, CConcat, CDisj, RegExp, CEpsilon, COption
from FAdo.conversions import FA2regexpCG


def convert_str_to_fado_regex(input_str: str) -> CConcat:
    atoms = []
    for ch in input_str:
        atoms.append(CAtom(escape(ch).regex))
    return reduce(lambda x, y: CConcat(x, y), atoms)


def make_regex_union(inputs: List[str]) -> RegExp:
    fado_regex_list = map(convert_str_to_fado_regex, inputs)
    return reduce(lambda x, y: CDisj(x, y), fado_regex_list)


def to_standard_regex(regex: RegExp) -> str:
    if isinstance(regex, CAtom):
        return regex.val
    elif isinstance(regex, COption):
        content = regex.arg
        if isinstance(content, CAtom):
            return f'{to_standard_regex(content)}?'
        else:
            return f'({to_standard_regex(content)})?'
    elif isinstance(regex, CConcat):
        return to_standard_regex(regex.arg1) + to_standard_regex(regex.arg2)
    elif isinstance(regex, CDisj):
        if isinstance(regex.arg1, CEpsilon):
            return to_standard_regex(COption(regex.arg2))
        if isinstance(regex.arg2, CEpsilon):
            return to_standard_regex(COption(regex.arg1))
        x1, x2 = to_standard_regex(regex.arg1), to_standard_regex(regex.arg2)
        if isinstance(regex.arg1, CAtom) and isinstance(regex.arg2, CAtom):
            set_str = f'{x1}{x2}'
            set_str = ''.join(sorted(set_str))
            return f'[{set_str}]'
        elif isinstance(regex.arg1, CDisj) and x1[0] == '[' and x1[-1] == ']' and isinstance(regex.arg2, CAtom):
            x1 = x1[1:-1]
            set_str = f'{x1}{x2}'
            set_str = ''.join(sorted(set_str))
            return f'[{set_str}]'
        elif isinstance(regex.arg1, CDisj) and x1[0] == '(' and x1[-1] == ')':
            x1 = x1[1:-1]
            return f'({x1}|{x2})'
        else:
            return f'({x1}|{x2})'

    elif isinstance(regex, CEpsilon):
        return '[]'


def generate_digit_range():
    for i in range(10):
        for j in range(i + 2, 10):
            content = ''.join([str(e) for e in range(i, j + 1)])
            if i == 0 and j == 9:
                yield f'[{content}]', '\\d'
            else:
                yield f'[{content}]', f'[{i}-{j}]'


def to_simplied_standard_regex(regex: RegExp):
    standard_regex = to_standard_regex(regex)
    standard_regex = reduce(lambda x, y: x.replace(
        *y), [standard_regex, *list(generate_digit_range())])
    return standard_regex


def infer_regex_from_patterns(inputs: List[str]) -> str:
    union_regex = make_regex_union(inputs)
    minimal_dfa = union_regex.nfaPD().toDFA().minimal()
    fado_regex = FA2regexpCG(minimal_dfa)
    standard_regex = to_simplied_standard_regex(fado_regex)
    return standard_regex


if __name__ == '__main__':
    # Take Inputs
    inputs = []
    with open('../tests/data/version.txt') as f:
        inputs = list(map(lambda x: x.replace('\n', ''), f))
    train_inputs = random.choices(inputs, k=200)
    test_inputs = list(set(inputs) - set(train_inputs))

    print('inputs:', train_inputs[:10])
    # Main Function
    standard_regex = infer_regex_from_patterns(train_inputs)
    print('regex:', standard_regex)
    # Testing
    sim_patterns = [e for e in exrex.generate(standard_regex)]
    assert set(sim_patterns) == set(train_inputs), 'sim_patterns != inputs'
    print('unneccessary patterns:', set(sim_patterns) - set(train_inputs))
    print('unmatched patterns:', set(train_inputs) - set(sim_patterns))
    # Evaluate on Training Data
    print('#Evaluate on Training Data')
    precision, recall, f1 = Evaluator.evaluate(
        standard_regex, train_inputs)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)
    # Evaluate on Testing Data
    print('#Evaluate on Testing Data')
    precision, recall, f1 = Evaluator.evaluate(
        standard_regex, test_inputs)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)
    # Evaluate on Testing Data after generalized
    print('#Evaluate on Testing Data After Regex Simplified')
    refined_regex = FAdoAIEngine()._run_simplify_regex(standard_regex)

    precision, recall, f1 = Evaluator.evaluate(
        refined_regex, test_inputs)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)
    print('#Evaluate on Testing Data After Regex Simplified 2nd time')
    refined_regex = FAdoAIEngine()._run_simplify_regex(refined_regex)
    # Evaluate on Testing Data after generalized
    precision, recall, f1 = Evaluator.evaluate(
        refined_regex, test_inputs)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)
