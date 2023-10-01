"""
Inference Engine Mixing FAdo automata tool and AI
"""
import re
from typing import List
from functools import reduce
from regexfactory import escape
from FAdo.reex import RegExp, CDisj, CAtom, CConcat, CEpsilon, COption
from FAdo.conversions import FA2regexpCG
from .engine import Engine


__all__ = ['FAdoEngine']


class FAdoEngine(Engine):
    """
    Engine that infer regex using both FAdo and ChatGPT.
    """

    def _run_new_inference(self, patterns: List[str]) -> str:
        for _ in range(self._max_iteration):
            regex = self.infer_by_fado(patterns)
            result = self._run_simplify_regex(regex).strip()
            try:
                re.compile(result)
                break
            except KeyboardInterrupt as e:
                raise e
            except BaseException:
                pass
        return result

    @staticmethod
    def infer_by_fado(inputs: List[str]) -> str:
        union_regex = FAdoEngine._make_regex_union(inputs)
        minimal_dfa = union_regex.nfaPD().toDFA().minimal()
        fado_regex = FA2regexpCG(minimal_dfa)
        standard_regex = FAdoEngine._to_simplied_standard_regex(fado_regex)
        return standard_regex

    @staticmethod
    def _make_regex_union(inputs: List[str]) -> RegExp:
        fado_regex_list = map(FAdoEngine._convert_str_to_fado_regex, inputs)
        return reduce(lambda x, y: CDisj(x, y), fado_regex_list)

    @staticmethod
    def _convert_str_to_fado_regex(input_str: str) -> CConcat:
        if input_str == '':
            return CEpsilon()
        else:
            atoms = []
            for ch in input_str:
                atoms.append(CAtom(escape(ch).regex))
            return reduce(lambda x, y: CConcat(x, y), atoms)

    @staticmethod
    def _to_simplied_standard_regex(regex: RegExp) -> str:
        standard_regex = FAdoEngine._to_standard_regex(regex)
        standard_regex = reduce(lambda x, y: x.replace(
            *y), [standard_regex, *list(FAdoEngine._generate_digit_range())])
        return standard_regex

    @staticmethod
    def _to_standard_regex(regex: RegExp) -> str:
        if isinstance(regex, CAtom):
            return regex.val
        elif isinstance(regex, COption):
            content = regex.arg
            regex = FAdoEngine._to_standard_regex(content)
            if isinstance(content, CAtom):
                return f'{regex}?'
            elif isinstance(content, CDisj) and regex[0] == '[' and regex[-1] == ']':
                return f'{regex}?'
            else:
                return f'({regex})?'
        elif isinstance(regex, CConcat):
            return FAdoEngine._to_standard_regex(
                regex.arg1) + FAdoEngine._to_standard_regex(regex.arg2)
        elif isinstance(regex, CDisj):
            if isinstance(regex.arg1, CEpsilon):
                return FAdoEngine._to_standard_regex(COption(regex.arg2))
            if isinstance(regex.arg2, CEpsilon):
                return FAdoEngine._to_standard_regex(COption(regex.arg1))
            x1, x2 = FAdoEngine._to_standard_regex(
                regex.arg1), FAdoEngine._to_standard_regex(
                regex.arg2)
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
        else:
            raise TypeError(f'type of regex: {type(regex)} cannot be handled')

    @staticmethod
    def _generate_digit_range():
        for i in range(10):
            for j in range(i + 2, 10):
                content = ''.join([str(e) for e in range(i, j + 1)])
                if i == 0 and j == 9:
                    yield f'[{content}]', '\\d'
                else:
                    yield f'[{content}]', f'[{i}-{j}]'
