from regex_inference.inference.candidate import CandidateRecords, Candidate
from regex_inference import Engine


def test_drop_bad():
    c1 = Candidate(Engine(), [], [])
    c2 = Candidate(Engine(), [], [])
    c1._value = 'b'
    c1._score = 0.8
    c2._value = 'a'
    c2._score = 1.0
    records = sorted(CandidateRecords([c1, c2]))
    records.drop_bad(1)
    assert len(records.candidates) == 1
    assert len(records.scores) == 1
    assert records.candidates[0] == 'a'
    assert records.scores[0] == 1.0


def test_or():
    c1 = Candidate(Engine(), [], [])
    c2 = Candidate(Engine(), [], [])
    c1._value = 'b'
    c1._score = 0.8
    c2._value = 'a'
    c2._score = 1.0
    records = sorted(CandidateRecords([c1, c2]))
    assert (records | records).candidates == records.candidates
    c3 = Candidate(Engine(), [], [])
    c3._value = 'c'
    c3._score = 0.7
    records2 = sorted(CandidateRecords([c1, c3]))
    assert records.candidates == ['a', 'b']
    assert records2.candidates == ['b', 'c']
    assert (records2 | records).candidates == ['a', 'b', 'c']
    assert (records2 | records).scores == [1.0, 0.8, 0.7]


def test_get_best():
    c1 = Candidate(Engine(), [], [])
    c2 = Candidate(Engine(), [], [])
    c1._value = 'b'
    c1._score = 0.8
    c2._value = 'a'
    c2._score = 1.0
    assert CandidateRecords([c1, c2]).get_best() == 'a'


def test_add():
    c1 = Candidate(Engine(), [], ['a'])
    c2 = Candidate(Engine(), [], ['b'])
    c1._value = ['a']
    c2._value = ['b']
    r1 = CandidateRecords([c1, c2])
    r2 = CandidateRecords([c1, c2])
    assert (r1 + r2).candidates == [['a', 'b'],
                                    ['b', 'a'], ['a', 'a'], ['b', 'b']]
    assert (r1 + r2).scores[:2] == [1.0, 1.0]


def test_sort_by_benefit():
    c1 = Candidate(Engine(), [], ['a'])
    c2 = Candidate(Engine(), [], ['a'])
    c1._value = ['a']
    c2._value = ['a', 'b']
    r1 = CandidateRecords([c1, c2])
    assert r1.sort_by_benefit(['a', 'b']) == ['b', 'a']
