from regex_inference.inference.candidate import CandidateRecords, Candidate
from regex_inference import Engine

def test_drop_bad():
    c1 = Candidate(Engine(), [], [])
    c2 = Candidate(Engine(), [], [])
    c1._value = 'b'
    c1._score = 0.8
    c2._value = 'a'
    c2._score = 1.0
    records = CandidateRecords([c1, c2], run=False)
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
    records = CandidateRecords([c1, c2], run=False)
    assert (records | records).candidates == records.candidates
    c3 = Candidate(Engine(), [], [])
    c3._value = 'c'
    c3._score = 0.7
    records2 = CandidateRecords([c1, c3], run=False)
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
    assert CandidateRecords([c1, c2], run=False).get_best() == 'a'

def test_add():
    c1 = Candidate(Engine(), [], ['a'])
    c2 = Candidate(Engine(), [], ['b'])
    c1._value = ['a']
    c2._value = ['b']
    r1 = CandidateRecords([c1, c2], run=False)
    r2 = CandidateRecords([c1, c2], run=False)
    assert (r1 + r2).candidates == [['a', 'b'], ['b', 'a'], ['a', 'a'], ['b', 'b']]
    assert (r1 + r2).scores[:2] == [1.0, 1.0]