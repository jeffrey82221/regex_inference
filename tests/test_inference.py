import re
from regex_inference import Inference


def test_run():
    inf = Inference(temperature=0, n_thread=3)
    regex = inf.run(['a', 'b', 'c'])
    assert re.compile(regex)
    regex = inf.run(['a', 'b', 'c'], val_patterns=['a', 'b', 'c'])
    assert re.compile(regex)
