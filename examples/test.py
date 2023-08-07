"""
Evaluate Inference using Precision / Recall / F1 Given Different Train count
"""
from regex_inference import Engine
import time
import random
TRAIN_CNT = 200
whole_patterns = []
with open('data/version.txt', 'r') as f:
    whole_patterns = f.read().split('\n')
train_patterns = random.sample(whole_patterns, TRAIN_CNT)
eval_patterns = list(set(whole_patterns) - set(train_patterns))
if __name__ == '__main__':
    e = Engine(verbose=True)
    start = time.time()
    regex_list = e.get_regex_sequence(train_patterns)
    end = time.time()
    print('run time =', end - start)
    precision, recall, f1 = Engine.evaluate_regex_list(
        regex_list, eval_patterns)
    print('regex_list count:', len(regex_list))
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)

"""
Result:

TRAIN_CNT = 1

regex_list count: 1
precision: 1.0
recall: 0.12842978384465378
f1: 0.2276256541316782

TRAIN_CNT = 10

regex_list count: 2
precision: 0.5001064735945485
recall: 0.5446978021978022
f1: 0.5214505785310993

TRAIN_CNT = 100

regex_list count: 5
precision: 0.4718731351244318
recall: 0.9224456072707243
f1: 0.6243583872893784

TRAIN_CNT = 200

regex_list count: 4
precision: 0.6253244357827017
recall: 0.9697045015189174
f1: 0.7603372028022296
"""
