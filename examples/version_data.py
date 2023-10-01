"""
Evaluate Inference using Precision / Recall / F1 Given Different Train count
"""
from regex_inference import Evaluator, Inference
import random
TRAIN_CNT = 200

whole_patterns = []
with open('data/version.txt', 'r') as f:
    whole_patterns = f.read().split('\n')
train_patterns = random.sample(whole_patterns, TRAIN_CNT)
eval_patterns = list(set(whole_patterns) - set(train_patterns))

if __name__ == '__main__':

    print('FAdoEngine with 1 thread')
    engine = 'fado+ai'
    inferencer = Inference(verbose=False, n_thread=1, engine=engine)
    regex = inferencer.run(train_patterns)
    precision, recall, f1 = Evaluator.evaluate(
        regex, eval_patterns)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)
    print('FAdoEngine with 3 thread')
    inferencer = Inference(verbose=False, n_thread=3, engine=engine)
    regex = inferencer.run(
        train_patterns, val_patterns=eval_patterns)
    precision, recall, f1 = Evaluator.evaluate(
        regex, eval_patterns)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)
    print('Pure AI Engine with 1 thread')
    engine = 'ai'
    inferencer = Inference(verbose=False, n_thread=1, engine=engine)
    regex = inferencer.run(train_patterns)
    precision, recall, f1 = Evaluator.evaluate(
        regex, eval_patterns)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)
    print('Pure AI Engine with 3 thread')
    inferencer = Inference(verbose=False, n_thread=3, engine=engine)
    regex = inferencer.run(
        train_patterns, val_patterns=eval_patterns)
    precision, recall, f1 = Evaluator.evaluate(
        regex, eval_patterns)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)

"""
#####################################################
# Experiment 1: infer with different training count #
#####################################################

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

###################################################
# Experiment 2: Compare FAdo and Pure AI approach #
###################################################
TRAIN_CNT = 1

FAdoEngine with 1 thread
precision: 1.0
recall: 0.1637781867120767
f1: 0.2814594543566506
FAdoEngine with 3 thread
precision: 1.0
recall: 0.3869647614600786
f1: 0.5580022971206781
Pure AI Engine with 1 thread
precision: 1.0
recall: 0.3869647614600786
f1: 0.5580022971206781
Pure AI Engine with 3 thread
precision: 1.0
recall: 0.3869647614600786
f1: 0.5580022971206781

TRAIN_CNT = 10

FAdoEngine with 1 thread
precision: 1.0
recall: 0.007197802197802198
f1: 0.01429272816540287
FAdoEngine with 3 thread
precision: 1.0
recall: 0.7436263736263736
f1: 0.8529652738387848
Pure AI Engine with 1 thread
precision: 1.0
recall: 0.8057967032967033
f1: 0.8924556145502123
Pure AI Engine with 3 thread
precision: 1.0
recall: 0.7122802197802198
f1: 0.8319668843358416


TRAIN_CNT = 100

FAdoEngine with 1 thread
precision: 1.0
recall: 0.8416689617185349
f1: 0.9140285026394103
FAdoEngine with 3 thread
precision: 1.0
recall: 0.8352795373175433
f1: 0.9102477528174192
Pure AI Engine with 1 thread
precision: 1.0
recall: 0.961993941063068
f1: 0.9806288601909041
Pure AI Engine with 3 thread
precision: 1.0
recall: 0.9700082621867254
f1: 0.9847758314576898
"""
