"""
Evaluate Inference using Precision / Recall / F1 Given Different Train count
"""
from regex_inference import Engine, Evaluator, Inference
import random
TRAIN_CNT = 200
whole_patterns = []
with open('data/version.txt', 'r') as f:
    whole_patterns = f.read().split('\n')
train_patterns = random.sample(whole_patterns, TRAIN_CNT)
eval_patterns = list(set(whole_patterns) - set(train_patterns))
if __name__ == '__main__':
    e = Engine(verbose=False, use_openai=False, model_id='EleutherAI/gpt-neox-20b', max_length=1000)
    regex_list = e.get_regex_sequence(train_patterns)
    precision, recall, f1 = Evaluator.evaluate_regex_list(
        regex_list, eval_patterns)
    print('regex_list count:', len(regex_list))
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)
    regex = Engine.merge_regex_sequence(regex_list)
    print(regex)
    inferencer = Inference(verbose=False, n_thread=5, use_openai=False, model_id='EleutherAI/gpt-neox-20b', max_length=1000)
    regex_list = inferencer.get_regex_sequence(
        train_patterns, val_patterns=eval_patterns)
    precision, recall, f1 = Evaluator.evaluate_regex_list(
        regex_list, eval_patterns)
    print('regex_list count:', len(regex_list))
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)
    regex = Engine.merge_regex_sequence(regex_list)
    print(regex)
    correction_data = e.get_correction_data(regex_list, train_patterns)
    regex_list = e.fix_regex_list(regex_list, correction_data)
    precision, recall, f1 = Evaluator.evaluate_regex_list(
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
