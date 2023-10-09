"""
Experimenting with the use of multi-selection in inner loop 
(for selection of sub-regex.)
"""
import random
from regex_inference import FAdoAIEngine, Engine, Evaluator
from more_itertools import chunked
import numpy as np
import pprint
import re
import pandas as pd
train_cnt = 1000
data_path = '../tests/data/version.txt'
with open(data_path, 'r') as f:
    whole_patterns = f.read().split('\n')
train_patterns = random.sample(whole_patterns, train_cnt)
eval_patterns = list(set(whole_patterns) - set(train_patterns))

train_buckets = list(chunked(train_patterns, 10))
print('number of train buckets:', len(train_buckets))

regex_candidates = []
for train in train_buckets:
    engine = FAdoAIEngine(max_iteration=3)
    # engine = Engine(max_iteration=1)
    regex = engine._run_new_inference(train)
    regex_candidates.append(regex)

pprint.pprint(regex_candidates)


# calculate precision, recall, f1 scores for each regex

scores = dict()
for regex in regex_candidates:
    scores[regex] = Evaluator.evaluate(regex, train_patterns)

pprint.pprint(scores)

match_matrix = np.array([[re.compile(regex).fullmatch(pattern) is not None for pattern in train_patterns] for regex in regex_candidates])
print('shape of match_matrix:', match_matrix.shape)


# order pattern by std
std_scores = np.std(match_matrix, axis=0)
mean_scores = np.mean(match_matrix, axis=0)

print('std_scores shape:', std_scores.shape)
print('mean_scores shape:', mean_scores.shape)


table = pd.DataFrame(zip(train_patterns, mean_scores, std_scores), 
                     columns=['pattern', 'mean', 'std'])
print(table)
# order pattern by match percentage
table.to_parquet('raw_fado.parquet')


