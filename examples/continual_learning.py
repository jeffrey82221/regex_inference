"""
Experimenting with the use of multi-selection in inner loop 
(for selection of sub-regex.)
"""
# %% load data
import random
from regex_inference import Inference, Engine, Evaluator
from more_itertools import chunked
import numpy as np
import pprint
import re
import pandas as pd
train_cnt = 3000
test_cnt = 1000
random.seed(1)
data_path = '../tests/data/version.txt'
with open(data_path, 'r') as f:
    whole_patterns = f.read().split('\n')
train_patterns = random.sample(whole_patterns, train_cnt)
test_patterns = list(set(whole_patterns)-set(train_patterns))
train_patterns_v1, train_patterns_v2, train_patterns_v3 = list(chunked(train_patterns, 1000))
# train_patterns_v1 = train_patterns_v1[:100]
print('# test instance:', len(test_patterns))
print('# train_patterns_v1:', len(train_patterns_v1))
print('# train_patterns_v2:', len(train_patterns_v2))
print('# train_patterns_v3:', len(train_patterns_v3))
import os
import pickle 
trainer = Inference(engine='ai')
if os.path.exists('regex_candidate.pkl'):
    regex_candidates = pickle.load(open('regex_candidate.pkl', 'rb'))
    best_regex = pickle.load(open('best_regex.pkl', 'rb'))
    print('load regex candidates')
else:
    best_regex = trainer.run(train_patterns_v1, train_rate=0.1)
    regex_candidates = trainer._regex_candidate
    pickle.dump(regex_candidates, open('regex_candidate.pkl', 'wb'))
    pickle.dump(best_regex, open('best_regex.pkl', 'wb'))
    print('save regex candidates')
pprint.pprint(regex_candidates)
p, r, f1 = Evaluator.evaluate(best_regex, train_patterns_v2)
print(f'Scores on train patterns v2: precision: {p}, recall: {r}, f1: {f1}')
import numpy as np
p_means = []
r_means = []
f_means = []
p_stds = []
r_stds = []
f_stds = []
for pattern in train_patterns_v2:
    ps = []
    rs = []
    f1s = []
    for regex_list in regex_candidates:
        p, r, f1 = Evaluator.evaluate_regex_list(regex_list, [pattern])
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
    p_means.append(np.mean(ps))
    p_stds.append(np.std(ps))
    r_means.append(np.mean(rs))
    r_stds.append(np.std(rs))
    f_means.append(np.mean(f1s)) 
    f_stds.append(np.std(f1s))

import pandas as pd
table = pd.DataFrame(zip(train_patterns_v2, p_means, r_means, f_means, p_stds, r_stds, f_stds), columns = [
    'pattern',
    'precision',
    'recall',
    'f1',
    'precision_std',
    'recall_std',
    'f1_std'
])
# from matplotlib import pylab as plt
# table.plot(
#     x='recall',
#     y='precision',
#     kind='scatter'
# )
# plt.show()

# Do continual learning on train_patterns_v2

# Mode1: random select 100 training data from train_patterns_v2
new_regex = trainer.run(random.sample(train_patterns_v2, 100), train_rate=0.1)
new_regex = f'({best_regex})|({new_regex})'
p, r, f1 = Evaluator.evaluate(new_regex, test_patterns)
print(f'Random Select Learning: precision: {p}, recall: {r}, f1: {f1}')
table['f1-f1_std'] = table['f1'] - table['f1_std']
# Model2: select 100 training data with largest recall std:
table.sort_values('f1-f1_std', inplace=True, ascending=True)
print(table)
new_regex = trainer.run(table.pattern.iloc[:100].tolist(), train_rate=0.1)
new_regex = f'({best_regex})|({new_regex})'
p, r, f1 = Evaluator.evaluate(new_regex, test_patterns)
print(f'F1 low F1-std high selected Learning: precision: {p}, recall: {r}, f1: {f1}')

# Do continual learning on train_patterns_v3
import numpy as np
p_means = []
r_means = []
f_means = []
p_stds = []
r_stds = []
f_stds = []
for pattern in train_patterns_v3:
    ps = []
    rs = []
    f1s = []
    for regex_list in regex_candidates:
        p, r, f1 = Evaluator.evaluate_regex_list(regex_list, [pattern])
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
    p_means.append(np.mean(ps))
    p_stds.append(np.std(ps))
    r_means.append(np.mean(rs))
    r_stds.append(np.std(rs))
    f_means.append(np.mean(f1s)) 
    f_stds.append(np.std(f1s))

import pandas as pd
table = pd.DataFrame(zip(train_patterns_v3, p_means, r_means, f_means, p_stds, r_stds, f_stds), columns = [
    'pattern',
    'precision',
    'recall',
    'f1',
    'precision_std',
    'recall_std',
    'f1_std'
])


# Mode1: random select 100 training data from train_patterns_v3
new_regex_v2 = trainer.run(random.sample(train_patterns_v3, 100), train_rate=0.1)
new_regex = f'({new_regex})|({new_regex_v2})'
p, r, f1 = Evaluator.evaluate(new_regex, test_patterns)
print(f'Random Select Learning: precision: {p}, recall: {r}, f1: {f1}')
table['f1-f1_std'] = table['f1'] - table['f1_std']
# Model2: select 100 training data with largest recall std:
table.sort_values('f1-f1_std', inplace=True, ascending=True)
print(table)
new_regex_v2 = trainer.run(table.pattern.iloc[:100].tolist(), train_rate=0.1)
new_regex = f'({new_regex})|({new_regex_v2})'
p, r, f1 = Evaluator.evaluate(new_regex, test_patterns)
print(f'F1 low F1-std high selected Learning: precision: {p}, recall: {r}, f1: {f1}')