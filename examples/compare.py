from regex_inference import Evaluator, Inference
import random


train_cnt = 10
whole_patterns = []
data_path = '../tests/data/version.txt'
with open(data_path, 'r') as f:
    whole_patterns = f.read().split('\n')
train_patterns = random.sample(whole_patterns, train_cnt)
eval_patterns = list(set(whole_patterns) - set(train_patterns))

if __name__ == '__main__':
    print('AI approach:')
    inferencer = Inference(verbose=False, engine='ai', max_iteration=10)
    regex = inferencer.run(
        train_patterns)
    print(inferencer.openai_summary)
    scores = Evaluator.evaluate(regex, eval_patterns)
    print('scores:', scores)

    print('FAdo+AI approach:')
    inferencer = Inference(verbose=False, engine='fado+ai', max_iteration=10)
    regex = inferencer.run(
        train_patterns)
    print(inferencer.openai_summary)
    scores = Evaluator.evaluate(regex, eval_patterns)
    print('scores:', scores)


"""
###########################################
# Experiment 1: Compare performace & cost #
###########################################

Data: version.txt 

AI approach:
{'total_tokens': 7053, 'prompt_tokens': 6237, 'completion_tokens': 816, 'total_cost': 0.14106000000000002}
scores: (1.0, 0.9124483613329661, 0.9542201293184142)
FAdo+AI approach:
{'total_tokens': 6407, 'prompt_tokens': 5182, 'completion_tokens': 1225, 'total_cost': 0.12814}
scores: (1.0, 0.7021206279261911, 0.8249951459452461)

Data: address.txt

AI approach:
{'total_tokens': 10185, 'prompt_tokens': 8997, 'completion_tokens': 1188, 'total_cost': 0.2037}
scores: (1.0, 0.9932692307692308, 0.9966232513265798)
FAdo+AI approach:
{'total_tokens': 9753, 'prompt_tokens': 7841, 'completion_tokens': 1912, 'total_cost': 0.19506}
scores: (1.0, 0.35865384615384616, 0.5279547062986554)
"""