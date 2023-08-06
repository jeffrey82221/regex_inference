from random_regex import RegexGenerator
from regex_inference import Engine
import exrex
SHOW_PRECISION = False
regex_gen = RegexGenerator(max_complexity=100, bloom_max_item_count=1000)
engine = Engine(verbose=False, simpify_regex=True)
regex_instance = next(regex_gen.generate())
regex_target = regex_instance['regex']
target_examples = regex_instance['examples']
print('regex target:', regex_target)
print('# of target examples:', len(target_examples))
regex_inferenced = engine.run(regex_instance['examples'])
print('regex predict:', regex_inferenced)
recall = 1. - len(engine.filter_mismatch(regex_inferenced, target_examples)) / len(target_examples)
print('recall:', recall)
if SHOW_PRECISION:
    simulated_examples = [e for e in exrex.generate(regex_inferenced)]
    print('# of simulated examples:', len(simulated_examples))
    precision = 1. - len(engine.filter_mismatch(regex_target, simulated_examples)) / len(simulated_examples)
    print('precision:', precision)
    print('F1-Score:', 2./(1./(precision)+1./(recall)))