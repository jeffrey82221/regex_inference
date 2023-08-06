from random_regex import RegexGenerator
from regex_inference import Engine
import exrex
regex_gen = RegexGenerator(max_complexity=10, bloom_max_item_count=10)
engine = Engine()
regex_instance = next(regex_gen.generate())
regex_target = regex_instance['regex']
target_examples = regex_instance['examples']
print('regex target:', regex_target)
print('# of target examples:', len(target_examples))
regex_inferenced = engine.run(regex_instance['examples'])
print('regex predict:', regex_inferenced)
simulated_examples = [e for e in exrex.generate(regex_inferenced)]
print('# of simulated examples:', len(simulated_examples))

mismatch_rate1 = len(engine.filter_mismatch(regex_target, simulated_examples)) / len(simulated_examples)
mismatch_rate2 = len(engine.filter_mismatch(regex_inferenced, target_examples)) / len(target_examples)
print('mismatch rate of simulated examples on target regex:', mismatch_rate1)
print('mismatch rate of target examples on inferenced regex:', mismatch_rate1)