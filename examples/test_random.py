from random_regex import RegexGenerator
from regex_inference import Engine
regex_gen = RegexGenerator(max_complexity=30, bloom_max_item_count=1000)
engine = Engine(verbose=False, simpify_regex=True)
regex_instance = next(regex_gen.generate())
regex_target = regex_instance['regex']
target_examples = regex_instance['examples']
print('regex target:', regex_target)
print('# of target examples:', len(target_examples))
regex_inferenced = engine.run(regex_instance['examples'])
