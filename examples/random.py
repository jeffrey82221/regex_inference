from random_regex import RegexGenerator
from regex_inference import Inference, Evaluator
regex_gen = RegexGenerator(max_complexity=30)
regex_instance = next(regex_gen.generate())
regex_target = regex_instance['regex']
target_examples = regex_instance['examples']
print('regex target:', regex_target)
print('# of target examples:', len(target_examples))
inference = Inference(verbose=True, engine='ai')
regex_inferenced = inference.run(regex_instance['examples'])
print('inferenced_regex:', regex_inferenced)
precision, recall, f1 = Evaluator.evaluate(
    regex_inferenced, target_examples)
print('precision:', precision)
print('recall:', recall)
print('f1:', f1)
print(inference.openai_summary)
