from random_regex import RegexGenerator
from regex_inference import Inference, Evaluator
from langchain import HuggingFacePipeline
from langchain import HuggingFaceHub
ONLINE = True
if ONLINE:
    llm = HuggingFaceHub(
        repo_id='stabilityai/stablecode-completion-alpha-3b-4k', 
        model_kwargs={"temperature": 0.8, "max_length": 500}
    )
else:
    llm = HuggingFacePipeline.from_model_id(
        model_id="stabilityai/stablecode-completion-alpha-3b-4k",
        task="text-generation",
        model_kwargs={"temperature": 0.8, "max_length": 500},
    )


regex_gen = RegexGenerator(max_complexity=30, bloom_max_item_count=1000)
regex_instance = next(regex_gen.generate())
regex_target = regex_instance['regex']
target_examples = regex_instance['examples']
print('regex target:', regex_target)
print('# of target examples:', len(target_examples))
inference = Inference(llm, verbose=True)
regex_inferenced = inference.run(regex_instance['examples'])
print('inferenced_regex:', regex_inferenced)
precision, recall, f1 = Evaluator.evaluate_regex_list(
    [regex_inferenced], target_examples)
print('precision:', precision)
print('recall:', recall)
print('f1:', f1)
