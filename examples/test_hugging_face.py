"""
Candidate:

* https://huggingface.co/bigcode/starcoder?text=Imagine+you+are+a+super+senior+computer+programmer+that+understand+a+lot+about+regex.+Now+I+will+provide+you+about+facts+about+some+existing+regex+and+some+patterns+to+match.+Reason+about+them+and+provide+the+answer+according+to+my+guide.%0A%0AFact+0%3A%0A%0AA+list+of+regex+describing+3+type+of+patterns+is+double+quoted+and+shown+as+the+following+bullet+points%3A%0A%0A1.+%22%5B1-4%5D%22%0A2.+%22%5B3-5%5D%22%0A3.+%22%5B5-8%5D%22%0A%0ANow%2C+I+will+provide+to+you+more+4+facts.%0A%0AFact+1%3A%0A%0AFor+regex+number+1%2C+it+correctly+match+the+patterns+double+quoted+and+shown+as+follows%3A%0A%0A%221%22%0A%222%22%0A%223%22%0A%0AHowever%2C+it+mistakenly+match+the+patterns+double+quoted+and+shown+as+follows%3A+%0A%0A%224%22%0A%0AFact+2%3A+%0A%0AFor+regex+number+2%2C+it+correctly+match+the+patterns+double+quoted+and+shown+as+follows%3A%0A%0A%224%22%0A%225%22%0A%0AHowever%2C+it+mistakenly+match+the+patterns+double+quoted+and+shown+as+follows%3A+%0A%0A%223%22%0A%0AFact+3%3A+%0A%0AFor+regex+number+3%2C+it+correctly+match+the+patterns+double+quoted+and+shown+as+follows%3A%0A%0A%226%22%0A%227%22%0A%228%22%0A%0AHowever%2C+it+mistakenly+match+the+patterns+double+quoted+and+shown+as+follows%3A+%0A%0A%225%22%0A%0AFact+4%3A+%0A%0ACombining+the+regex+with+%22%7C%22+mark+to+get+a+combined+regex.+%0AThe+combined+regex+does+not+but+should+also+matched+the+following+patterns+double+quoted+and+shown+as+follows%3A%0A%0A%229%22%0A%220%22%0A%0AI+demand+you+to+alter+each+regex+and+show+each+altered+regex+as+well+as+the+combined+altered+regex+as+answer.+%0A%0AThe+criteria+for+the+combined+altered+regex+is+that%3A%0A1.+The+combined+altered+regex+is+a+combination+of+3+altered+regex+separated+by+%22%7C%22.+%0A2.+Each+of+the+altered+regex+of+the+combined+altered+regex+corresponds+to+a+provided+regex+that+should+be+altered.+%0A3.+The+regex+provided+in+Fact+1-3+should+be+altered+such+that+the+combined+altered+regex+should+match+not+only+the+patterns+provided+in+Fact+1-3+but+also+those+provided+in+Fact+4.%0A4.+Do+not+introduce+new+regex+for+the+combined+regex.+%0A%0AThe+criteria+for+each+altered+regex+in+the+combined+altered+regex+is+that%3A%0A1.+Each+altered+regex+should+still+correctly+match+the+patterns+that+is+correctly+match.+For+example%2C+altered+regex+of+Fact+1+should+still+match+those+correctly+matched+patterns+of+Fact+1.+%0A2.+The+altered+regex+should+exclude+the+pattern+mistakenly+matched.+That+is%2C+those+mistakenly+matched+patterns+should+not+be+matched.+For+example%2C+after+alteration%2C+regex+of+Fact+1+should+not+match+those+mistakenly-matched+patterns+of+Fact+1.%0A%0A%0ADuring+answer+generation%2C+make+sure+that%3A%0A1.+The+regex+before+and+after+the+alteration+should+be+double+quoted.+%0A2.+The+regex+before+and+after+the+alteration+should+be+shown+line-by-line.%0A3.+The+regex+before+and+after+the+alteration+should+be+listed+in+the+same+line.+%0A4.+The+regex+before+and+after+the+alteration+should+be+separated+by+%22%2C%22+mark.%0A5.+The+combined+altered+regex+should+be+shown+in+the+final+line+single+quoted.%0A6.+Do+not+show+any+additional+text+besides+regex.+%0A7.+In+the+answer%2C+the+regex+before+the+alteration+should+not+be+different+from+those+provided+in+Fact+0.+%0A%0AAn+example+to+the+answer+is%3A%0A%0A%0A%22original_regex_1%22%2C%22altered_regex_1%22%0A%22original_regex_2%22%2C%22altered_regex_2%22%0A%22original_regex_3%22%2C%22altered_regex_3%22%0A%27%28altered_regex_1%29%7C%28altered_regex_2%29%7C%28altered_regex_3%29%27%0A%0AThe+answer+is%3A
* https://beebom.com/best-large-language-models-llms/
* https://huggingface.co/docs/transformers/tasks/language_modeling - fine tuning 


Candidate Models:
1. Salesforce/codegen-350M-mono (local)
2. EleutherAI/gpt-neox-20b (online)
3. stabilityai/stablecode-completion-alpha-3b-4k
4. bigcode/starcoder 
5. decapoda-research/llama-65b-hf
6. huggyllama/llama-7b

"""
from regex_inference.inference.chain import Chain
from regex_inference.inference.engine import Engine
from langchain import HuggingFacePipeline
from langchain import HuggingFaceHub
import os
import random
TRAIN_CNT = 10
whole_patterns = []
with open('data/version.txt', 'r') as f:
    whole_patterns = f.read().split('\n')
train_patterns = random.sample(whole_patterns, TRAIN_CNT)
eval_patterns = list(set(whole_patterns) - set(train_patterns))

# huggingfacehub_api_token = os.environ["HUGGINGFACEHUB_API_TOKEN"]
ONLINE = False
if ONLINE:
    llm = HuggingFaceHub(repo_id='gnsepili/coder-llama2-7b', 
                        model_kwargs={"max_new_tokens": 1000})
else:
    llm = HuggingFacePipeline.from_model_id(
        model_id='Salesforce/codegen25-7b-mono',
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 1000, "temperature": 0.8},
        model_kwargs={"trust_remote_code": True}
    )
# chain = Chain(use_openai=False, model_id='bigcode/starcoder', max_length=1000, temperature=0.1)
e = Engine(llm, verbose=True)
# chain = Chain(use_openai=False, model_id='meta-llama/Llama-2-70b-chat-hf', max_length=1000, temperature=0.8)
print('1.')
ans = e.get_regex_sequence(train_patterns)
print(ans)
