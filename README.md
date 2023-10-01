# Introduction:

Welcome to regex_inference!

regex_inference is a Python package dedicated to making regular expression (regex) inference a breeze. With the power of the ChatGPT model, this package can effortlessly derive regex patterns from a list of strings you provide. 

Here are some of the cool features you can expect:
- **Regex Inference**: Just feed in a list of strings, and regex_inference will do the heavy lifting to create a regex that fits your data. 
- **Multi-Threading Support**: We know that performance matters. That's why regex_inference is designed to fully support multi-threading, helping you maximize your interaction with ChatGPT.
- **Built-In Evaluator**: Wondering how effective your inferred regex is? regex_inference has you covered with a built-in evaluator. It calculates precision, recall, and the F1 score, providing you with a quantitative measure of your regex's performance. 

Whether you're a machine learning enthusiast, a data scientist, or a Python dev looking to further leverage the power of regex, regex_inference is here to make your life easier. We look forward to seeing the amazing things you'll do with this tool!

# Installation 

You can install regex_inference using pip:

```bash
pip install regex_inference
```
# Obtain API key and add it to the environment


1) Get an OpenAI api token following the link below:

https://www.maisieai.com/help/how-to-get-an-openai-api-key-for-chatgpt

2) Insert it as an environement variable. 
```bash
export OPENAI_API_KEY=<key>
```
# Basic Usage

Here's a simple guide on how you can use regex_inference package:

```python
from regex_inference import Evaluator, Inference
import random

# Define the number of training samples
TRAIN_CNT = 200

# Read the whole patterns from a txt file
whole_patterns = []
with open('data/version.txt', 'r') as f:
    whole_patterns = f.read().split('\n')

# Randomly sample some patterns for training
train_patterns = random.sample(whole_patterns, TRAIN_CNT)

# Use the remaining patterns for evaluation
eval_patterns = list(set(whole_patterns) - set(train_patterns))

# Create an instance of Inference class
inferencer = Inference(verbose=False, n_thread=1, engine='fado+ai')

# Run the inferencer on the training patterns
regex = inferencer.run(train_patterns)

# Evaluate the inferred regex
precision, recall, f1 = Evaluator.evaluate(regex, eval_patterns)

# Print the evaluation metrics
print('precision:', precision)
print('recall:', recall)
print('f1:', f1)
```

In the above code snippet, we first read the patterns from a file. Then, we randomly select some of these patterns for training and use the remaining for evaluation. After that, we create an instance of Inference class and run it on the training patterns to infer a regex. Lastly, we evaluate the inferred regex and print the evaluation metrics.

You can adjust the number of threads (n_thread) and choose a different engine (engine) as per your needs. The Inference class currently supports two engine mode: `fado+ai` and `ai`. 

The `fado+ai` engine do the inference by minimize a DFA of the training patterns, convert the DFA to regex, and ask ChatGPT to make to generalize to other similar patterns. 


The `ai` engine simply make inference by sending the training patterns to ChatGPT and ask it to produce a regex that match the training patterns. 

The `fado+ai` approach is cheaper than `ai` approach since it sends less amount of token to ChatGPT. 



# Contribute

We welcome contributions to regex_inference. Whether it's improving the documentation, adding new features, reporting bugs, or any other improvements, we appreciate all kinds of contributions. 

