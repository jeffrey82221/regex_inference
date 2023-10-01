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
# Add API key from OpenAI


1) You can get an OpenAI api key following the link below:

https://www.maisieai.com/help/how-to-get-an-openai-api-key-for-chatgpt

2) Then, simply insert it to your environement:
```bash
export OPENAI_API_KEY=<key>
```

# Getting Started with regex_inference

The regex_inference package is a powerful tool for inferring regular expressions (regex) from a set of training patterns. Here's a step-by-step guide on how to use it:

```python
from regex_inference import Evaluator, Inference
import random

# Define the number of training samples
TRAIN_CNT = 200

# Load patterns from a text file
with open('data/version.txt', 'r') as f:
    whole_patterns = f.read().split('\n')

# Randomly select some patterns for training
train_patterns = random.sample(whole_patterns, TRAIN_CNT)

# Use the remaining patterns for evaluation
eval_patterns = list(set(whole_patterns) - set(train_patterns))

# Initialize an Inference object
inferencer = Inference(verbose=False, n_thread=1, engine='fado+ai')

# Generate a regex from a subset of the training patterns, with the rest used for validation
regex = inferencer.run(train_patterns[:100], val_patterns=train_patterns[100:])

# Evaluate the inferred regex
precision, recall, f1 = Evaluator.evaluate(regex, eval_patterns)

# Print the evaluation results
print(f'Precision: {precision}\nRecall: {recall}\nF1 Score: {f1}')
```

In this example, after loading patterns from a text file, we randomly select some of these patterns for training. We further divide the training set into a subset for training and another for validation. The validation patterns (`val_patterns`) guide the selection of the best regex from the candidates generated by ChatGPT. The remaining patterns are used for evaluation.

The `Inference` object is customizable. You can adjust the number of threads (`n_thread`), which corresponds to the number of regex candidates obtained from ChatGPT. The higher the `n_thread` value, the more candidates you get, but note that this also increases the inference cost. You can also select the inference engine (`engine`), with options being `fado+ai` and `ai`.

The `fado+ai` engine minimizes a DFA (Deterministic Finite Automaton) of the training patterns, converts the DFA to a regex, and then uses ChatGPT to generalize to other similar patterns. The `ai` engine sends the training patterns directly to ChatGPT, asking it to produce a regex matching the patterns. The `fado+ai` approach is generally more economical than the `ai` approach, as it sends fewer tokens to ChatGPT.

# Contribute

We welcome contributions to regex_inference. Whether it's improving the documentation, adding new features, reporting bugs, or any other improvements, we appreciate all kinds of contributions. 

