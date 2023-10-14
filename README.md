# Introduction:

Welcome to regex_inference!

This is a Python package dedicated to making regular expression (regex) inference a breeze. With the power of the ChatGPT model, this package can effortlessly derive regex patterns from a list of strings you provide. 

Here are some of the cool features you can expect:

- **Regex Inference**: Give regex_inference a list of strings and it swiftly outputs a suitable regex for your data, eliminating the need to grapple with complex regex syntax and saving you precious time.

- **Built-In Evaluator**: regex_inference comes equipped with a built-in evaluator that provides a quantitative measure of your regex's performance by calculating precision, recall, and the F1 score in a snap.

- **Multi-Threaded Regex Candidate Generation**: By leveraging Python's multi-threading capabilities, regex_inference can generate multiple regex candidates simultaneously through parallel calls to ChatGPT within permissible rate limits, ensuring efficient and quick regex generation.

- **Post-Generation Evaluation and Selection**: After generating the regex candidates, regex_inference evaluates each one based on their F1 scores against validation patterns and selects the best performing regex, ensuring maximum efficiency and human-like pattern recognition with minimal effort.


Whether you're a machine learning enthusiast, a data scientist, or a Python dev looking to further leverage the power of regex, regex_inference is here to make your life easier. We look forward to seeing the amazing things you'll do with this tool!

Sure, here's how you might address those questions in your README:


## Why leverage ChatGPT for regex inference?

Traditional rule-based methods for regex inference rely on predefined rules and patterns, which can be limiting. These methods may struggle with complex or unusual cases and are often not very flexible. 

On the other hand, ChatGPT is a transformer-based language model trained on a diverse range of internet text. Its ability to predict the next word in a sentence can be adapted for our purpose of predicting the next character in a regex pattern. This makes it incredibly versatile and capable of handling a wider range of patterns compared to rule-based methods.

In addition, using ChatGPT allows `regex_inference` to leverage the model's understanding of semantics and context. This means our package can generate regex patterns that not only match the strings provided but also capture the underlying pattern in a way that is meaningful and intuitive for humans.

## Use Cases in Data-Related Industries

Regular expressions are a powerful tool for working with text data. They can be used for tasks like data extraction, data cleaning, and data validation. However, writing regex patterns can be complex and time-consuming, especially for those who are not familiar with regex syntax.

`regex_inference` can be a game changer in such scenarios. By automating the process of regex generation, it allows data scientists, analysts, and other professionals to focus on their core work without getting bogged down in the intricacies of regex syntax. This can lead to significant time savings and productivity improvements.

Consider a scenario where a company receives large amounts of text data in various formats and needs to extract specific information from this data. Instead of manually writing and tweaking regex patterns, they could use `regex_inference` to generate the necessary patterns automatically. This not only simplifies the task but also makes the process more reliable and repeatable.

# Installation 

You can install regex_inference using pip:

```bash
pip install regex_inference
```
# Configuration

## OpenAI API Key

Before you start using `regex_inference`, you'll need to obtain an OpenAI API key. Here's how you can do it:

1. Follow the guide on this page to get your OpenAI API key: [How to get an OpenAI API Key for ChatGPT](https://www.maisieai.com/help/how-to-get-an-openai-api-key-for-chatgpt)
2. Export the key to your environment:

```bash
export OPENAI_API_KEY=<your_key>
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
inferencer = Inference(verbose=False, n_thread=3, engine='fado+ai')

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

# Contributing

We welcome your contributions to `regex_inference`! Whether you're improving the documentation, adding new features, reporting bugs, or making other enhancements, your input is greatly appreciated. 

# Contact

If you have any questions, feature requests, or just want to chat, feel free to reach out to me at [jeffrey82221@gmail.com](mailto:jeffrey82221@gmail.com) or open an issue on our GitHub page.


# License

This project is licensed under the terms of the MIT License. For more details, see the [LICENSE](LICENSE) file in the repository.


