import os
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain import LLMChain
from ..utils import make_verbose
__all__ = ['Chain']


class Chain:
    def __init__(self, openai_api_key=None, temperature=0.8):
        if openai_api_key is None:
            openai_api_key = os.environ["OPENAI_API_KEY"]
        self._openai_llm = OpenAI(
            openai_api_key=openai_api_key,
            temperature=temperature,
            model='text-davinci-003',  # https://platform.openai.com/docs/models/gpt-3-5
            client='regex_inference'
        )
        self._setup_lang_chains()

    def _setup_lang_chains(self):
        self.inference_regex = LLMChain(
            prompt=self.new_inference_prompt,
            llm=self._openai_llm
        )
        self.alter_regex = LLMChain(
            prompt=self.alter_regex_prompt,
            llm=self._openai_llm
        )
        self.simplify_regex = LLMChain(
            prompt=self.simplify_regex_prompt,
            llm=self._openai_llm
        )
        self.explain_regex = LLMChain(
            prompt=self.explain_regex_prompt,
            llm=self._openai_llm
        )

    @property
    def new_inference_prompt(self) -> PromptTemplate:
        template = """Question: Show me the best and shortest regex that can fully match the strings that I provide to you.
Note that:
*. The regex should be as short as possible.
*. Match sure the resulting regex does not have syntax error.
*. The regex should full match as many strings as possible.
*. The regex should not match strings that is not provided.
*. The number of string combinations matching the resulting regex should be as smaller than the number of target strings provided.
Now, each instance of the strings that should be fully matched is provided line-by-line and wrapped by double quotes as follows:
{strings}

Note that:
1. The double quote is not part of the string instance. Ignore the double quote during inferencing the regex.
2. Provide the resulting regex without wrapping it in quote

The resulting regex is: """
        prompt = PromptTemplate(
            template=template,
            input_variables=['strings']
        )
        return prompt

    @property
    def alter_regex_prompt(self) -> PromptTemplate:
        template = """Question: Alter the regex "{regex}" such that the following requirements is matched:
*. The pattern fully match the regex still fully match the regex.
*. The regex should full match as many strings provided as possible.
*. The regex should be as short as possible.
*. The regex should not match strings that is not provided except for those full match the original regex.
Now, each instance of the strings is provided line-by-line and wrapped by double quotes as follows:
{strings}

Note that:
1. The double quote is not part of the string instance. Ignore the double quote during inferencing the regex.
2. Provide the resulting regex without wrapping it in quote

The resulting altered regex is: """
        prompt = PromptTemplate(
            template=template,
            input_variables=['regex', 'strings']
        )
        return prompt

    @property
    def simplify_regex_prompt(self) -> PromptTemplate:
        template = """
Please revise the regex "{regex}"
such that the following constraint start with *. can be met:
*. The original regex consists of multiple regex seperated by "|". Try to combine the similar regex.
*. After combine, the resulting regex should be as short as possible.
*. The revised regex should still fully match all the strings full matched the original regex
*. The revised regex should still fully match each of the strings I provided to you.
Now, each instance of the strings is provided line-by-line and wrapped by double quotes as follows:
{strings}


Note that:
1. The double quote is not part of the string instance. Ignore the double quote during inferencing the regex.
2. Provide the resulting regex without wrapping it in quote

The resulting revise regex is:
"""
        prompt = PromptTemplate(
            template=template,
            input_variables=['regex', 'strings']
        )
        return prompt

    @property
    def explain_regex_prompt(self) -> PromptTemplate:
        template = """Question: Explain the regex "{regex}" such that
1. The role of each character in the regex is elaberated.
2. Provide 5 most interpretive example strings that fullmatch the regex.

The explaination is: """
        prompt = PromptTemplate(
            template=template,
            input_variables=['regex']
        )
        return prompt
