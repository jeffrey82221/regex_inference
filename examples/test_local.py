from langchain import HuggingFacePipeline
from langchain import HuggingFaceHub
ONLINE = True
if ONLINE:
    llm = HuggingFaceHub(repo_id='stabilityai/stablecode-completion-alpha-3b-4k', 
                        model_kwargs={"max_new_tokens": 100})
else:
    llm = HuggingFacePipeline.from_model_id(
        model_id='Salesforce/codegen-350M-mono',
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 100}
    )
llm = HuggingFacePipeline.from_model_id(
    model_id="stabilityai/stablecode-completion-alpha-3b-4k",
    task="text-generation",
    model_kwargs={"temperature": 0, "max_length": 1000},
)
