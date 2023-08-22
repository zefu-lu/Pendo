from .message import *
from .openai import OpenAILlm
from .llama import LlamaLlm
from .base import LlmUsage, BaseLlm

LLM_MAPPING = {
    "openai": OpenAILlm,
    "llama": LlamaLlm,
}

REGISERED_LLM = {}

def register_llms(config):
    REGISERED_LLM.update(config)

def get_llm(registered_name):
    if registered_name not in REGISERED_LLM:
        raise ValueError(f"Unknown llm: {registered_name}")
    llm_type = REGISERED_LLM[registered_name]["type"]
    kwargs = REGISERED_LLM[registered_name]["params"]
    return LLM_MAPPING[llm_type](**kwargs)
