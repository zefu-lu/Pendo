from .message import *
from .openai import OpenAILlm
from .llama import LlamaLlm
from .base import LlmUsage

LLM_MAPPING = {
    "openai": OpenAILlm,
    "llama": LlamaLlm,
}

def get_llm(llm_type: str, **kwargs):
    if llm_type not in LLM_MAPPING:
        raise ValueError(f"Unknown llm_type: {llm_type}")
    return LLM_MAPPING[llm_type](**kwargs)
