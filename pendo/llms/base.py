from abc import ABC
from typing import List
from .message import Message
from dataclasses import dataclass
from datetime import timedelta

class BaseLlm(ABC):
    def __init__(self, max_tokens=4096):
        self.max_tokens = max_tokens
        self.total_usage = LlmUsage(completion_tokens=0, prompt_tokens=0, response_time=timedelta(0))
    
    def chat_completion(self, messages : List[Message]) -> Message:
        raise NotImplementedError
    
    def completion(self, prompt: str):
        raise NotImplementedError

@dataclass
class LlmUsage:
    completion_tokens: int
    prompt_tokens: int
    response_time: timedelta

    @property
    def total_tokens(self):
        return self.completion_tokens + self.prompt_tokens

    def __add__(self, other):
        return LlmUsage(
            completion_tokens=self.completion_tokens + other.completion_tokens,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            response_time=self.response_time + other.response_time
        )
@dataclass
class StreamedChatCompletion(ABC):
    reply_message: Message = None
    usage: LlmUsage = None

    def generate(self):
        raise NotImplementedError