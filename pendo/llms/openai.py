import openai

from .base import BaseLlm, LlmUsage, StreamedChatCompletion
from typing import List, Tuple
from .message import Message, MessageRole
from datetime import datetime

# TODO Stream
class OpenAIStreamedChatCompletion(StreamedChatCompletion):

    def __init__(self, streamed_reply, **kwargs):
        super().__init__(**kwargs)
        self.streamed_reply = streamed_reply

    def generate(self):
        start_time = datetime.now()
        for chunk in self.streamed_reply:
            yield chunk
        end_time = datetime.now()


class OpenAILlm(BaseLlm):
    
    def __init__(self, openai_api_key: str, model: str = "gpt-3.5-turbo", **kwargs) -> None:
        self._openai_api_key = openai_api_key
        self._model = model
        assert self._openai_api_key != "", "OpenAILlm: openai_api_key is empty"
        assert self._model != "", "OpenAILlm: model is empty"
        super().__init__(**kwargs)

    def _prepare_messages(self, messages: List[Message]) -> Tuple[List[Message], LlmUsage]:
        return [{"role": message.role, "content": message.content} for message in messages]

    def chat_completion(self, messages: List[Message], stream=False, **kwargs) -> Message:
        messages = self._prepare_messages(messages)

        start_time = datetime.now()
        result = openai.ChatCompletion.create(
            api_key = self._openai_api_key,
            model=self._model,
            messages = messages,
            #max_tokens = self.max_tokens,
            **kwargs
        )
        end_time = datetime.now()

        usage = LlmUsage(result.usage.completion_tokens, result.usage.prompt_tokens, end_time - start_time)
        self.total_usage += usage
        
        reply_message = result.choices[0].message
        if reply_message.role == "assistant":
            return (Message(MessageRole.ASSISTANT, reply_message.content.strip()), usage)
        if reply_message.role == "function":
            return (Message(MessageRole.FUNCTION, reply_message.content.strip()), usage)
        
        raise ValueError(f"OpenAILlm: unexpected role during chat completion: {reply_message.role}")

    async def chat_completion_async(self, messages: List[Message], **kwargs) -> Message:
        messages = self._prepare_messages(messages)

        start_time = datetime.now()
        result = await openai.ChatCompletion.acreate(
            api_key = self._openai_api_key,
            model=self._model,
            messages = messages,
            #max_tokens = self.max_tokens,
            **kwargs
        )
        end_time = datetime.now()

        usage = LlmUsage(result.usage.completion_tokens, result.usage.prompt_tokens, end_time - start_time)
        self.total_usage += usage
        
        reply_message = result.choices[0].message
        if reply_message.role == "assistant":
            return (Message(MessageRole.ASSISTANT, reply_message.content.strip()), usage)
        if reply_message.role == "function":
            return (Message(MessageRole.FUNCTION, reply_message.content.strip()), usage)
        
        raise ValueError(f"OpenAILlm: unexpected role during chat completion: {reply_message.role}")

    def completion(self, prompt: str):
        start_time = datetime.now()
        result = openai.Completion.create(
            api_key = self._openai_api_key,
            model=self._model,
            prompt = prompt,
           # max_tokens = self.max_tokens
        )
        end_time = datetime.now()

        usage = LlmUsage(result.usage.completion_tokens, result.usage.prompt_tokens, end_time - start_time)
        self.total_usage += usage

        return result.choices[0].text, usage