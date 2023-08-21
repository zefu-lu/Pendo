import requests

from .base import BaseLlm, LlmUsage
from typing import List, Tuple
from .message import Message, MessageRole
from datetime import datetime


class LlamaLlm(BaseLlm):
    
    def __init__(self, llama_cpp_server_url: str, api_version: str = "v1", **kwargs) -> None:
        # TODO: Validate base_url
        assert llama_cpp_server_url.startswith("http://"), "LlamaLlm: base_url must be a valid http url (e.g. http://localhost:8000)"
        self.base_url = llama_cpp_server_url
        self.api_version = api_version

        super().__init__(**kwargs)
        

    def _prepare_messages(self, messages: List[Message]) -> Tuple[List[Message], LlmUsage]:
        return [{"role": message.role, "content": message.content} for message in messages]

    def chat_completion(self, messages: List[Message]) -> Message:
        messages = self._prepare_messages(messages)

        start_time = datetime.now()
        response = requests.post("/".join([self.base_url, self.api_version, "chat/completions"]), json={
            "messages": messages,
            "max_tokens": self.max_tokens,
        })
        if response.status_code != 200:
            print(response.text)
            raise ValueError(f"LlamaLlm: unexpected response code from server: {response.status_code}")
        end_time = datetime.now()

        result = response.json()

        usage = LlmUsage(result.get("usage", {}).get("completion_tokens", None), result.get("usage", {}).get("prompt_tokens", None), end_time - start_time)
        self.total_usage += usage
        
        reply_message = result.get("choices",[])[0].get("message", {})
        if reply_message["role"] == "assistant":
            return (Message(MessageRole.ASSISTANT, reply_message["content"].strip()), usage)
        if reply_message["role"] == "function":
            return (Message(MessageRole.FUNCTION, reply_message["content"].strip()), usage)
        
        raise ValueError(f"OpenAILlm: unexpected role during chat completion: {reply_message['role']}")

    
    def completion(self, prompt: str):
        raise NotImplementedError