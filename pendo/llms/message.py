from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import NewType
import json

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"

@dataclass
class Message():
    """
    A message is simply a role and a string content. 
    e.g. {"role": "user", "content": "Hello world!"}
    Timestamp in ISO format is automatically added at the time of creation.
    """
    role: NewType("MessageRole", MessageRole)
    content: str
    timestamp : str = field(default_factory=lambda : datetime.now().isoformat())

    def __str__(self) -> str:
        return f"{str(self.role)}: {self.content}"

    def to_json(self):
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str):
        json_dict = json.loads(json_str)
        return cls(**json_dict)
