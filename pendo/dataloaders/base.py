from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List
from pendo.core import TIMESTAMPS_PATH
from asyncio import Semaphore

from datetime import datetime

@dataclass
class ChunkedDoc():
    id: str
    title: str
    last_edited_time: str
    chunks: List[str]
    metadata: Dict[str, str] = None

class BaseDataloader(ABC):

    def __init__(self, name, config, tokenizer):
        self.name = name
        self.config = config
        self.tokenizer = tokenizer
        self.max_tokens = config.get("max_tokens", 1024)
        self.semaphore = Semaphore(20)
        

    @abstractmethod
    async def retrieve_doc_ids(self, after: datetime = None) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    async def retrieve_chunked_doc(self, doc_id: str) -> ChunkedDoc:
        raise NotImplementedError

    def get_timestamp(self) -> datetime:
        timestamp_file = TIMESTAMPS_PATH / f"{self.name}.txt"
        if timestamp_file.exists():
            with open(timestamp_file, "r") as f:
                x = f.read()
                x = x.strip()
                return datetime.fromisoformat(x)
        return datetime.fromisoformat("1970-01-01T00:00:00+00:00")
    
    def save_timestamp(self, timestamp: datetime):
        timestamp_file = TIMESTAMPS_PATH / f"{self.name}.txt"

        with open(timestamp_file, "w") as f:
            f.write(timestamp.isoformat())