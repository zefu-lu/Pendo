from .base import BaseDataloader, ChunkedDoc
from datetime import datetime
from typing import Dict, List

class GmailDataloader(BaseDataloader):

    def __init__(self, name, config, tokenizer):
        super().__init__(name, config, tokenizer) 
        

    async def retrieve_doc_ids(self, after: datetime = None) -> List[str]:
        raise NotImplementedError

    async def retrieve_chunked_doc(self, doc_id: str) -> ChunkedDoc:
        raise NotImplementedError