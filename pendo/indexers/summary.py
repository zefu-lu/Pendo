from .base import BaseIndexer
from pendo.dataloaders import ChunkedDoc
from pendo.llms import get_llm, Message, MessageRole

from asyncio import Semaphore
from typing import List
from tqdm.asyncio import tqdm_asyncio

class SummaryIndexer(BaseIndexer):

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.llm = get_llm(kwargs.get("llm", "openai-gpt3.5-16k"))
        self.semaphore = Semaphore(kwargs.get("llm_coroutines", 20))

        self.summary_prompt = "Read the article, and summarize in no more than 8 sentences on behalf of the author. Make sure to cover the main points of the article in the summary. "


    async def index_docs(self, docs: List[ChunkedDoc]):
        await tqdm_asyncio.gather(*[self._get_summary(doc) for doc in docs])

    async def _get_summary(self, doc, temperature=0.6):
        full_text = "\n".join(doc.chunks)
        async with self.semaphore:
            result, usage = await self.llm.chat_completion_async(
                    messages = [Message(MessageRole.SYSTEM, self.summary_prompt), Message(MessageRole.USER, full_text)],
                    temperature = temperature,
            )
        metadata = {"title": doc.title, "last_edited_time": doc.last_edited_time}
        for k, v in doc.metadata.items():
            if v is None:
                continue
            metadata[k] = v
        self.index.upsert(
            ids = [doc.id],
            metadatas = [metadata],
            documents = [result.content],
        )