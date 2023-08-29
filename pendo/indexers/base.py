from abc import ABC, abstractmethod
from pendo.core import CHROMA_PATH
from pendo.dataloaders import ChunkedDoc
from typing import List
import chromadb

class BaseIndexer(ABC):
    def __init__(self, name, **kwargs):
        self.name = name
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

        self.index = self.chroma_client.get_or_create_collection(name=self.name, embedding_function=chromadb.utils.embedding_functions.DefaultEmbeddingFunction())

    @abstractmethod
    async def index_docs(docs: List[ChunkedDoc]):
        raise NotImplementedError

class DefaultIndexer(BaseIndexer):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    async def index_docs(self, docs: List[ChunkedDoc]):
        for doc in docs:
            print(doc.title)
            print("\n".join(doc.chunks))
            print("\n\n")
        