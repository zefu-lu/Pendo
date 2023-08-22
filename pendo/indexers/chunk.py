from .base import BaseIndexer
from pendo.dataloaders import ChunkedDoc

from typing import List

class ChunkIndexer(BaseIndexer):

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    async def index_docs(self, docs: List[ChunkedDoc]):
        ids = []
        metadatas = []
        documents = []
        for doc in docs:
            for i, chunk in enumerate(doc.chunks):
                ids.append(f"{doc.id}_{i+1}")
                metadata = {
                    "title": doc.title,
                    "last_edited_time": doc.last_edited_time,
                    "doc_id": doc.id,
                    "chunk_id": i+1,
                }
                for k, v in doc.metadata.items():
                    if v is None:
                        continue
                    metadata[k] = v
                metadatas.append(metadata)
                documents.append(chunk)
        self.index.upsert(
            ids = ids, 
            metadatas= metadatas,
            documents= documents
        )