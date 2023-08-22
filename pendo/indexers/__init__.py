from .chunk import ChunkIndexer
from .summary import SummaryIndexer
from .base import BaseIndexer

INDEXER_MAPPER = {
    "summary": SummaryIndexer,
    "chunk": ChunkIndexer,
}

REGISTERED_INDEXERS = {}

def register_indexers(config):
    for k, v in config.items():
        indexer_type = v.get("type", None)
        kwargs = v.get("params", {})
        if kwargs is None:
            kwargs = {}
        index_name = v.get("index_name", None)
        if indexer_type is None:
            raise ValueError(f"Indexer type is not specified for {k}")
        if index_name is None:
            raise ValueError(f"Index name is not specified for {k}")
        if indexer_type not in INDEXER_MAPPER:
            raise ValueError(f"Unknown indexer type: {indexer_type}")
        REGISTERED_INDEXERS[k] = INDEXER_MAPPER[indexer_type](index_name, **kwargs)

def get_indexer(indexer_name):
    if indexer_name not in REGISTERED_INDEXERS:
        raise ValueError(f"Unknown indexer: {indexer_name}")
    return REGISTERED_INDEXERS[indexer_name]