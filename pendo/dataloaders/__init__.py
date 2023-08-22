from .base import ChunkedDoc, BaseDataloader
from .notion import NotionDataloader

DATALOADER_MAPPER = {
    "notion": NotionDataloader,
}

def get_dataloader(dataloader_type: str, name: str, config, tokenizer):
    if dataloader_type not in DATALOADER_MAPPER:
        raise ValueError(f"Unknown dataloader: {dataloader_type}")
    return DATALOADER_MAPPER[dataloader_type](name, config, tokenizer)
