from pendo.core import load_config, initialize_workspace_paths
from pendo.llms import register_llms
from pendo.dataloaders import BaseDataloader, get_dataloader
from pendo.indexers import BaseIndexer, register_indexers, get_indexer
from datetime import datetime, timedelta
from tqdm.asyncio import tqdm_asyncio
import asyncio
import tiktoken

async def main():
    initialize_workspace_paths()
    
    config = load_config()
    register_llms(config["llms"])
    register_indexers(config["indexers"])

    dataloaders_config = config["dataloaders"]
    for k, v in dataloaders_config.items():

        dataloader = get_dataloader(v["type"], k, config=v["config"], tokenizer=tiktoken.get_encoding("cl100k_base"))
        print(f"{k}: checking docs after {dataloader.get_timestamp()}")
        doc_ids = await dataloader.retrieve_doc_ids()
        print(f"{k}: {len(doc_ids)} docs need to be indexed")
        now = datetime.now()

        if len(doc_ids) == 0:
            dataloader.save_timestamp(now)
            continue

        print(f"{k}: gathering and chunking docs")
        docs = await tqdm_asyncio.gather(*[dataloader.retrieve_chunked_doc(doc_id) for doc_id in doc_ids])
        print(f"{k}: indexing {len(docs)} docs")

        for indexer_name in v.get("indexers", []):
            print(f"{k}: indexing to `{indexer_name}`")
            indexer = get_indexer(indexer_name)
            await indexer.index_docs(docs)
            print(f"{k}: `{indexer_name}` updated")
        dataloader.save_timestamp(now)
    
    


if __name__ == "__main__":
    asyncio.run(main())