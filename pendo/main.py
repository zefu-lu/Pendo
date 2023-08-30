from pendo.core import load_config, initialize_workspace_paths
from pendo.llms import register_llms, get_llm
from pendo.dataloaders import get_dataloader
from pendo.indexers import register_indexers, get_indexer
from pendo.agents import PerplexitySearchAgent

from datetime import datetime
from tqdm.asyncio import tqdm_asyncio

import asyncio
import tiktoken

async def main():
    initialize_workspace_paths()
    
    config = load_config()
    register_llms(config["llms"])
    register_indexers(config["indexers"])

    dataloaders_config = config["dataloaders"]
    local_timezone = datetime.now().astimezone().tzinfo
    

    for k, v in dataloaders_config.items():

        dataloader = get_dataloader(v["type"], k, config=v["config"], tokenizer=tiktoken.get_encoding("cl100k_base"))
        print(f"{k}: checking docs after {dataloader.get_timestamp()}")
        doc_ids = await dataloader.retrieve_doc_ids()
        print(f"{k}: {len(doc_ids)} docs need to be indexed")
        timestamp = datetime.now(tz=local_timezone)

        if len(doc_ids) == 0:
            dataloader.save_timestamp(timestamp)
            continue

        print(f"{k}: gathering and chunking docs")
        docs = await tqdm_asyncio.gather(*[dataloader.retrieve_chunked_doc(doc_id) for doc_id in doc_ids])
        docs = list(filter(lambda item: item is not None, docs))
        print(f"{k}: indexing {len(docs)} docs")

        for indexer_name in v.get("indexers", []):
            print(f"{k}: indexing to `{indexer_name}`")
            indexer = get_indexer(indexer_name)
            await indexer.index_docs(docs)
            print(f"{k}: `{indexer_name}` updated")
        dataloader.save_timestamp(timestamp)
    
    agent = PerplexitySearchAgent(get_llm("openai-gpt3.5-16k"), tiktoken.get_encoding("cl100k_base"), get_indexer("summary").index, get_indexer("chunks").index)

    while True:
        query = input("> ")
        total_usage = None
        async for message, usage in agent.run(query):
            if usage is not None:
                total_usage = usage if total_usage is None else total_usage + usage
            print(message)
            print("\033[2m" + str(total_usage) + "\033[0m")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())