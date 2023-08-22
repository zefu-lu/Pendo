from .base import BaseDataloader, ChunkedDoc

from notion_client import AsyncClient, Client
from typing import Dict, List
from datetime import datetime

import logging
import asyncio

BLOCKS_IGNORED = [
    "unsupported",
    "breadcrumb",
    "child_database",
    "child_page",
    "column_list",
    "column",
    "embed",
    "file",
    "image",
    "link_preview",
    "link_to_page",
    "pdf",
    "mention",
    "synced_block",
    "table",
    "table_row",
    "template",
    "video"
]

BLOCKS_UNSEPARABLE = [
    "bulleted_list_item",
    "numbered_list_item",
    "to_do",
]

BLOCKS_HEADINGS = [
    "heading_1",
    "heading_2",
    "heading_3",
]

BLOCKS_CONTENT = [
    "bookmark",
    "callout",
    "code",
    "equation",
    "paragraph",
    "quote",
    "toggle"
]

BLOCKS_BREAK = [
    "divider",
    "table_of_contents"
]


def _parse_relation_prop(prop: Dict, client: Client) -> str:
    target_ids = [sec["id"] for sec in prop.get("relation", [])]
    results = []
    related_pages = [client.pages.retrieve(page_id = id) for id in target_ids]
    for page in related_pages:
        for prop in page["properties"].values():
            if prop["type"] == "title":
                results.append(" ".join([sec.get("plain_text", "") for sec in prop.get("title", [])]))
    return ";".join(results)


PROP_PRASER_MAPPER = {
    "title": lambda prop, client: " ".join([sec.get("plain_text", "") for sec in prop.get("title", [])]),
    "relation": _parse_relation_prop,
    "last_edited_time": lambda prop, client: prop.get("last_edited_time", ""),
    "date": lambda prop, client: prop.get("date", {}).get("start", ""),
    "rich_text": lambda prop, client: " ".join([sec.get("plain_text", "") for sec in prop.get("rich_text", [])]),
}


class NotionDataloader(BaseDataloader):

    def __init__(self, name, config, tokenizer):
        super().__init__(name, config, tokenizer)

        notion_token = config.get("notion_token", None)
        try:
            self.notion_client = AsyncClient(auth=notion_token)
        except Exception as e:
            logging.error(f"Unable to initiate Notion client: {e}")
            raise e
        try:
            self.sync_notion_client = Client(auth=notion_token)
        except Exception as e:
            logging.error(f"Unable to initiate Notion client: {e}")
            raise e

        self.db_id = config.get("database_id", None)
        self.title_prop = config.get("title_prop", None)
        self.last_edited_prop = config.get("last_edited_prop", None)
        assert self.db_id is not None, "Database ID not provided"
        assert self.title_prop is not None, "Title property not provided"

        self.metadata_map = {}
        for mt in config.get("metadata", {}):
            self.metadata_map[mt["key"]] = mt



    """
    Retrieve all page_ids from the target Notion database
    """
    async def retrieve_doc_ids(self, after: datetime = None) -> List[str]:
        doc_ids = []
        start_cursor = None

        if after is None:
            after = self.get_timestamp()

        while True:
            response = await self.notion_client.databases.query(
                database_id=self.db_id,
                start_cursor=start_cursor,
                page_size=100,  # this is the maximum page size allowed by the Notion API
                filter={
                    "property": self.last_edited_prop,
                    "date": {
                        "after": after.isoformat() if after is not None else None
                    }
                }
            )
            doc_ids.extend([doc["id"] for doc in response["results"]])

            if "next_cursor" in response and response["next_cursor"] is not None:
                start_cursor = response["next_cursor"]
            else:
                break

        return doc_ids

    async def retrieve_chunked_doc(self, doc_id: str) -> ChunkedDoc:
        async with self.semaphore:
            page = await self.notion_client.pages.retrieve(page_id=doc_id)

        properties = page["properties"]
        title = self._parse_prop(properties.get(self.title_prop, {}))
        last_edited = self._parse_prop(properties.get(self.last_edited_prop, {}))

        metadata = {}
        for mt in self.metadata_map.values():
            parsed_prop = self._parse_prop(properties.get(mt["property_name"], {}))
            if parsed_prop is not None:
                metadata[mt["key"]] = parsed_prop

        blocks = await self.notion_client.blocks.children.list(block_id = doc_id)

        chunks = self._chunk_blocks(blocks["results"], self.max_tokens)

        return ChunkedDoc(
            id=doc_id,
            title=title,
            last_edited_time=last_edited,
            chunks=chunks,
            metadata=metadata
        )

    def _parse_prop(self, prop):
        prop_type = prop.get("type", "")
        if prop_type not in PROP_PRASER_MAPPER:
            return None
        if prop[prop_type] is None:
            return None
        return PROP_PRASER_MAPPER[prop_type](prop, self.sync_notion_client)


    def _chunk_blocks(self, blocks, max_tokens):
        chunks = []
        current_chunk = []
        current_size = 0
        last_item_size = 0

        cache = []
        cache_size = 0
        cache_type = None

        def get_plain_text(rich_text):
            plain_text = ""
            for t in rich_text:
                plain_text += t["plain_text"]
            return plain_text

        def add_to_chunk(text, size):
            nonlocal current_chunk, current_size, last_item_size
            last_item_size = size
            current_chunk.append(text)
            current_size += size

        def reset_chunk():
            nonlocal current_chunk, current_size, chunks, last_item_size
            if len(current_chunk) > 0:
                chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_size = 0
            last_item_size = 0

        def reset_chunk_and_pop_last():
            nonlocal current_chunk, current_size, chunks, last_item_size
            if len(current_chunk) > 2:
                chunks.append("\n".join(current_chunk[:-1]))
                current_chunk = [current_chunk[-1]]
                current_size = last_item_size

        def add_to_cache(text, size):
            nonlocal cache, cache_size
            cache.append(text)
            cache_size += size

        def reset_cache():
            nonlocal cache, cache_size, current_chunk, current_size, cache_type
            if cache_size + current_size > max_tokens:
                if cache_size > current_size or current_size == 0:
                    reset_chunk()
                else:
                    reset_chunk_and_pop_last()
            add_to_chunk("\n".join(cache), cache_size)
            cache = []
            cache_size = 0
            cache_type = None

        for block in blocks:
            # Reset cache, if cache type changes
            if block["type"] != cache_type and not (cache_type == "heading" and block["type"] in BLOCKS_HEADINGS):
                reset_cache()

            # Ignore unnecessary blocks
            if block["type"] in BLOCKS_IGNORED:
                continue

            # If unseparable block, cache first
            if block["type"] in BLOCKS_UNSEPARABLE:
                rich_text = get_plain_text(block.get(block["type"],{}).get("rich_text", []))
                num_tokens = len(self.tokenizer.encode(rich_text))
                add_to_cache(rich_text, num_tokens)
                cache_type = block["type"]
                continue
            
            # If is divider, break the chunk
            if block["type"] in BLOCKS_BREAK:
                if len(current_chunk) > 0:
                    reset_chunk()
                continue

            # If is heading, break the chunk and cache
            if block["type"] in BLOCKS_HEADINGS:
                #if len(current_chunk) > 0:
                #    reset_chunk()
                rich_text = get_plain_text(block.get(block["type"],{}).get("rich_text", []))
                num_tokens = len(self.tokenizer.encode(rich_text))
                add_to_cache(rich_text, num_tokens)
                cache_type = "heading"
                continue
            
            # Ignore all other block types
            if block["type"] not in BLOCKS_CONTENT:
                continue
            
            # logic for content blocks

            caption = "".join(block.get(block["type"],{}).get("caption", []))
            rich_text = get_plain_text(block.get(block["type"],{}).get("rich_text", []))
            expression = block.get(block["type"],{}).get("expression", "")
            if len(caption) > 0:
                caption = f"<caption> {caption} <\caption> "
            if len(expression) > 0:
                expression = f"<equation> {expression} <\equation> "
            
            plain_text = caption + rich_text + expression
            num_tokens = len(self.tokenizer.encode(plain_text))

            if num_tokens + current_size > max_tokens:
                reset_chunk()
            add_to_chunk(plain_text, num_tokens)

        if len(cache) > 0:
            reset_cache()
        if len(current_chunk) > 0:
            reset_chunk()

        return chunks