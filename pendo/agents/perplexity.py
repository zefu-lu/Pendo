from pendo.llms import Message, MessageRole, BaseLlm
import asyncio
import itertools

_PROMPT = """
        Follow exactly those 3 steps:
        1. Read the document snippets above
        2. Answer the question at the end using only the snippets.
        3. For each sentence you reply, cite the snippet number that you used to answer the question at the end. For example, if you used snippet 2 and 3 in your answer, append [2, 3] to the sentence.

        If you don't think the snippets are relevant and are unsure of the answer, reply that you don't know about this topic and are always learning. Otherwise, reply with as much detail as possible.

        Now answer the question from the user:
        """

class PerplexitySearchAgent():
    def __init__(self, llm: BaseLlm, tokenizer, summary_index, chunk_index, temperature=0.5, shortlisting_threshold = 0.8, n_summary_results=20, n_chunk_results=50, max_context_tokens=12288):
        self.llm = llm
        self.tokenizer = tokenizer
        self.summary_index = summary_index
        self.chunk_index = chunk_index
        self.temperature = temperature
        self.shortlisting_threshold = shortlisting_threshold
        self.n_summary_results = n_summary_results
        self.n_chunk_results = n_chunk_results
        self.max_context_tokens = max_context_tokens

    def _generate_search_queries(self, query):
        messages = [Message(MessageRole.SYSTEM, "Generate search engine queries for the question that the user is asking. Return the queries in the form of a list separated by ; . For example, if the user asks 'What is the capital of France?', you can return 'capital of France; France capital city'. Return the queries only, do not answer the question directly. Return no more than 6 queries.\n")]
        messages.append(Message(MessageRole.USER, query))
        reply, usage = self.llm.chat_completion(messages, temperature=0.5)

        return reply.content.split(";"), usage

    async def _retrieve_relevant_docs(self, query):
        candidates = self.summary_index.query(query_texts=[query], n_results=self.n_summary_results)
        
        docs = []
        for idx, doc_id, distance, metadata in zip(range(1, self.n_summary_results+1), candidates["ids"][0], candidates["distances"][0], candidates["metadatas"][0]):
            score = 1.0/idx
            docs.append({"id": doc_id, "score": score, "distance": distance, "metadata": metadata})
        return docs
    
    async def _retrieve_shortlisted_docs(self, quries):
        agg_candidates = await asyncio.gather(*(self._retrieve_relevant_docs(q) for q in quries))

        doc_candidates = {}
        for cand in itertools.chain(*agg_candidates):
            if cand["id"] not in doc_candidates:
                doc_candidates[cand["id"]] = cand
                doc_candidates[cand["id"]].pop("distance")
            else:
                doc_candidates[cand["id"]]["score"] += cand["score"]
        doc_candidates = list(doc_candidates.values())
        doc_candidates.sort(key=lambda x: x["score"], reverse=True)

        threshold = doc_candidates[0]["score"] - self.shortlisting_threshold * (doc_candidates[0]["score"] - doc_candidates[-1]["score"])
        return [cand for cand in doc_candidates if cand["score"] >= threshold]

    async def _retrieve_snippets_from_doc(self, query, doc_id):
        results = self.chunk_index.query(query_texts=[query], where={"doc_id": {"$eq": doc_id}}, n_results=self.n_chunk_results)
        snippets = []
        for sid, distance, metadata, document in list(zip(results["ids"][0], results["distances"][0], results["metadatas"][0], results["documents"][0])):
            doc_id = metadata["doc_id"]
            snippets.append({
                "id": sid,
                "distance": distance,
                "content": document,
                "metadata": metadata,
            })
        return snippets

    async def _retrieve_relevant_snippets(self, query, shortlisted_doc_ids):
        agg_snippets = await asyncio.gather(*(self._retrieve_snippets_from_doc(query, doc_id) for doc_id in shortlisted_doc_ids))
        snippets = [s for s in itertools.chain(*agg_snippets)]
        snippets.sort(key=lambda x: x["distance"])

        doc_ids = set()
        shortlisted_snippets = []
        current_tokens = 0
        for snippet in snippets:
            current_tokens += len(self.tokenizer.encode(snippet["content"]))
            if current_tokens > self.max_context_tokens:
                break
            doc_ids.add(snippet["metadata"]["doc_id"])
            shortlisted_snippets.append(snippet)
        
        result = {}
        for doc_id in doc_ids:
            targets = sorted([s for s in shortlisted_snippets if s["metadata"]["doc_id"] == doc_id], key=lambda x: x["metadata"]["chunk_id"])
            if len(targets) == 0:
                continue
            result[doc_id] = " ".join([t["content"] for t in targets])
        return result



    async def run(self, query):
        search_queries, usage = self._generate_search_queries(query)
        yield Message(MessageRole.SYSTEM, f"Expanding your queries: \n {';'.join(search_queries)}\n"), usage

        shortlisted_docs = await self._retrieve_shortlisted_docs(search_queries)

        if len(shortlisted_docs) == 0:
            yield Message(MessageRole.SYSTEM, "No relevant documents found.\n"), None
            return
        
        message = f"Considering the following {len(shortlisted_docs)} relevant documents.\n"
        for doc in shortlisted_docs:
            message += f"{doc['score']:.2f} \t {doc['metadata']['title']}\n"
        yield Message(MessageRole.SYSTEM, message), None

        snippets = await self._retrieve_relevant_snippets(query, [doc["id"] for doc in shortlisted_docs])
        context = []
        for idx, doc in enumerate(shortlisted_docs):
            if snippets.get(doc["id"], None) is None:
                continue
            context.append(f"======\n document {idx+1}\n title {doc['metadata']['title']}\n {snippets[doc['id']]}\n")

        messages = [Message(MessageRole.SYSTEM, "\n".join(context))]
        messages.append(Message(MessageRole.SYSTEM, _PROMPT))
        messages.append(Message(MessageRole.USER, query))

        yield self.llm.chat_completion(messages, temperature=self.temperature)