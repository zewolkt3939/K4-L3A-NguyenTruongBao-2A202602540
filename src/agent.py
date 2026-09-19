from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3, metadata_filter: dict | None = None) -> str:
        if metadata_filter:
            chunks = self.store.search_with_filter(question, top_k=top_k, metadata_filter=metadata_filter)
        else:
            chunks = self.store.search(question, top_k=top_k)
        if not chunks:
            return "Không tìm thấy thông tin phù hợp trong cơ sở tri thức để trả lời câu hỏi."

        context_blocks = []
        for idx, chunk in enumerate(chunks, 1):
            source = (
                chunk.get("metadata", {}).get("source")
                or chunk.get("metadata", {}).get("doc_id")
                or chunk.get("id")
                or f"doc_{idx}"
            )
            context_blocks.append(f"[{idx}] (Nguồn: {source})\n{chunk['content']}")

        context_text = "\n\n".join(context_blocks)
        prompt = (
            "Bạn là trợ lý AI trả lời câu hỏi dựa trên tài liệu được cung cấp.\n"
            "Hãy trả lời câu hỏi dựa trên ngữ cảnh dưới đây. Trích dẫn nguồn hoặc số thứ tự đoạn [1], [2] tương ứng.\n"
            "Nếu không có đủ thông tin để trả lời, hãy nêu rõ là không tìm thấy.\n\n"
            f"--- Ngữ cảnh ---\n{context_text}\n\n"
            f"--- Câu hỏi ---\n{question}\n\n"
            "--- Trả lời ---"
        )
        return self.llm_fn(prompt)
