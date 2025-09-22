# tools.py
from langchain_community.retrievers import BM25Retriever
from smolagents import Tool  # example agent-tool framework from your notes

class GuestInfoRetrieverTool(Tool):
    name = "guest_info_retriever"
    description = "Retrieves guest information by name, relation, or short query."
    inputs = {
        "query": {"type": "string", "description": "Name or relation to search for"}
    }
    output_type = "string"

    def __init__(self, docs):
        # BM25Retriever expects a list of Document objects
        self.retriever = BM25Retriever.from_documents(docs)

    def forward(self, query: str) -> str:
        hits = self.retriever.get_relevant_documents(query)
        if not hits:
            return "No matching guest information found."
        return "\n\n".join(h.page_content for h in hits[:3])
