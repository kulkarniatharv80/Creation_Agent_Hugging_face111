# retriever.py
from datasets import load_dataset
# Prefer langchain Document; adjust if your project uses different Document class
try:
    from langchain.schema import Document
except Exception:
    from langchain_core.documents import Document  # fallback

def load_guest_docs(dataset_id="agents-course/unit3-invitees"):
    ds = load_dataset(dataset_id, split="train")
    docs = []
    for item in ds:
        page = "\n".join([
            f"Name: {item.get('name','')}",
            f"Relation: {item.get('relation','')}",
            f"Description: {item.get('description','')}",
            f"Email: {item.get('email','')}",
        ])
        docs.append(Document(page_content=page, metadata={"name": item.get("name","")}))
    return docs

if __name__ == "__main__":
    d = load_guest_docs()
    print(f"Loaded {len(d)} docs. Example:\n", d[0].page_content[:400])
