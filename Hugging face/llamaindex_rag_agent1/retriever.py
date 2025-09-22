import datasets
from langchain.schema import Document

def load_guest_docs():
    guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")
    docs = [
        Document(
            page_content="\n".join([
                f"Name: {guest['name']}",
                f"Relation: {guest['relation']}",
                f"Description: {guest['description']}",
                f"Email: {guest['email']}"
            ]),
            metadata={"name": guest["name"]}
        )
        for guest in guest_dataset
    ]
    return docs
