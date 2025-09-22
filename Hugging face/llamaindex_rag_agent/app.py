# app_hf.py (alternative)
import os
from dotenv import load_dotenv
from retriever import load_guest_docs
from tools import GuestInfoRetrieverTool
from huggingface_hub import InferenceClient

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

docs = load_guest_docs()
tool = GuestInfoRetrieverTool(docs)

client = InferenceClient(token=HF_TOKEN)  # huggingface_hub.InferenceClient

def alfred_query(user_query: str):
    # 1) retrieve
    retrieved = tool.forward(user_query)  # plain text
    # 2) create prompt that includes retrieved facts
    prompt = f"Use these guest facts to answer the question.\n\nFacts:\n{retrieved}\n\nQuestion: {user_query}\nAnswer concisely:"
    # 3) call model (chat or text generation depending on model)
    # Use chat_completion for chat-capable models, otherwise text_generation
    # Example: chat-style
    response = client.chat_completion(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=[{"role":"system","content":"You are an assistant that answers using the facts provided."},
                  {"role":"user","content": prompt}],
        max_tokens=200
    )
    return response.choices[0].message["content"]

if __name__ == "__main__":
    print(alfred_query("Tell me about 'Lady Ada Lovelace'."))
