# agent_example.py
import asyncio
from huggingface_hub import InferenceClient

# Initialize Hugging Face Inference Client with your model
client = InferenceClient("Qwen/Qwen2.5-Coder-32B-Instruct", token="HF_TOKEN")

async def main():
    print("Running Hugging Face agent example...\n")

    # Example conversation
    response = client.chat_completion(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is 2 times 2?"}
        ],
        max_tokens=50,
        temperature=0.2,
    )

    # response.choices[0].message.content → similar to OpenAI
    print("Assistant:", response.choices[0].message["content"])

if __name__ == "__main__":
    asyncio.run(main())
