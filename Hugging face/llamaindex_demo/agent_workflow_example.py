import os
import asyncio
from dotenv import load_dotenv
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

async def main():
    # Initialize LLM
    llm = HuggingFaceInferenceAPI(
        model_name=MODEL_NAME,
        token=hf_token,
        provider="auto",
    )

    # Define agents (add description!)
    math_agent = ReActAgent(
        llm=llm,
        name="MathAgent",
        description="Solves math problems and performs arithmetic calculations.",  # ✅ added
        system_prompt="You are a helpful assistant that solves math problems."
    )

    text_agent = ReActAgent(
        llm=llm,
        name="TextAgent",
        description="Summarizes and rewrites text for clarity.",  # ✅ added
        system_prompt="You are a helpful assistant that summarizes text."
    )

    # Create workflow
    workflow = AgentWorkflow(
        agents=[math_agent, text_agent],
        root_agent="TextAgent"  # Starting agent
    )

    # Run workflow
    result = await workflow.run("Summarize this text: 'LlamaIndex helps build LLM-powered apps.' Then solve 3 * 7")
    print("📝 Agent Workflow Response:\n", result)

if __name__ == "__main__":
    asyncio.run(main())
