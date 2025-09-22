import asyncio
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.workflow import Context

# Define tools
async def add(ctx: Context, a: int, b: int) -> int:
    cur_state = await ctx.store.get("state")
    cur_state["num_fn_calls"] += 1
    await ctx.store.set("state", cur_state)
    return a + b

async def multiply(ctx: Context, a: int, b: int) -> int:
    cur_state = await ctx.store.get("state")
    cur_state["num_fn_calls"] += 1
    await ctx.store.set("state", cur_state)
    return a * b

# Initialize LLM
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# Create agents
multiply_agent = ReActAgent(
    name="multiply_agent",
    description="Multiply two integers",
    system_prompt="A helpful assistant that multiplies numbers.",
    tools=[multiply],
    llm=llm,
)

addition_agent = ReActAgent(
    name="add_agent",
    description="Add two integers",
    system_prompt="A helpful assistant that adds numbers.",
    tools=[add],
    llm=llm,
)

# Create workflow
workflow = AgentWorkflow(
    agents=[multiply_agent, addition_agent],
    root_agent="multiply_agent",
    initial_state={"num_fn_calls": 0},
    state_prompt="Current state: {state}. User message: {msg}",
)

# Run workflow
async def main():
    ctx = Context(workflow)
    response = await workflow.run(user_msg="Can you add 5 and 3?", ctx=ctx)
    state = await ctx.store.get("state")
    print("Agent Workflow Response:", response)
    print("Number of function calls:", state["num_fn_calls"])

if __name__ == "__main__":
    asyncio.run(main())
