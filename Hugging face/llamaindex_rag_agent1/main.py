from smolagents import CodeAgent, InferenceClientModel
from tools import guest_info_tool, DuckDuckGoSearchTool, WeatherInfoTool, HubStatsTool

# Initialize the model
model = InferenceClientModel()

# Create Alfred
alfred = CodeAgent(
    tools=[guest_info_tool, WeatherInfoTool(), HubStatsTool(), DuckDuckGoSearchTool()],
    model=model,
    add_base_tools=True,
    planning_interval=3
)

# Example usage
query = "Tell me about 'Lady Ada Lovelace'"
response = alfred.run(query)
print(response)
