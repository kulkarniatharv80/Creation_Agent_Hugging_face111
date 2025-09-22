# tools.py
from langchain_community.retrievers import BM25Retriever
from retriever import load_guest_docs
from smolagents.tools import BaseTool

# ---------------- Guest Info Tool ----------------
class GuestInfoRetrieverTool(BaseTool):
    name = "guest_info_retriever"
    description = "Retrieves detailed information about gala guests based on their name or relation."
    inputs = {"query": {"type": "string", "description": "Guest name or relation"}}
    output_type = "string"

    def __init__(self, docs):
        self.retriever = BM25Retriever.from_documents(docs)

    def forward(self, query: str):
        results = self.retriever.get_relevant_documents(query)
        if results:
            return "\n\n".join([doc.page_content for doc in results[:3]])
        return "No matching guest information found."

    def __call__(self, **kwargs):
        query = kwargs.get("query", "")
        return self.forward(query)

    def to_code_prompt(self):
        return f"{self.name}: {self.description}\nInputs: {self.inputs}\nOutputs: {self.output_type}"


# Initialize the tool with documents
guest_info_tool = GuestInfoRetrieverTool(load_guest_docs())

# ---------------- DuckDuckGo Search Tool ----------------
class DuckDuckGoSearchTool(BaseTool):
    name = "duckduckgo_search"
    description = "Performs a web search using DuckDuckGo and returns top results."
    inputs = {"query": {"type": "string", "description": "Search query"}}
    output_type = "string"

    def forward(self, query: str):
        # For now, mock results; replace with actual DuckDuckGo API if needed
        return f"Top DuckDuckGo search results for '{query}' [mock data]"

    def __call__(self, **kwargs):
        query = kwargs.get("query", "")
        return self.forward(query)

    def to_code_prompt(self):
        return f"{self.name}: {self.description}\nInputs: {self.inputs}\nOutputs: {self.output_type}"


# ---------------- Weather Info Tool ----------------
class WeatherInfoTool(BaseTool):
    name = "weather_info"
    description = "Provides weather information for a given location."
    inputs = {"location": {"type": "string", "description": "Location name"}}
    output_type = "string"

    def forward(self, location: str):
        # Mock data; replace with real weather API if needed
        return f"The weather in {location} is sunny, 25°C [mock data]"

    def __call__(self, **kwargs):
        location = kwargs.get("location", "")
        return self.forward(location)

    def to_code_prompt(self):
        return f"{self.name}: {self.description}\nInputs: {self.inputs}\nOutputs: {self.output_type}"


# ---------------- Hub Stats Tool ----------------
class HubStatsTool(BaseTool):
    name = "hub_stats"
    description = "Provides statistics for AI models from Hugging Face Hub."
    inputs = {"model_name": {"type": "string", "description": "Name of the model"}}
    output_type = "string"

    def forward(self, model_name: str):
        # Mock data; replace with HF Hub API if needed
        return f"The most popular model is {model_name}/ExampleModel with 3,313,345 downloads [mock data]"

    def __call__(self, **kwargs):
        model_name = kwargs.get("model_name", "")
        return self.forward(model_name)

    def to_code_prompt(self):
        return f"{self.name}: {self.description}\nInputs: {self.inputs}\nOutputs: {self.output_type}"
