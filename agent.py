import json
import warnings
from typing import Any, Callable, Generator, Optional
from uuid import uuid4

import backoff
import mlflow
import openai
import faiss
import pickle
import numpy as np
from databricks_openai import DatabricksOpenAI, UCFunctionToolkit, VectorSearchRetrieverTool
from databricks_langchain import DatabricksEmbeddings
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from openai import OpenAI
from pydantic import BaseModel
from unitycatalog.ai.core.base import get_uc_function_client

############################################
# Define your LLM endpoint and system prompt
############################################
# TODO: Replace with your model serving endpoint
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"

# TODO: Update with your system prompt
SYSTEM_PROMPT = """
You are an AI assistant with document search access.

Workflow:
1. Use search_documents to find relevant information
2. Generate answers based ONLY on retrieved documents
3. Search multiple times if needed
4. If no relevant documents are found (tool returns threshold message), respond: "I don't have information about this topic in my document corpus."
5. NEVER use your pre-trained knowledge - only use information from documents
"""

############################################
# Global variables - will be initialized by the agent
############################################
index = None
chunks_data = None
embedding_model = None

############################################
# Define the search_documents tool
############################################
def search_documents(query: str, k: int = 3, threshold: float = 0.65) -> str:
    """
    Search for relevant document chunks based on a query.
    Use this tool when you need to find information from the document corpus to answer user questions.
    
    Args:
        query: The search query or question to find relevant documents for
        k: Number of top relevant chunks to retrieve (default: 3)
        threshold: Minimum similarity score to consider a result relevant (default: 0.65)
    
    Returns:
        A formatted string with relevant document chunks and their similarity scores.
        Returns a message if no relevant documents are found above the threshold.
    """
    # Embed and normalize query
    query_vector = np.array(embedding_model.embed_query(query), dtype='float32').reshape(1, -1)
    
    # Normalize for cosine similarity
    query_norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
    query_norm[query_norm == 0] = 1e-12
    normalized_query = query_vector / query_norm
    
    # Search FAISS index
    scores, indices = index.search(normalized_query, k)
    
    # Filter results by threshold
    results = []
    for i, idx in enumerate(indices[0]):
        score = float(scores[0][i])
        
        # Skip results below threshold
        if score < threshold:
            continue
            
        chunk_text = chunks_data[idx]['chunk']
        chunk_path = chunks_data[idx]['path'].split('/')[-1]
        results.append(f"\n[Source {i+1}] (Score: {score:.4f}) - {chunk_path}\n{chunk_text}")
    
    # Return appropriate message based on results
    if not results:
        return f"No relevant documents found above similarity threshold ({threshold}). The query may be outside the scope of the available documents."
    
    return "\n".join(results)

# Create tool spec for OpenAI Responses API
search_documents_tool = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Search for relevant document chunks based on a query. Use this tool when you need to find information from the document corpus to answer user questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query or question to find relevant documents for"
                },
                "k": {
                    "type": "integer",
                    "description": "Number of top relevant chunks to retrieve",
                    "default": 3
                },
                "threshold": {
                    "type": "number",
                    "description": "Minimum similarity score to consider a result relevant",
                    "default": 0.65
                }
            },
            "required": ["query"]
        }
    }
}

###############################################################################
## Define tools for your agent, enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html
###############################################################################
class ToolInfo(BaseModel):
    """
    Class representing a tool for the agent.
    - "name" (str): The name of the tool.
    - "spec" (dict): JSON description of the tool (matches OpenAI Responses format)
    - "exec_fn" (Callable): Function that implements the tool logic
    """

    name: str
    spec: dict
    exec_fn: Callable


def create_tool_info(tool_spec, exec_fn_param: Optional[Callable] = None):
    """
    Factory function to create ToolInfo objects from a given tool spec
    and (optionally) a custom execution function.
    """
    # Remove 'strict' property, as Claude models do not support it in tool specs.
    tool_spec["function"].pop("strict", None)
    tool_name = tool_spec["function"]["name"]
    # Converts tool name with double underscores to UDF dot notation.
    udf_name = tool_name.replace("__", ".")

    # Define a wrapper that accepts kwargs for the UC tool call,
    # then passes them to the UC tool execution client
    def exec_fn(**kwargs):
        function_result = uc_function_client.execute_function(udf_name, kwargs)
        # Return error message if execution fails, result value if not.
        if function_result.error is not None:
            return function_result.error
        else:
            return function_result.value

    return ToolInfo(name=tool_name, spec=tool_spec, exec_fn=exec_fn_param or exec_fn)


# List to store information about all tools available to the agent.
TOOL_INFOS = [create_tool_info(search_documents_tool, search_documents)]

# UDFs in Unity Catalog can be exposed as agent tools.
# The following code enables a python code interpreter tool using the system.ai.python_exec UDF.

# TODO: Add additional tools
UC_TOOL_NAMES = ["system.ai.python_exec"]

uc_function_client = get_uc_function_client()
uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
for tool_spec in uc_toolkit.tools:
    TOOL_INFOS.append(create_tool_info(tool_spec))


# Use Databricks vector search indexes as tools
# See https://docs.databricks.com/en/generative-ai/agent-framework/unstructured-retrieval-tools.html#locally-develop-vector-search-retriever-tools-with-ai-bridge
# List to store vector search tool instances for unstructured retrieval.
VECTOR_SEARCH_TOOLS = []

# To add vector search retriever tools,
# use VectorSearchRetrieverTool and create_tool_info,
# then append the result to TOOL_INFOS.
# Example:
# VECTOR_SEARCH_TOOLS.append(
#     VectorSearchRetrieverTool(
#         index_name="",
#         # filters="..."
#     )
# )

for vs_tool in VECTOR_SEARCH_TOOLS:
    TOOL_INFOS.append(create_tool_info(vs_tool.tool, vs_tool.execute))


class ToolCallingAgent(ResponsesAgent):
    """
    Class representing a tool-calling Agent.
    Handles both tool execution via exec_fn and LLM interactions via model serving.
    """

    def __init__(self, llm_endpoint: str, tools: list[ToolInfo]):
        """Initializes the ToolCallingAgent with tools."""
        self.llm_endpoint = llm_endpoint
        self.model_serving_client: OpenAI = DatabricksOpenAI()
        self._tools_dict = {tool.name: tool for tool in tools}
        
        # Initialize FAISS index and embedding model
        self._initialize_search_resources()

    def _initialize_search_resources(self):
        """Initialize FAISS index, chunks data, and embedding model."""
        global index, chunks_data, embedding_model
        
        if index is None:
            import os
            # Use Volume path directly
            index_path = "/Volumes/rag_on_databricks/landing/vol_landing/faiss_index/index.faiss"
            chunks_path = "/Volumes/rag_on_databricks/landing/vol_landing/faiss_index/chunks_data.pkl"
            
            # Load FAISS index
            index = faiss.read_index(index_path)
            
            # Load chunks metadata
            with open(chunks_path, "rb") as f:
                chunks_data = pickle.load(f)
            
            # Load embedding model
            embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
    
    def get_tool_specs(self) -> list[dict]:
        """Returns tool specifications in the format OpenAI expects."""
        return [tool_info.spec for tool_info in self._tools_dict.values()]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Executes the specified tool with the given arguments."""
        return self._tools_dict[tool_name].exec_fn(**args)

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    @mlflow.trace(span_type=SpanType.LLM)
    def call_llm(self, messages: list[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
            for chunk in self.model_serving_client.chat.completions.create(
                model=self.llm_endpoint,
                messages=to_chat_completions_input(messages),
                tools=self.get_tool_specs(),
                stream=True,
            ):
                yield chunk.to_dict()

    def handle_tool_call(
        self, tool_call: dict[str, Any], messages: list[dict[str, Any]]
    ) -> ResponsesAgentStreamEvent:
        """
        Execute tool calls, add them to the running message history, and return a ResponsesStreamEvent w/ tool output
        """
        args = json.loads(tool_call["arguments"])
        result = str(self.execute_tool(tool_name=tool_call["name"], args=args))

        tool_call_output = self.create_function_call_output_item(tool_call["call_id"], result)
        messages.append(tool_call_output)
        return ResponsesAgentStreamEvent(type="response.output_item.done", item=tool_call_output)

    def call_and_run_tools(
        self,
        messages: list[dict[str, Any]],
        max_iter: int = 10,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        for _ in range(max_iter):
            last_msg = messages[-1]
            if last_msg.get("role", None) == "assistant":
                return
            elif last_msg.get("type", None) == "function_call":
                yield self.handle_tool_call(last_msg, messages)
            else:
                yield from output_to_responses_items_stream(
                    chunks=self.call_llm(messages), aggregator=messages
                )

        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item("Max iterations reached. Stopping.", str(uuid4())),
        )

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        session_id = None
        if request.custom_inputs and "session_id" in request.custom_inputs:
            session_id = request.custom_inputs.get("session_id")
        elif request.context and request.context.conversation_id:
            session_id = request.context.conversation_id

        if session_id:
            mlflow.update_current_trace(
                metadata={
                    "mlflow.trace.session": session_id,
                }
            )

        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        session_id = None
        if request.custom_inputs and "session_id" in request.custom_inputs:
            session_id = request.custom_inputs.get("session_id")
        elif request.context and request.context.conversation_id:
            session_id = request.context.conversation_id

        if session_id:
            mlflow.update_current_trace(
                metadata={
                    "mlflow.trace.session": session_id,
                }
            )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
            i.model_dump() for i in request.input
        ]
        yield from self.call_and_run_tools(messages=messages)


# Log the model using MLflow
mlflow.openai.autolog()
AGENT = ToolCallingAgent(llm_endpoint=LLM_ENDPOINT_NAME, tools=TOOL_INFOS)
mlflow.models.set_model(AGENT)
