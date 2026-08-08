# Databricks notebook source
# MAGIC %md
# MAGIC # Mosaic AI Agent Framework: Author and deploy an OpenAI Responses API agent using a model hosted on Mosaic AI Foundation Model API
# MAGIC
# MAGIC This notebook shows how to author an OpenAI Responses agent and wrap it using the [`ResponsesAgent`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ResponsesAgent) interface to make it compatible with Mosaic AI. In this notebook you learn to:
# MAGIC
# MAGIC - Author an [Open AI Responses API](https://platform.openai.com/docs/api-reference/responses) agent (wrapped with `ResponsesAgent`) that calls an LLM hosted using Mosaic AI Foundation Models.
# MAGIC - Manually test the agent
# MAGIC - Evaluate the agent using Mosaic AI Agent Evaluation
# MAGIC - Log and deploy the agent
# MAGIC
# MAGIC To learn more about authoring an agent using Mosaic AI Agent Framework, see Databricks documentation ([AWS](https://docs.databricks.com/aws/generative-ai/agent-framework/author-agent) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/create-chat-model)).
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Address all `TODO`s in this notebook.

# COMMAND ----------

# MAGIC %pip install -U -qqqq backoff databricks-openai databricks-langchain faiss-cpu uv databricks-agents mlflow-skinny[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Define the agent in code
# MAGIC Define the agent code in a single cell below. This lets you easily write the agent code to a local Python file, using the `%%writefile` magic command, for subsequent logging and deployment.
# MAGIC
# MAGIC #### Agent tools
# MAGIC This agent code adds the built-in Unity Catalog function `system.ai.python_exec` to the agent. The agent code also includes commented-out sample code for adding a vector search index to perform unstructured data retrieval.
# MAGIC
# MAGIC For more examples of tools to add to your agent, see Databricks documentation ([AWS](https://docs.databricks.com/aws/generative-ai/agent-framework/agent-tool) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/agent-tool))

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC import json
# MAGIC import warnings
# MAGIC from typing import Any, Callable, Generator, Optional
# MAGIC from uuid import uuid4
# MAGIC
# MAGIC import backoff
# MAGIC import mlflow
# MAGIC import openai
# MAGIC import faiss
# MAGIC import pickle
# MAGIC import numpy as np
# MAGIC from databricks_openai import DatabricksOpenAI, UCFunctionToolkit, VectorSearchRetrieverTool
# MAGIC from databricks_langchain import DatabricksEmbeddings
# MAGIC from mlflow.entities import SpanType
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC     ResponsesAgentStreamEvent,
# MAGIC     output_to_responses_items_stream,
# MAGIC     to_chat_completions_input,
# MAGIC )
# MAGIC from openai import OpenAI
# MAGIC from pydantic import BaseModel
# MAGIC from unitycatalog.ai.core.base import get_uc_function_client
# MAGIC
# MAGIC ############################################
# MAGIC # Define your LLM endpoint and system prompt
# MAGIC ############################################
# MAGIC # TODO: Replace with your model serving endpoint
# MAGIC LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
# MAGIC
# MAGIC # TODO: Update with your system prompt
# MAGIC SYSTEM_PROMPT = """
# MAGIC You are an AI assistant with document search access.
# MAGIC
# MAGIC Workflow:
# MAGIC 1. Use search_documents to find relevant information
# MAGIC 2. Generate answers based ONLY on retrieved documents
# MAGIC 3. Search multiple times if needed
# MAGIC 4. If no relevant documents are found (tool returns threshold message), respond: "I don't have information about this topic in my document corpus."
# MAGIC 5. NEVER use your pre-trained knowledge - only use information from documents
# MAGIC """
# MAGIC
# MAGIC ############################################
# MAGIC # Global variables - will be initialized by the agent
# MAGIC ############################################
# MAGIC index = None
# MAGIC chunks_data = None
# MAGIC embedding_model = None
# MAGIC
# MAGIC ############################################
# MAGIC # Define the search_documents tool
# MAGIC ############################################
# MAGIC def search_documents(query: str, k: int = 3, threshold: float = 0.65) -> str:
# MAGIC     """
# MAGIC     Search for relevant document chunks based on a query.
# MAGIC     Use this tool when you need to find information from the document corpus to answer user questions.
# MAGIC     
# MAGIC     Args:
# MAGIC         query: The search query or question to find relevant documents for
# MAGIC         k: Number of top relevant chunks to retrieve (default: 3)
# MAGIC         threshold: Minimum similarity score to consider a result relevant (default: 0.65)
# MAGIC     
# MAGIC     Returns:
# MAGIC         A formatted string with relevant document chunks and their similarity scores.
# MAGIC         Returns a message if no relevant documents are found above the threshold.
# MAGIC     """
# MAGIC     # Embed and normalize query
# MAGIC     query_vector = np.array(embedding_model.embed_query(query), dtype='float32').reshape(1, -1)
# MAGIC     
# MAGIC     # Normalize for cosine similarity
# MAGIC     query_norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
# MAGIC     query_norm[query_norm == 0] = 1e-12
# MAGIC     normalized_query = query_vector / query_norm
# MAGIC     
# MAGIC     # Search FAISS index
# MAGIC     scores, indices = index.search(normalized_query, k)
# MAGIC     
# MAGIC     # Filter results by threshold
# MAGIC     results = []
# MAGIC     for i, idx in enumerate(indices[0]):
# MAGIC         score = float(scores[0][i])
# MAGIC         
# MAGIC         # Skip results below threshold
# MAGIC         if score < threshold:
# MAGIC             continue
# MAGIC             
# MAGIC         chunk_text = chunks_data[idx]['chunk']
# MAGIC         chunk_path = chunks_data[idx]['path'].split('/')[-1]
# MAGIC         results.append(f"\n[Source {i+1}] (Score: {score:.4f}) - {chunk_path}\n{chunk_text}")
# MAGIC     
# MAGIC     # Return appropriate message based on results
# MAGIC     if not results:
# MAGIC         return f"No relevant documents found above similarity threshold ({threshold}). The query may be outside the scope of the available documents."
# MAGIC     
# MAGIC     return "\n".join(results)
# MAGIC
# MAGIC # Create tool spec for OpenAI Responses API
# MAGIC search_documents_tool = {
# MAGIC     "type": "function",
# MAGIC     "function": {
# MAGIC         "name": "search_documents",
# MAGIC         "description": "Search for relevant document chunks based on a query. Use this tool when you need to find information from the document corpus to answer user questions.",
# MAGIC         "parameters": {
# MAGIC             "type": "object",
# MAGIC             "properties": {
# MAGIC                 "query": {
# MAGIC                     "type": "string",
# MAGIC                     "description": "The search query or question to find relevant documents for"
# MAGIC                 },
# MAGIC                 "k": {
# MAGIC                     "type": "integer",
# MAGIC                     "description": "Number of top relevant chunks to retrieve",
# MAGIC                     "default": 3
# MAGIC                 },
# MAGIC                 "threshold": {
# MAGIC                     "type": "number",
# MAGIC                     "description": "Minimum similarity score to consider a result relevant",
# MAGIC                     "default": 0.65
# MAGIC                 }
# MAGIC             },
# MAGIC             "required": ["query"]
# MAGIC         }
# MAGIC     }
# MAGIC }
# MAGIC
# MAGIC ###############################################################################
# MAGIC ## Define tools for your agent, enabling it to retrieve data or take actions
# MAGIC ## beyond text generation
# MAGIC ## To create and see usage examples of more tools, see
# MAGIC ## https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html
# MAGIC ###############################################################################
# MAGIC class ToolInfo(BaseModel):
# MAGIC     """
# MAGIC     Class representing a tool for the agent.
# MAGIC     - "name" (str): The name of the tool.
# MAGIC     - "spec" (dict): JSON description of the tool (matches OpenAI Responses format)
# MAGIC     - "exec_fn" (Callable): Function that implements the tool logic
# MAGIC     """
# MAGIC
# MAGIC     name: str
# MAGIC     spec: dict
# MAGIC     exec_fn: Callable
# MAGIC
# MAGIC
# MAGIC def create_tool_info(tool_spec, exec_fn_param: Optional[Callable] = None):
# MAGIC     """
# MAGIC     Factory function to create ToolInfo objects from a given tool spec
# MAGIC     and (optionally) a custom execution function.
# MAGIC     """
# MAGIC     # Remove 'strict' property, as Claude models do not support it in tool specs.
# MAGIC     tool_spec["function"].pop("strict", None)
# MAGIC     tool_name = tool_spec["function"]["name"]
# MAGIC     # Converts tool name with double underscores to UDF dot notation.
# MAGIC     udf_name = tool_name.replace("__", ".")
# MAGIC
# MAGIC     # Define a wrapper that accepts kwargs for the UC tool call,
# MAGIC     # then passes them to the UC tool execution client
# MAGIC     def exec_fn(**kwargs):
# MAGIC         function_result = uc_function_client.execute_function(udf_name, kwargs)
# MAGIC         # Return error message if execution fails, result value if not.
# MAGIC         if function_result.error is not None:
# MAGIC             return function_result.error
# MAGIC         else:
# MAGIC             return function_result.value
# MAGIC
# MAGIC     return ToolInfo(name=tool_name, spec=tool_spec, exec_fn=exec_fn_param or exec_fn)
# MAGIC
# MAGIC
# MAGIC # List to store information about all tools available to the agent.
# MAGIC TOOL_INFOS = [create_tool_info(search_documents_tool, search_documents)]
# MAGIC
# MAGIC # UDFs in Unity Catalog can be exposed as agent tools.
# MAGIC # The following code enables a python code interpreter tool using the system.ai.python_exec UDF.
# MAGIC
# MAGIC # TODO: Add additional tools
# MAGIC UC_TOOL_NAMES = ["system.ai.python_exec"]
# MAGIC
# MAGIC uc_function_client = get_uc_function_client()
# MAGIC uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
# MAGIC for tool_spec in uc_toolkit.tools:
# MAGIC     TOOL_INFOS.append(create_tool_info(tool_spec))
# MAGIC
# MAGIC
# MAGIC # Use Databricks vector search indexes as tools
# MAGIC # See https://docs.databricks.com/en/generative-ai/agent-framework/unstructured-retrieval-tools.html#locally-develop-vector-search-retriever-tools-with-ai-bridge
# MAGIC # List to store vector search tool instances for unstructured retrieval.
# MAGIC VECTOR_SEARCH_TOOLS = []
# MAGIC
# MAGIC # To add vector search retriever tools,
# MAGIC # use VectorSearchRetrieverTool and create_tool_info,
# MAGIC # then append the result to TOOL_INFOS.
# MAGIC # Example:
# MAGIC # VECTOR_SEARCH_TOOLS.append(
# MAGIC #     VectorSearchRetrieverTool(
# MAGIC #         index_name="",
# MAGIC #         # filters="..."
# MAGIC #     )
# MAGIC # )
# MAGIC
# MAGIC for vs_tool in VECTOR_SEARCH_TOOLS:
# MAGIC     TOOL_INFOS.append(create_tool_info(vs_tool.tool, vs_tool.execute))
# MAGIC
# MAGIC
# MAGIC class ToolCallingAgent(ResponsesAgent):
# MAGIC     """
# MAGIC     Class representing a tool-calling Agent.
# MAGIC     Handles both tool execution via exec_fn and LLM interactions via model serving.
# MAGIC     """
# MAGIC
# MAGIC     def __init__(self, llm_endpoint: str, tools: list[ToolInfo]):
# MAGIC         """Initializes the ToolCallingAgent with tools."""
# MAGIC         self.llm_endpoint = llm_endpoint
# MAGIC         self.model_serving_client: OpenAI = DatabricksOpenAI()
# MAGIC         self._tools_dict = {tool.name: tool for tool in tools}
# MAGIC         
# MAGIC         # Initialize FAISS index and embedding model
# MAGIC         self._initialize_search_resources()
# MAGIC
# MAGIC     def _initialize_search_resources(self):
# MAGIC         """Initialize FAISS index, chunks data, and embedding model."""
# MAGIC         global index, chunks_data, embedding_model
# MAGIC         
# MAGIC         if index is None:
# MAGIC             import os
# MAGIC             # Use Volume path directly
# MAGIC             index_path = "/Volumes/rag_on_databricks/landing/vol_landing/faiss_index/index.faiss"
# MAGIC             chunks_path = "/Volumes/rag_on_databricks/landing/vol_landing/faiss_index/chunks_data.pkl"
# MAGIC             
# MAGIC             # Load FAISS index
# MAGIC             index = faiss.read_index(index_path)
# MAGIC             
# MAGIC             # Load chunks metadata
# MAGIC             with open(chunks_path, "rb") as f:
# MAGIC                 chunks_data = pickle.load(f)
# MAGIC             
# MAGIC             # Load embedding model
# MAGIC             embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
# MAGIC     
# MAGIC     def get_tool_specs(self) -> list[dict]:
# MAGIC         """Returns tool specifications in the format OpenAI expects."""
# MAGIC         return [tool_info.spec for tool_info in self._tools_dict.values()]
# MAGIC
# MAGIC     @mlflow.trace(span_type=SpanType.TOOL)
# MAGIC     def execute_tool(self, tool_name: str, args: dict) -> Any:
# MAGIC         """Executes the specified tool with the given arguments."""
# MAGIC         return self._tools_dict[tool_name].exec_fn(**args)
# MAGIC
# MAGIC     @backoff.on_exception(backoff.expo, openai.RateLimitError)
# MAGIC     @mlflow.trace(span_type=SpanType.LLM)
# MAGIC     def call_llm(self, messages: list[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
# MAGIC         with warnings.catch_warnings():
# MAGIC             warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
# MAGIC             for chunk in self.model_serving_client.chat.completions.create(
# MAGIC                 model=self.llm_endpoint,
# MAGIC                 messages=to_chat_completions_input(messages),
# MAGIC                 tools=self.get_tool_specs(),
# MAGIC                 stream=True,
# MAGIC             ):
# MAGIC                 yield chunk.to_dict()
# MAGIC
# MAGIC     def handle_tool_call(
# MAGIC         self, tool_call: dict[str, Any], messages: list[dict[str, Any]]
# MAGIC     ) -> ResponsesAgentStreamEvent:
# MAGIC         """
# MAGIC         Execute tool calls, add them to the running message history, and return a ResponsesStreamEvent w/ tool output
# MAGIC         """
# MAGIC         args = json.loads(tool_call["arguments"])
# MAGIC         result = str(self.execute_tool(tool_name=tool_call["name"], args=args))
# MAGIC
# MAGIC         tool_call_output = self.create_function_call_output_item(tool_call["call_id"], result)
# MAGIC         messages.append(tool_call_output)
# MAGIC         return ResponsesAgentStreamEvent(type="response.output_item.done", item=tool_call_output)
# MAGIC
# MAGIC     def call_and_run_tools(
# MAGIC         self,
# MAGIC         messages: list[dict[str, Any]],
# MAGIC         max_iter: int = 10,
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         for _ in range(max_iter):
# MAGIC             last_msg = messages[-1]
# MAGIC             if last_msg.get("role", None) == "assistant":
# MAGIC                 return
# MAGIC             elif last_msg.get("type", None) == "function_call":
# MAGIC                 yield self.handle_tool_call(last_msg, messages)
# MAGIC             else:
# MAGIC                 yield from output_to_responses_items_stream(
# MAGIC                     chunks=self.call_llm(messages), aggregator=messages
# MAGIC                 )
# MAGIC
# MAGIC         yield ResponsesAgentStreamEvent(
# MAGIC             type="response.output_item.done",
# MAGIC             item=self.create_text_output_item("Max iterations reached. Stopping.", str(uuid4())),
# MAGIC         )
# MAGIC
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         session_id = None
# MAGIC         if request.custom_inputs and "session_id" in request.custom_inputs:
# MAGIC             session_id = request.custom_inputs.get("session_id")
# MAGIC         elif request.context and request.context.conversation_id:
# MAGIC             session_id = request.context.conversation_id
# MAGIC
# MAGIC         if session_id:
# MAGIC             mlflow.update_current_trace(
# MAGIC                 metadata={
# MAGIC                     "mlflow.trace.session": session_id,
# MAGIC                 }
# MAGIC             )
# MAGIC
# MAGIC         outputs = [
# MAGIC             event.item
# MAGIC             for event in self.predict_stream(request)
# MAGIC             if event.type == "response.output_item.done"
# MAGIC         ]
# MAGIC         return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self, request: ResponsesAgentRequest
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         session_id = None
# MAGIC         if request.custom_inputs and "session_id" in request.custom_inputs:
# MAGIC             session_id = request.custom_inputs.get("session_id")
# MAGIC         elif request.context and request.context.conversation_id:
# MAGIC             session_id = request.context.conversation_id
# MAGIC
# MAGIC         if session_id:
# MAGIC             mlflow.update_current_trace(
# MAGIC                 metadata={
# MAGIC                     "mlflow.trace.session": session_id,
# MAGIC                 }
# MAGIC             )
# MAGIC
# MAGIC         messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
# MAGIC             i.model_dump() for i in request.input
# MAGIC         ]
# MAGIC         yield from self.call_and_run_tools(messages=messages)
# MAGIC
# MAGIC
# MAGIC # Log the model using MLflow
# MAGIC mlflow.openai.autolog()
# MAGIC AGENT = ToolCallingAgent(llm_endpoint=LLM_ENDPOINT_NAME, tools=TOOL_INFOS)
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the agent
# MAGIC
# MAGIC Interact with the agent to test its output. Since we manually traced methods within `ResponsesAgent`, you can view the trace for each step the agent takes, with any LLM calls made via the OpenAI SDK automatically traced by autologging.
# MAGIC
# MAGIC Replace this placeholder input with an appropriate domain-specific example for your agent.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from agent import AGENT

result = AGENT.predict({"input": [{"role": "user", "content": "How does the Orion system prevent overheating?"}], "custom_inputs": {"session_id": "test-session"}})
print(result.model_dump(exclude_none=True))

# COMMAND ----------

for chunk in AGENT.predict_stream(
    {"input": [{"role": "user", "content": "What are the safety features of Orion A1?"}], "custom_inputs": {"session_id": "test-session-stream"}}
):
    print(chunk.model_dump(exclude_none=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the agent as an MLflow model
# MAGIC
# MAGIC Log the agent as code from the `agent.py` file. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).
# MAGIC
# MAGIC ### Enable automatic authentication for Databricks resources
# MAGIC For the most common Databricks resource types, Databricks supports and recommends declaring resource dependencies for the agent upfront during logging. This enables automatic authentication passthrough when you deploy the agent. With automatic authentication passthrough, Databricks automatically provisions, rotates, and manages short-lived credentials to securely access these resource dependencies from within the agent endpoint.
# MAGIC
# MAGIC To enable automatic authentication, specify the dependent Databricks resources when calling `mlflow.pyfunc.log_model().`
# MAGIC
# MAGIC   - **TODO**: If your Unity Catalog tool queries a [vector search index](docs link) or leverages [external functions](docs link), you need to include the dependent vector search index and UC connection objects, respectively, as resources. See docs ([AWS](https://docs.databricks.com/generative-ai/agent-framework/log-agent.html#specify-resources-for-automatic-authentication-passthrough) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/log-agent#resources)).

# COMMAND ----------

# Determine Databricks resources to specify for automatic auth passthrough at deployment time
from agent import UC_TOOL_NAMES, VECTOR_SEARCH_TOOLS
import mlflow
from mlflow.models.resources import DatabricksFunction
from pkg_resources import get_distribution

resources = []
for tool in VECTOR_SEARCH_TOOLS:
    resources.extend(tool.resources)
for tool_name in UC_TOOL_NAMES:
    resources.append(DatabricksFunction(function_name=tool_name))

# Path to FAISS index files
index_save_path = "/Volumes/rag_on_databricks/landing/vol_landing/faiss_index/"

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        pip_requirements=[
            "databricks-openai",
            "databricks-langchain",
            "backoff",
            "faiss-cpu",
            f"databricks-connect=={get_distribution('databricks-connect').version}",
        ],
        resources=resources,
        artifacts={
            "faiss_index": index_save_path
        },
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the agent with Agent Evaluation
# MAGIC
# MAGIC Use Mosaic AI Agent Evaluation to evalaute the agent's responses based on expected responses and other evaluation criteria. Use the evaluation criteria you specify to guide iterations, using MLflow to track the computed quality metrics.
# MAGIC See Databricks documentation ([AWS]((https://docs.databricks.com/aws/generative-ai/agent-evaluation) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/agent-evaluation/)).
# MAGIC
# MAGIC
# MAGIC To evaluate your tool calls, add custom metrics. See Databricks documentation ([AWS](https://docs.databricks.com/en/generative-ai/agent-evaluation/custom-metrics.html#evaluating-tool-calls) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-evaluation/custom-metrics#evaluating-tool-calls)).

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import RelevanceToQuery, RetrievalGroundedness, RetrievalRelevance, Safety

eval_dataset = [
    {
        "inputs": {"input": [{"role": "user", "content": "Calculate the 15th Fibonacci number"}]},
        "expected_response": "The 15th Fibonacci number is 610.",
    }
]

eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda input: AGENT.predict({"input": input, "custom_inputs": {"session_id": "evaluation-session"}}),
    scorers=[RelevanceToQuery(), Safety()],  # add more scorers here if they're applicable
)

# Review the evaluation results in the MLfLow UI (see console output)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-deployment agent validation
# MAGIC Before registering and deploying the agent, perform pre-deployment checks using the [mlflow.models.predict()](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.predict) API. See Databricks documentation ([AWS](https://docs.databricks.com/en/machine-learning/model-serving/model-serving-debug.html#validate-inputs) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/model-serving-debug#before-model-deployment-validation-checks)).

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"input": [{"role": "user", "content": "Hello!"}], "custom_inputs": {"session_id": "validation-session"}},
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Before you deploy the agent, you must register the agent to Unity Catalog.
# MAGIC
# MAGIC - **TODO** Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = "rag_on_databricks"
schema = "model_schema"
model_name = "rag_application"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents

agents.deploy(
    UC_MODEL_NAME,
    uc_registered_model_info.version,
    scale_to_zero=True,
    tags={"endpointSource": "docs"},
    deploy_feedback_model=False,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC After your agent is deployed, you can chat with it in AI playground to perform additional checks, share it with SMEs in your organization for feedback, or embed it in a production application. See docs ([AWS](https://docs.databricks.com/en/generative-ai/deploy-agent.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/deploy-agent)) for details