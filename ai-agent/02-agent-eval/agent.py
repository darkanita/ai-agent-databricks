# Agente de soporte al cliente construido con LangGraph y empaquetado como ResponsesAgent de MLflow.
# Este archivo se registra en MLflow con el enfoque "models from code": MLflow guarda el código
# fuente (no un objeto serializado), lo que evita problemas de serialización al desplegar.
import asyncio
import json
from typing import Annotated, Any, Generator, Optional, Sequence, TypedDict, Union
from uuid import uuid4

import mlflow
from databricks_langchain import (
    ChatDatabricks,
    UCFunctionToolkit,
    VectorSearchRetrieverTool,
    DatabricksFunctionClient,
    set_uc_function_client,
)
from databricks_mcp import DatabricksMCPClient
from databricks.sdk import WorkspaceClient
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode

from mlflow.pyfunc import ResponsesAgent
from mlflow.models import ModelConfig
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

# Activamos el autolog de LangChain: cada llamada queda trazada en MLflow (LLM, herramientas, etc.)
mlflow.langchain.autolog()

# Necesario para usar funciones (UDF) de Unity Catalog como herramientas
set_uc_function_client(DatabricksFunctionClient())


class AgentState(TypedDict):
    """Estado del grafo: historial de mensajes y entradas/salidas personalizadas."""
    messages: Annotated[Sequence[BaseMessage], add_messages]  # add_messages acumula los mensajes
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]


# Esquema genérico para las herramientas MCP (acepta cualquier argumento)
GENERIC_SCHEMA = {
    "title": "MCPToolArgs",
    "type": "object",
    "properties": {},
    "additionalProperties": True
}


class MCPToolWrapper(BaseTool):
    """Envuelve una herramienta de un servidor MCP de Databricks como BaseTool de LangChain."""

    def __init__(self, name: str, description: str, server_url: str, ws_client: WorkspaceClient):
        super().__init__(name=name, description=description, args_schema=GENERIC_SCHEMA)
        # Guardamos la información del servidor de forma interna (no como campo de Pydantic)
        self._tool_data = {
            "server_url": server_url,
            "ws_client": ws_client,
        }

    def _run(self, **kwargs) -> str:
        # Cada llamada abre un cliente MCP y ejecuta la herramienta con los argumentos del LLM
        client = DatabricksMCPClient(
            server_url=self._tool_data["server_url"],
            workspace_client=self._tool_data["ws_client"]
        )
        response = client.call_tool(self.name, kwargs)
        return "".join([c.text for c in response.content])


def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[ToolNode, Sequence[BaseTool]],
    system_prompt: Optional[str] = None,
):
    """Crea un grafo LangGraph sencillo: LLM -> (herramientas -> LLM)* -> fin."""
    # Le indicamos al LLM qué herramientas puede llamar
    model = model.bind_tools(tools)

    def should_continue(state: AgentState):
        # Si el último mensaje del LLM pide herramientas, seguimos; si no, terminamos
        last = state["messages"][-1]
        return "continue" if isinstance(last, AIMessage) and last.tool_calls else "end"

    # Antepone el system prompt al historial de mensajes (si existe)
    pre = (
        RunnableLambda(lambda s: [{"role": "system", "content": system_prompt}] + s["messages"])
        if system_prompt
        else RunnableLambda(lambda s: s["messages"])
    )
    model_runnable = pre | model

    def call_model(state: AgentState, config: RunnableConfig):
        return {"messages": [model_runnable.invoke(state, config)]}

    # Definición del grafo: nodo "agent" (LLM) y nodo "tools" (ejecuta herramientas)
    graph = StateGraph(AgentState)
    graph.add_node("agent", RunnableLambda(call_model))
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
    graph.add_edge("tools", "agent")  # tras ejecutar herramientas, volvemos al LLM

    return graph.compile()


class LangGraphResponsesAgent(ResponsesAgent):
    """Agente compatible con la interfaz ResponsesAgent de MLflow (formato OpenAI Responses API)."""

    def __init__(
        self,
        uc_tool_names: Sequence[str] = ("workspace.dbdemos_ai_agent.*",),
        llm_endpoint_name: str = "databricks-meta-llama-3-3-70b-instruct",
        system_prompt: Optional[str] = None,
        retriever_config: Optional[dict] = None,
        mcp_server_urls: Optional[Sequence[str]] = None,
        max_history_messages: int = 20,
    ):
        self.llm_endpoint_name = llm_endpoint_name
        self.system_prompt = system_prompt
        self.max_history_messages = max_history_messages

        # LLM servido por Databricks (Foundation Model API)
        self.llm = ChatDatabricks(endpoint=llm_endpoint_name)

        # Cargamos las funciones de Unity Catalog como herramientas ("catalogo.esquema.*" = todas las del esquema)
        self.tools: list[BaseTool] = UCFunctionToolkit(function_names=list(uc_tool_names)).tools

        # Agregamos el retriever de Vector Search si está configurado (notebook 03)
        if retriever_config:
            self.tools.append(
                VectorSearchRetrieverTool(
                    index_name=retriever_config.get("index_name"),
                    name=retriever_config.get("tool_name", "retriever"),
                    description=retriever_config.get("description", "Herramienta de búsqueda vectorial"),
                    num_results=retriever_config.get("num_results", 3),
                )
            )

        # Agregamos las herramientas de los servidores MCP configurados (notebook 07)
        if mcp_server_urls:
            ws_client = WorkspaceClient()
            for url in mcp_server_urls:
                try:
                    client = DatabricksMCPClient(server_url=url, workspace_client=ws_client)
                    tool_defs = client.list_tools()
                    for t in tool_defs:
                        self.tools.append(MCPToolWrapper(t.name, t.description or t.name, url, ws_client))
                    print(f"Herramientas MCP cargadas desde {url}: {[t.name for t in self.tools if isinstance(t, MCPToolWrapper)]}")
                except Exception as e:
                    print(f"No se pudo cargar el servidor MCP {url}: {e}")

        # Creamos el grafo del agente
        self.agent = create_tool_calling_agent(self.llm, self.tools, system_prompt)

    # -----------------------
    # Conversión entre formatos: Responses API <-> mensajes de chat de LangChain
    # -----------------------
    def _responses_to_cc(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        """Convierte un mensaje en formato Responses API a formato chat-completions (cc)."""
        msg_type = message.get("type")
        if msg_type == "function_call":
            return [{
                "role": "assistant",
                "content": "tool_call",
                "tool_calls": [{
                    "id": message["call_id"],
                    "type": "function",
                    "function": {
                        "arguments": message["arguments"],
                        "name": message["name"],
                    },
                }],
            }]
        elif msg_type == "message" and isinstance(message["content"], list):
            return [{"role": message["role"], "content": content["text"]} for content in message["content"]]
        elif msg_type == "reasoning":
            return [{"role": "assistant", "content": json.dumps(message["summary"])}]
        elif msg_type == "function_call_output":
            return [{
                "role": "tool",
                "content": message["output"],
                "tool_call_id": message["call_id"],
            }]
        # Mensajes simples {"role": ..., "content": ...}: solo conservamos los campos conocidos
        filtered = {k: v for k, v in message.items() if k in {"role", "content", "name", "tool_calls", "tool_call_id"}}
        return [filtered] if filtered else []

    def _langchain_to_responses(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convierte los mensajes de LangChain (salida del grafo) en items de la Responses API."""
        for message in messages:
            message = message.model_dump()
            if message["type"] == "ai":
                # El LLM pidió llamar herramientas
                if tool_calls := message.get("tool_calls"):
                    return [
                        self.create_function_call_item(
                            id=message.get("id") or str(uuid4()),
                            call_id=tc["id"],
                            name=tc["name"],
                            arguments=json.dumps(tc["args"]),
                        )
                        for tc in tool_calls
                    ]
                # Respuesta de texto final del LLM
                mlflow.update_current_trace(response_preview=message["content"])
                return [self.create_text_output_item(
                    text=message["content"],
                    id=message.get("id") or str(uuid4())
                )]
            elif message["type"] == "tool":
                # Resultado de una herramienta
                return [self.create_function_call_output_item(
                    call_id=message["tool_call_id"],
                    output=message["content"]
                )]
        return []

    # -----------------------
    # Métodos de predicción
    # -----------------------
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Predicción no-streaming: recolecta todos los items finales del stream.

        No hace falta @mlflow.trace: MLflow traza automáticamente los métodos de ResponsesAgent."""
        outputs = [
            event.item for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Predicción en streaming: emite los items a medida que el grafo avanza."""
        cc_msgs = []
        mlflow.update_current_trace(request_preview=request.input[0].content)
        for msg in request.input:
            cc_msgs.extend(self._responses_to_cc(msg.model_dump()))

        # Limitamos el historial para no exceder el contexto del LLM
        if len(cc_msgs) > self.max_history_messages:
            cc_msgs = cc_msgs[-self.max_history_messages:]

        # stream_mode "updates" = salida de cada nodo; "messages" = tokens del LLM
        for event in self.agent.stream({"messages": cc_msgs}, stream_mode=["updates", "messages"]):
            if event[0] == "updates":
                for node_data in event[1].values():
                    for item in self._langchain_to_responses(node_data["messages"]):
                        yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)
            elif event[0] == "messages":
                try:
                    chunk = event[1][0]
                    if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                        yield ResponsesAgentStreamEvent(
                            **self.create_text_delta(delta=content, item_id=chunk.id),
                        )
                except Exception:
                    pass

    # -----------------------
    # Recursos usados por el agente
    # -----------------------
    def get_resources(self):
        """Lista los recursos de Databricks que usa el agente (LLM, funciones UC, índices VS).

        MLflow usa esta lista para dar permisos automáticamente al endpoint al desplegar.
        """
        res = [DatabricksServingEndpoint(endpoint_name=self.llm.endpoint)]
        for t in self.tools:
            if isinstance(t, VectorSearchRetrieverTool):
                res.extend(t.resources)
            elif hasattr(t, "uc_function_name"):
                res.append(DatabricksFunction(function_name=t.uc_function_name))
        return res

    # -----------------------
    # Utilidad para listar las herramientas cargadas
    # -----------------------
    def list_tools(self) -> list[str]:
        return [t.name for t in self.tools]

# ==========================
# Instanciamos el agente a partir de la configuración (agent_config.yaml)
# ==========================
# Al registrar el modelo, MLflow reemplaza esta configuración por la pasada en `model_config`.
model_config = ModelConfig(development_config="../02-agent-eval/agent_config.yaml")

AGENT = LangGraphResponsesAgent(
    uc_tool_names=model_config.get("uc_tool_names"),
    llm_endpoint_name=model_config.get("llm_endpoint_name"),
    system_prompt=model_config.get("system_prompt"),
    retriever_config=model_config.get("retriever_config"),
    mcp_server_urls=model_config.get("mcp_server_urls"),
    max_history_messages=model_config.get("max_history_messages"),
)

# Indicamos a MLflow cuál es el objeto modelo de este archivo
mlflow.models.set_model(AGENT)
