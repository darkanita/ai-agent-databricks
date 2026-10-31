# Databricks notebook source
# MAGIC %md
# MAGIC ## Exponer tus herramientas desde un servidor MCP
# MAGIC
# MAGIC Si quieres exponer tus herramientas o datos para que otros agentes puedan usarlos, puedes aprovechar los servidores MCP (Model Context Protocol).
# MAGIC
# MAGIC Databricks soporta 3 tipos de MCP:
# MAGIC
# MAGIC - **Servidores MCP gestionados por Databricks**:
# MAGIC servidores listos para usar que permiten a los agentes consultar datos y usar herramientas de Unity Catalog. Exponen servicios de Databricks como recursos MCP (Vector Search, funciones de UC, Genie Spaces, DBSQL...).
# MAGIC
# MAGIC - **Servidores MCP externos**:
# MAGIC conexión a servidores MCP de terceros alojados fuera de Databricks, usando proxies gestionados y conexiones de Unity Catalog.
# MAGIC
# MAGIC - **Servidores MCP personalizados**:
# MAGIC aloja de forma segura tu propio servidor MCP como Databricks App, ya sea propio o de terceros.
# MAGIC
# MAGIC
# MAGIC Consulta la [documentación de servidores MCP gestionados](https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp#available-managed-servers) para más detalles.
# MAGIC
# MAGIC
# MAGIC En este ejemplo mostramos cómo darle al agente las herramientas de un servidor MCP gestionado por Databricks, como las funciones de IA del sistema en UC (`system.ai`). ¡Después podrás configurar tu agente con cualquier otra herramienta MCP, ya sea de un servidor gestionado por Databricks o de uno personalizado!

# COMMAND ----------

# DBTITLE 1,Instalación de librerías
# MAGIC %pip install "mlflow>=3.10.1" langchain==0.3.27 langgraph==0.6.11 databricks-langchain databricks-agents "unitycatalog-langchain[databricks]" databricks-feature-engineering==0.14.0 databricks-mcp "mcp<2" "typing_extensions>=4.13" nest_asyncio
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/01-setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## MCP gestionado por Databricks
# MAGIC
# MAGIC Para usar las funciones de IA del sistema en UC desde el servidor MCP gestionado por Databricks, solo hay que agregar la URL correspondiente al archivo `agent_config.yaml` (en la carpeta `02-agent-eval`).
# MAGIC
# MAGIC - **Funciones de IA del sistema en Unity Catalog**: `https://{workspace-hostname}/api/2.0/mcp/functions/system/ai`

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import os, sys, yaml, mlflow
import nest_asyncio
# El cliente MCP usa asyncio; nest_asyncio permite ejecutarlo dentro del event loop del notebook
nest_asyncio.apply()

# --- Rutas ---
agent_eval_path = os.path.abspath(os.path.join(os.getcwd(), "../02-agent-eval"))
sys.path.append(agent_eval_path)  # para poder importar agent.py
conf_path = os.path.join(agent_eval_path, "agent_config.yaml")

# --- Detectamos la URL del workspace con el SDK de Databricks ---
ws = WorkspaceClient()
workspace_url = ws.config.host.rstrip("/")
mcp_url = f"{workspace_url}/api/2.0/mcp/functions/system/ai"

# ==========================
# Actualizamos la configuración para incluir el servidor MCP
# ==========================
try:
    config = yaml.safe_load(open(conf_path))
    config["config_version_name"] = "model_with_mcp"
    config["mcp_server_urls"] = [mcp_url]

    with open(conf_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

except Exception as e:
    print(f"Se omitió la actualización de MCP: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Instanciar el agente
# MAGIC
# MAGIC A continuación instanciamos el agente y, para confirmar que recibió las herramientas del servidor MCP gestionado, revisamos todas las herramientas disponibles. En la lista, además de las funciones de UC que creamos, veremos las herramientas del servidor MCP con las funciones de IA del sistema, es decir **system__ai__python_exec**.

# COMMAND ----------

# ==========================
# Instanciamos el agente
# ==========================
from agent import LangGraphResponsesAgent
import mlflow.models

model_config = mlflow.models.ModelConfig(development_config=conf_path)

AGENT = LangGraphResponsesAgent(
    uc_tool_names=model_config.get("uc_tool_names"),
    llm_endpoint_name=model_config.get("llm_endpoint_name"),
    system_prompt=model_config.get("system_prompt"),
    mcp_server_urls=model_config.get("mcp_server_urls"),
    max_history_messages=model_config.get("max_history_messages"),
)

print("✅ Herramientas disponibles:", AGENT.list_tools())

# COMMAND ----------

# Prueba rápida: una pregunta que se resuelve ejecutando código Python con la herramienta MCP system.ai.python_exec
answer = AGENT.predict({"input": [{"role": "user", "content": "Usa Python para calcular cuántos días faltan para el 31 de diciembre de este año."}]})
answer

# COMMAND ----------

# MAGIC %md
# MAGIC ## Probar el agente en el AI Playground

# COMMAND ----------

# MAGIC %md
# MAGIC Para probar el agente, solo tienes que elegir el endpoint de tu preferencia y agregar las herramientas necesarias. En nuestro caso agregaremos las herramientas del servidor MCP gestionado, seleccionando las funciones de IA del sistema en la caja de herramientas de funciones de UC. Del mismo modo podrías agregar cualquier otra herramienta, incluso de servidores MCP personalizados o externos.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/main/images/product/ai-agent/pg-mcp-img1.png" width="800px">

# COMMAND ----------

# MAGIC %md
# MAGIC Cuando empecemos a hacer preguntas, veremos que el agente cargó correctamente la herramienta del servidor MCP gestionado que agregamos arriba.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/main/images/product/ai-agent/pg-mcp-img2.png" width="800">

# COMMAND ----------

# MAGIC %md
# MAGIC Ahora podemos seguir explorando...

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/main/images/product/ai-agent/pg-mcp-img3.png" width="800">
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Siguientes pasos
# MAGIC Si quieres explorar otras opciones de servidores MCP en Databricks, consulta la [documentación de MCP en Databricks](https://docs.databricks.com/aws/en/generative-ai/mcp/).
