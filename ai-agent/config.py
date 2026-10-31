# Databricks notebook source
# MAGIC %md
# MAGIC ## Archivo de configuración
# MAGIC
# MAGIC Cambia aquí el catálogo y el esquema si quieres ejecutar la demo en otra ubicación.
# MAGIC
# MAGIC **Adaptado para Databricks Free Edition (solo cómputo serverless):**
# MAGIC - En Free Edition no existe el catálogo `main`; el catálogo por defecto se llama `workspace`.
# MAGIC - Free Edition permite **un solo** endpoint de Vector Search; si ya tienes uno, la demo lo reutiliza.
# MAGIC - No todos los modelos fundacionales están disponibles; por eso se elige automáticamente
# MAGIC   el primer modelo disponible de una lista de candidatos.

# COMMAND ----------

# Catálogo y esquema donde se crearán las tablas, funciones, volumen, índice y modelo.
# En Free Edition el catálogo por defecto es "workspace".
catalog = "workspace"
schema = dbName = db = "dbdemos_ai_agent"

# Volumen de Unity Catalog donde se guardan los datos crudos (parquet, PDFs, dataset de evaluación)
volume_name = "raw_data"

# Nombre del endpoint de Vector Search. Free Edition solo permite 1 endpoint:
# si ya existe otro con distinto nombre, el notebook 03 lo reutilizará automáticamente.
VECTOR_SEARCH_ENDPOINT_NAME = "dbdemos_vs_endpoint"

# Nombre del modelo registrado en Unity Catalog y del endpoint de serving del agente
MODEL_NAME = "dbdemos_ai_agent_demo"
ENDPOINT_NAME = f'{MODEL_NAME}_{catalog}_{db}'[:60]

# Si es False, se omite el despliegue del agente como endpoint de Model Serving
# (útil si alcanzas el límite de endpoints activos de Free Edition).
DEPLOY_AGENT_ENDPOINT = True

# COMMAND ----------

# Modelos candidatos, en orden de preferencia. El LLM DEBE soportar llamadas a herramientas (tool calling).
LLM_CANDIDATES = [
    "databricks-meta-llama-3-3-70b-instruct",
    "databricks-gpt-oss-120b",
    "databricks-claude-sonnet-4-5",
    "databricks-llama-4-maverick",
]

# Modelos de embeddings candidatos para el índice de Vector Search (la documentación PDF está en inglés)
EMBEDDING_CANDIDATES = [
    "databricks-gte-large-en",
    "databricks-bge-large-en",
]


def pick_available_endpoint(candidates):
    """Devuelve el primer endpoint de la lista que exista en el workspace.

    Si no se puede listar los endpoints (por permisos o error de red), devuelve el primero de la lista.
    """
    try:
        from databricks.sdk import WorkspaceClient
        # Lista todos los endpoints de serving visibles (incluye los modelos fundacionales pay-per-token)
        available = {e.name for e in WorkspaceClient().serving_endpoints.list()}
        for c in candidates:
            if c in available:
                return c
        print(f"ADVERTENCIA: ninguno de {candidates} está disponible en este workspace. Se usará {candidates[0]}.")
    except Exception as e:
        print(f"ADVERTENCIA: no se pudo listar los endpoints ({e}). Se usará {candidates[0]}.")
    return candidates[0]


# LLM usado por el agente y por las funciones de IA (ai_query) en la generación de datos
LLM_ENDPOINT_NAME = pick_available_endpoint(LLM_CANDIDATES)
# Modelo de embeddings usado por el índice de Vector Search
EMBEDDING_ENDPOINT_NAME = pick_available_endpoint(EMBEDDING_CANDIDATES)

# Modelo para los jueces (scorers) de MLflow. None = usar los jueces gestionados por Databricks.
# Si los jueces gestionados fallan en tu workspace, usa por ejemplo: f"databricks:/{LLM_ENDPOINT_NAME}"
JUDGE_MODEL = None

print(f"Catálogo/esquema: {catalog}.{db} | LLM: {LLM_ENDPOINT_NAME} | Embeddings: {EMBEDDING_ENDPOINT_NAME}")
