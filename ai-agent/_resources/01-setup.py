# Databricks notebook source
# MAGIC %md
# MAGIC # Setup de la demo
# MAGIC Este notebook se ejecuta con `%run` desde los notebooks principales. Se encarga de:
# MAGIC 1. Cargar la configuración (`config`) y las utilidades (`00-global-setup-v2`).
# MAGIC 2. Crear el esquema y el volumen.
# MAGIC 3. Descargar los datos de ejemplo o, si no hay acceso a internet (Free Edition), **generarlos localmente**.
# MAGIC 4. Crear las tablas `customers`, `subscriptions` y `billing`.
# MAGIC 5. Definir funciones auxiliares para el agente, la evaluación y Vector Search.

# COMMAND ----------

# Verificamos la versión de Python: la demo requiere Python 3.11 o superior
import sys
major, minor = sys.version_info[:2]
assert (major, minor) >= (3, 11), f"Esta demo requiere Python 3.11 o superior, pero se encontró {major}.{minor}. \nEn serverless, abre el panel 'Environment' a la derecha del notebook, selecciona la versión de entorno 2 o superior y aplica los cambios."

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

# MAGIC %run ./00-global-setup-v2

# COMMAND ----------

# Creamos catálogo/esquema/volumen (el widget "reset_all_data" permite borrar todo y empezar de nuevo)
reset_all_data = dbutils.widgets.get("reset_all_data") == "true"
DBDemos.setup_schema(catalog, db, reset_all_data, volume_name)
volume_folder = f"/Volumes/{catalog}/{db}/{volume_name}"

# COMMAND ----------

# MAGIC %md
# MAGIC Para acelerar la demo intentamos **descargar** los datos ya generados desde el repositorio de dbdemos.
# MAGIC
# MAGIC En Free Edition el acceso a internet está restringido a un conjunto de dominios de confianza, así que la
# MAGIC descarga puede fallar. En ese caso se **generan** los datos con los notebooks de la carpeta `_resources`:
# MAGIC - `02-data-generation`: clientes, suscripciones y facturación (con Faker).
# MAGIC - `03-doc-pdf-documentation`: manuales PDF generados con un LLM del workspace.

# COMMAND ----------

# Carpetas que necesita la demo dentro del volumen
tabular_folders = ["customers", "subscriptions", "billing"]

def try_download(folder):
    """Intenta descargar una carpeta del repositorio dbdemos-dataset. Devuelve True si quedó con archivos."""
    if not DBDemos.is_folder_empty(f"{volume_folder}/{folder}"):
        return True  # ya existen datos: no hay que descargar nada
    try:
        DBDemos.download_file_from_git(f"{volume_folder}/{folder}", "databricks-demos", "dbdemos-dataset", f"/llm/ai-agent/{folder}")
    except Exception as e:
        print(f"No se pudo descargar '{folder}': {str(e)[:300]}")
    return not DBDemos.is_folder_empty(f"{volume_folder}/{folder}")

# 1) Datos tabulares: si falta alguna carpeta, generamos las tres con Faker
if not all([try_download(f) for f in tabular_folders]):
    print("Generando los datos tabulares localmente (puede tardar 1-2 minutos)...")
    # Ruta relativa al notebook principal (todos los notebooks principales están un nivel por debajo de la raíz)
    dbutils.notebook.run("../_resources/02-data-generation", 1800)

# 2) Documentación PDF: si no se pudo descargar, la generamos con el LLM del workspace
if not try_download("pdf_documentation"):
    print("Generando la documentación PDF con el LLM del workspace (puede tardar varios minutos)...")
    dbutils.notebook.run("../_resources/03-doc-pdf-documentation", 3600)

print("Datos disponibles en", volume_folder)

# COMMAND ----------

# Creamos las tablas Delta a partir de los archivos parquet del volumen (solo si no existen)
if not spark.catalog.tableExists("customers") or \
    not spark.catalog.tableExists("subscriptions") or \
    not spark.catalog.tableExists("billing"):
        spark.read.parquet(f"{volume_folder}/customers").write.mode('overwrite').saveAsTable('customers')
        spark.read.parquet(f"{volume_folder}/subscriptions").write.mode('overwrite').saveAsTable('subscriptions')
        spark.read.parquet(f"{volume_folder}/billing").write.mode('overwrite').saveAsTable('billing')

# COMMAND ----------

import pandas as pd
import importlib.metadata

def log_customer_support_agent_model(resources, request_example):
    """Registra el agente (agent.py + agent_config.yaml) como modelo de MLflow ("models from code")."""
    import mlflow
    import mlflow.models
    agent_eval_dir = os.path.abspath(os.path.join(os.getcwd(), "../02-agent-eval"))
    model_config = mlflow.models.ModelConfig(development_config=f"{agent_eval_dir}/agent_config.yaml")
    with mlflow.start_run(run_name=model_config.get('config_version_name')):
        return mlflow.pyfunc.log_model(
            name="agent",
            python_model=f"{agent_eval_dir}/agent.py",
            model_config=f"{agent_eval_dir}/agent_config.yaml",
            input_example={"input": [{"role": "user", "content": request_example}]},
            resources=resources,  # Recursos de Databricks (endpoints, funciones, índices VS...) para la autenticación automática al desplegar
            extra_pip_requirements=["databricks-connect", f"mcp=={importlib.metadata.version('mcp')}"]  # misma versión de mcp que el entorno actual
        )

def predict_wrapper(question):
    """Envía una pregunta al modelo cargado (`loaded_model`) y devuelve solo el texto final de la respuesta."""
    # Formato de entrada estilo chat (Responses API)
    model_input = pd.DataFrame({
        "input": [[{"role": "user", "content": question}]]
    })
    response = loaded_model.predict(model_input)
    # El último elemento de la salida contiene el mensaje final del asistente
    return response['output'][-1]['content'][-1]['text']


def get_scorers(judge_model=None):
    """Devuelve la lista de jueces (scorers) de MLflow usados para evaluar el agente.

    judge_model: None para usar los jueces gestionados por Databricks, o un modelo como "databricks:/<endpoint>".
    """
    from mlflow.genai.scorers import RetrievalGroundedness, RelevanceToQuery, Safety, Guidelines
    kwargs = {"model": judge_model} if judge_model else {}
    return [
        RetrievalGroundedness(**kwargs),  # ¿La respuesta se basa en los datos recuperados por las herramientas?
        RelevanceToQuery(**kwargs),       # ¿La respuesta atiende la pregunta del usuario?
        Safety(**kwargs),                 # ¿Contiene contenido dañino o inapropiado?
        Guidelines(                       # Regla personalizada: no mostrar razonamiento ni herramientas
            guidelines="""La respuesta debe darse sin mostrar el razonamiento.
            - no mencionar que necesita buscar o consultar información
            - no mencionar las herramientas o funciones utilizadas
            - no describir los pasos intermedios ni el razonamiento""",
            name="steps_and_reasoning",
            **kwargs
        )
    ]


# COMMAND ----------

import time

def endpoint_exists(vsc, vs_endpoint_name):
  """Devuelve True si el endpoint de Vector Search existe."""
  try:
    return vs_endpoint_name in [e['name'] for e in vsc.list_endpoints().get('endpoints', [])]
  except Exception as e:
    # Solución temporal para el error REQUEST_LIMIT_EXCEEDED
    if "REQUEST_LIMIT_EXCEEDED" in str(e):
      print("ADVERTENCIA: no se pudo obtener el estado del endpoint (REQUEST_LIMIT_EXCEEDED). Se asumirá que existe")
      return True
    else:
      raise e

def wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name):
  """Espera (hasta ~30 min) a que el endpoint de Vector Search esté ONLINE."""
  for i in range(180):
    try:
      endpoint = vsc.get_endpoint(vs_endpoint_name)
    except Exception as e:
      if "REQUEST_LIMIT_EXCEEDED" in str(e):
        print("ADVERTENCIA: no se pudo obtener el estado del endpoint (REQUEST_LIMIT_EXCEEDED). Revisa su estado manualmente")
        return
      else:
        raise e
    status = endpoint.get("endpoint_status", endpoint.get("status"))["state"].upper()
    if "ONLINE" in status:
      return endpoint
    elif "PROVISIONING" in status or i < 6:
      if i % 20 == 0:
        print(f"Esperando a que el endpoint esté listo, puede tardar unos minutos... {endpoint}")
      time.sleep(10)
    else:
      raise Exception(f'''Error con el endpoint {vs_endpoint_name} - esto no debería pasar: {endpoint}.\n Bórralo y vuelve a ejecutar la celda anterior: vsc.delete_endpoint("{vs_endpoint_name}")''')
  raise Exception(f"Tiempo de espera agotado, el endpoint aún no está listo: {vsc.get_endpoint(vs_endpoint_name)}")

# COMMAND ----------

def index_exists(vsc, endpoint_name, index_full_name):
    """Devuelve True si el índice de Vector Search existe."""
    try:
        vsc.get_index(endpoint_name, index_full_name).describe()
        return True
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e) and 'does not exist' not in str(e) and type(e).__name__ != 'NotFound':
            print(f'Error inesperado al describir el índice. Puede ser un problema de permisos.')
            raise e
    return False

def wait_for_index_to_be_ready(vsc, vs_endpoint_name, index_name):
  """Espera a que el índice termine de sincronizar y esté ONLINE."""
  for i in range(180):
    idx = vsc.get_index(vs_endpoint_name, index_name).describe()
    index_status = idx.get('status', idx.get('index_status', {}))
    status = index_status.get('detailed_state', index_status.get('status', 'UNKNOWN')).upper()
    url = index_status.get('index_url', index_status.get('url', 'UNKNOWN'))
    if "ONLINE" in status:
      return
    if "UNKNOWN" in status:
      print(f"No se pudo obtener el estado - se asume que el índice está listo {idx} - url: {url}")
      return
    elif "PROVISIONING" in status:
      if i % 40 == 0: print(f"Esperando a que el índice esté listo, puede tardar unos minutos... {index_status} - url del pipeline: {url}")
      time.sleep(10)
    else:
        raise Exception(f'''Error con el índice - esto no debería pasar. Puede que el pipeline se haya detenido.\n Bórralo y vuelve a ejecutar la celda anterior: vsc.delete_index("{index_name}, {vs_endpoint_name}") \nDetalles del índice: {idx}''')
  raise Exception(f"Tiempo de espera agotado, el índice aún no está listo: {vsc.get_index(vs_endpoint_name, index_name)}")

# COMMAND ----------

import mlflow
from mlflow.exceptions import RestException

# Guardamos la función original de mlflow.set_experiment
_original_set_experiment = mlflow.set_experiment

def patched_set_experiment(experiment_name_or_path):
    """Versión tolerante de mlflow.set_experiment.

    Si no se puede crear el experimento en la ruta pedida (por ejemplo, dentro de un repo Git o porque
    ya existe una carpeta con ese nombre), se usa un experimento alternativo en /Shared.
    """
    try:
        _original_set_experiment(experiment_name_or_path)
    except RestException as e:
        if ("experiment creation is not permitted in a repo" in str(e)
                or "already exists" in str(e)
                or "MLFLOW_EXPERIMENT" in str(e)):
            from databricks.sdk import WorkspaceClient
            fallback_path = "/Shared/dbdemos/ai-agent"
            fallback_path_xp = fallback_path + "/ai_agent_experiment"
            w = WorkspaceClient()
            w.workspace.mkdirs(fallback_path)
            print(f"mlflow.set_experiment('{experiment_name_or_path}') falló (ruta de repo o conflicto de nombres).")
            print(f"Se usará el experimento alternativo: {fallback_path_xp} - ver el parche en _resources/01-setup")

            # Creamos el experimento alternativo si no existe
            try:
                mlflow.get_experiment_by_name(fallback_path_xp) or mlflow.create_experiment(fallback_path_xp)
            except Exception as create_err:
                print(f"❌ No se pudo crear el experimento alternativo: {create_err}")
                raise create_err

            # Usamos el EXPERIMENTO (no la carpeta) para evitar de nuevo el conflicto de nombres
            _original_set_experiment(fallback_path_xp)
        else:
            raise

# Aplicamos el parche
mlflow.set_experiment = patched_set_experiment
