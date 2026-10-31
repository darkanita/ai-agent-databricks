# Databricks notebook source
# MAGIC %md
# MAGIC # Parte 2 - Evaluar y desplegar nuestro sistema de agentes
# MAGIC
# MAGIC Vamos a empaquetar nuestras herramientas con LangChain/LangGraph (en el archivo `agent.py`) y usar esa primera versión del agente para ejecutar la evaluación.
# MAGIC
# MAGIC ## Evaluación de agentes con MLflow 3
# MAGIC Ya tenemos un agente; ahora necesitamos medir su desempeño y poder compararlo con versiones anteriores.
# MAGIC
# MAGIC Databricks lo hace muy fácil con MLflow 3. Automáticamente puedes:
# MAGIC
# MAGIC - Trazar todas las entradas y salidas del agente
# MAGIC - Capturar el feedback de los usuarios finales
# MAGIC - Evaluar el agente con un dataset de evaluación propio o sintético
# MAGIC - Construir datasets etiquetados con los expertos del negocio
# MAGIC - Comparar cada evaluación con la anterior
# MAGIC - Desplegar y seguir las evaluaciones una vez en producción
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/ai-agent/mlflow-evaluate-0.png?raw=true?raw=true" width="800px">
# MAGIC
# MAGIC ### Nuestro agente se compone de:
# MAGIC
# MAGIC - [**agent.py**]($./agent.py): en este archivo usamos LangChain/LangGraph para preparar un agente listo para usar.
# MAGIC - **agent_config.yaml**: contiene la configuración del agente, incluidos el system prompt y el endpoint del LLM. Este notebook lo crea.
# MAGIC
# MAGIC ¡Empecemos y probemos el agente en este notebook!

# COMMAND ----------

# DBTITLE 1,Instalación de librerías
# MAGIC %pip install "mlflow>=3.10.1" langchain==0.3.27 langgraph==0.6.11 databricks-langchain databricks-agents "unitycatalog-langchain[databricks]" databricks-feature-engineering==0.14.0 databricks-mcp "mcp<2" "typing_extensions>=4.13"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/01-setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1/ Construir y registrar nuestro agente
# MAGIC
# MAGIC ### 1.1/ Definir la configuración del agente
# MAGIC Primero escribimos el archivo de configuración con las herramientas que usará el agente, un system prompt básico y el endpoint del LLM.

# COMMAND ----------

# DBTITLE 1,Actualizamos el archivo de configuración
import yaml
import mlflow

rag_chain_config = {
    "config_version_name": "first_config",                     # nombre de la versión (se usa como nombre del run)
    "input_example": [{"role": "user", "content": "Dame los pedidos de john21@example.net"}],
    "uc_tool_names": [f"{catalog}.{dbName}.*"],                # todas las funciones UC del esquema
    "system_prompt": "Tu trabajo es ayudar a los clientes. Llama a la herramienta para responder.",
    "llm_endpoint_name": LLM_ENDPOINT_NAME,                    # LLM elegido en config
    "max_history_messages": 20,                                # mensajes de historial que se envían al LLM
    "retriever_config": None,                                  # se agregará en el notebook 03 (RAG)
    "mcp_server_urls": []                                      # se agregará en el notebook 07 (MCP)
}
try:
    with open('agent_config.yaml', 'w') as f:
        yaml.dump(rag_chain_config, f)
except:
    print('Se omite la escritura (ejecución como job de build)')
model_config = mlflow.models.ModelConfig(development_config='agent_config.yaml')

# COMMAND ----------

# MAGIC %md
# MAGIC Creamos el AGENTE con LangGraph en el archivo `agent.py`. Puedes revisarlo para ver el código.
# MAGIC
# MAGIC Aquí lo mantendremos simple: lo importamos y le enviamos una solicitud para explorar su traza en la interfaz de MLflow Tracing:

# COMMAND ----------

# Al importar agent.py se crea el objeto AGENT con la configuración anterior
from agent import AGENT

# Formato de solicitud de la Responses API
request_example = "Dame la información de john21@example.net"
answer = AGENT.predict({"input":[{"role": "user", "content": request_example}]})
answer

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### 1.2/ Abrir MLflow Tracing
# MAGIC
# MAGIC <img src="https://i.imgur.com/tNYUHdC.gif" style="float: right" width="700px">
# MAGIC
# MAGIC Abre el experimento desde el menú derecho del notebook. En las trazas verás el mensaje que acabamos de enviar: `Dame la información de john21@example.net`.
# MAGIC
# MAGIC MLflow registra todas las entradas/salidas y la traza interna, para que podamos analizar las solicitudes y construir mejores datasets de evaluación con el tiempo.
# MAGIC
# MAGIC Además de trazar las solicitudes, MLflow permite capturar fácilmente el feedback de los usuarios para detectar qué respuestas fueron incorrectas y mejorar el agente.
# MAGIC
# MAGIC *¡Mostraremos cómo capturar feedback cuando despleguemos la aplicación!*

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3/ Registrar el `agent` como modelo de MLflow
# MAGIC
# MAGIC ¡Se ve bien! Registremos el agente en MLflow usando el archivo de Python [agent]($./agent) para evitar problemas de serialización. Ver [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).
# MAGIC
# MAGIC *También pasamos la lista de recursos de Databricks (funciones, endpoints, etc.) que necesita el agente. ¡Así se gestionan los permisos automáticamente al desplegarlo!*

# COMMAND ----------

import mlflow
import importlib.metadata
def log_customer_support_agent_model(resources, request_example):
    """Registra agent.py + agent_config.yaml como un modelo pyfunc de MLflow dentro de un run."""
    with mlflow.start_run(run_name=model_config.get('config_version_name')):
        return mlflow.pyfunc.log_model(
            name="agent",
            python_model="agent.py",              # el código del agente ("models from code")
            model_config="agent_config.yaml",     # su configuración
            input_example={"input": [{"role": "user", "content": request_example}]},
            resources=resources,  # Recursos de Databricks (endpoints, funciones, VS...) para la autenticación automática al desplegar
            extra_pip_requirements=["databricks-connect", f"mcp=={importlib.metadata.version('mcp')}"]  # misma versión de mcp que el entorno actual
        )
logged_agent_info = log_customer_support_agent_model(AGENT.get_resources(), request_example)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.4/ Cargar y probar el modelo
# MAGIC ¡El modelo quedó guardado en MLflow! Carguémoslo y probémoslo. Envolvemos la función de predicción para extraer más fácilmente la respuesta final y facilitar la evaluación:

# COMMAND ----------

import pandas as pd
# Cargamos el modelo registrado y creamos una función de predicción
loaded_model = mlflow.pyfunc.load_model(f"runs:/{logged_agent_info.run_id}/agent")
def predict_wrapper(question):
    # Formato estilo chat
    model_input = pd.DataFrame({
        "input": [[{"role": "user", "content": question}]]
    })
    response = loaded_model.predict(model_input)
    # Devolvemos solo el texto del último mensaje (la respuesta final)
    return response['output'][-1]['content'][-1]['text']

answer = predict_wrapper("Dame los pedidos de john21@example.net.")
print(answer)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2/ Evaluación
# MAGIC
# MAGIC ### 2.1/ Evaluar el agente con [Agent Evaluation](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/)
# MAGIC
# MAGIC Preparamos un dataset de evaluación listo para usar en esta demo. Sin embargo, existen varias estrategias:
# MAGIC
# MAGIC - crear tu propio dataset de evaluación (lo que haremos)
# MAGIC - usar trazas existentes de MLflow y agregarlas al dataset (desde la API o desde la interfaz del experimento)
# MAGIC - usar la generación de datasets sintéticos de Databricks (ver un ejemplo en el [notebook de RAG con PDF]($../03-knowledge-base-rag/03.1-pdf-rag-tool))
# MAGIC - crear sesiones de etiquetado para obtener la opinión de expertos, directamente en la interfaz de MLflow
# MAGIC
# MAGIC Nota: también puedes seleccionar llamadas existentes en las trazas y agregarlas al dataset desde la interfaz, o usar la API directamente:
# MAGIC
# MAGIC ```
# MAGIC traces = mlflow.search_traces(filter_string=f"attributes.timestamp_ms > {ten_minutes_ago} AND attributes.status = 'OK'", order_by=["attributes.timestamp_ms DESC"])
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC Generamos (o regeneramos) el dataset de evaluación en el volumen con el notebook `_resources/04-eval-dataset-generation`:

# COMMAND ----------

# MAGIC %run ../_resources/04-eval-dataset-generation

# COMMAND ----------

# Leemos el dataset de evaluación generado
eval_example = spark.read.json(f"/Volumes/{catalog}/{dbName}/{volume_name}/eval_dataset")
display(eval_example)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2/ Crear nuestro dataset de MLflow
# MAGIC Usamos la API para crear el dataset (también se puede hacer desde la interfaz del experimento).

# COMMAND ----------

import mlflow
import mlflow.genai.datasets

# El dataset de evaluación de MLflow se guarda como tabla en Unity Catalog
eval_dataset_table_name = f"{catalog}.{dbName}.ai_agent_mlflow_eval"

try:
  # Si ya existe, lo reutilizamos
  eval_dataset = mlflow.genai.datasets.get_dataset(eval_dataset_table_name)
except Exception as e:
  if 'does not exist' in str(e) or 'not found' in str(e).lower():
    # Si no existe, lo creamos y le agregamos los ejemplos
    eval_dataset = mlflow.genai.datasets.create_dataset(eval_dataset_table_name)
    eval_dataset.merge_records(eval_example)
    print("Registros agregados al dataset de evaluación.")
  else:
    raise e

# Vista previa del dataset
display(eval_dataset.to_df())

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 2.3/ Agregar pautas (guidelines) para seguir el comportamiento del agente
# MAGIC
# MAGIC <img src="https://i.imgur.com/M3kLBHF.gif" style="float:right" width="700px">
# MAGIC
# MAGIC MLflow 3 permite crear pautas personalizadas para evaluar el comportamiento del agente.
# MAGIC
# MAGIC Usaremos algunos jueces integrados y agregaremos un `Guidelines` personalizado sobre pasos y razonamiento: queremos que el LLM dé la respuesta sin mencionar las herramientas internas que usa.

# COMMAND ----------

from mlflow.genai.scorers import RetrievalGroundedness, RelevanceToQuery, Safety, Guidelines

def get_scorers(judge_model=None):
    # judge_model=None usa los jueces gestionados por Databricks (ver JUDGE_MODEL en config)
    kwargs = {"model": judge_model} if judge_model else {}
    return [
        RetrievalGroundedness(**kwargs),  # ¿La respuesta se basa en los datos recuperados?
        RelevanceToQuery(**kwargs),       # ¿La respuesta atiende la solicitud del usuario?
        Safety(**kwargs),                 # ¿Hay contenido dañino o inapropiado?
        Guidelines(
            guidelines="""
            La respuesta debe darse sin mostrar el razonamiento.
            - no mencionar que necesita buscar o consultar información
            - no mencionar las herramientas o funciones utilizadas
            - no describir los pasos intermedios ni el razonamiento
            """,
            name="steps_and_reasoning",
            **kwargs
        )
    ]

scorers = get_scorers(JUDGE_MODEL)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.4/ Ejecutar la evaluación con nuestras pautas
# MAGIC
# MAGIC ¡Listo! Evaluemos ahora el dataset con nuestras pautas:

# COMMAND ----------

# Cada pregunta del dataset se envía a predict_wrapper y los jueces califican la respuesta
with mlflow.start_run(run_name='eval_with_no_reasoning_instructions'):
    results = mlflow.genai.evaluate(data=eval_dataset, predict_fn=predict_wrapper, scorers=scorers)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3/ Mejorar las métricas con un mejor system prompt
# MAGIC
# MAGIC Como vemos en la evaluación, el agente da mucha información sobre sus herramientas internas y sus pasos.
# MAGIC Por ejemplo, podría decir cosas como:
# MAGIC
# MAGIC `Primero necesito encontrar el registro del cliente con su correo. Como aún no tengo el correo de Thomas Green, debo pedírselo.`
# MAGIC
# MAGIC Aunque es un buen razonamiento, ¡no lo queremos en la respuesta final!
# MAGIC
# MAGIC ### 3.1/ Registrar una nueva versión del modelo con un mejor system prompt
# MAGIC
# MAGIC Actualicemos el system prompt con mejores instrucciones para evitar este comportamiento y volvamos a evaluar para confirmar la mejora.

# COMMAND ----------

try:
    # Cargamos la configuración actual y cambiamos la versión y el system prompt
    config = yaml.safe_load(open("agent_config.yaml"))
    config["config_version_name"] = "better_prompt"
    config["system_prompt"] = (
        "Eres un asistente de una empresa de telecomunicaciones. Llama a la herramienta adecuada para ayudar al usuario con facturación, soporte o información de su cuenta. "
        "Los pedidos o compras de un cliente corresponden a sus suscripciones y facturación. "
        "NO menciones ninguna herramienta interna ni los pasos de tu razonamiento en la respuesta final. No digas 'según los registros' ni insinúes que estás buscando información. "
        "Responde siempre en español."
    )
    yaml.dump(config, open("agent_config.yaml", "w"))
except Exception as e:
    print(f"Se omite la actualización (ignorar en ejecución como job) - {e}")

# COMMAND ----------

# Volvemos a registrar el agente en MLflow para capturar el nuevo prompt
logged_agent_info = log_customer_support_agent_model(AGENT.get_resources(), request_example)

# COMMAND ----------

# Cargamos el nuevo modelo para usarlo en la evaluación a través de `predict_wrapper`
loaded_model = mlflow.pyfunc.load_model(f"runs:/{logged_agent_info.run_id}/agent")

with mlflow.start_run(run_name='eval_with_reasoning_instructions'):
    results = mlflow.genai.evaluate(data=eval_dataset, predict_fn=predict_wrapper, scorers=scorers)

# COMMAND ----------

# MAGIC %md
# MAGIC Abre tu experimento y revisa los resultados.
# MAGIC
# MAGIC Selecciona el run anterior y este, y compáralos. ¡Deberías ver mejoras!

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4/ ¡Desplegar el agente como endpoint!
# MAGIC
# MAGIC Todo se ve bien: la última versión tiene un buen puntaje de evaluación. Despleguémosla como endpoint en tiempo real para la aplicación de chat de los usuarios.
# MAGIC
# MAGIC ### 4.1/ Registrar la nueva versión del modelo en Unity Catalog
# MAGIC

# COMMAND ----------

from mlflow import MlflowClient
UC_MODEL_NAME = f"{catalog}.{dbName}.{MODEL_NAME}"

# Registramos el modelo en Unity Catalog (crea una nueva versión)
client = MlflowClient()
uc_registered_model_info = mlflow.register_model(model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME, tags={"model": "customer_support_agent"})
# El alias "model-to-deploy" apunta a la versión que queremos desplegar
client.set_registered_model_alias(name=UC_MODEL_NAME, alias="model-to-deploy", version=uc_registered_model_info.version)

# Enlace HTML al agente registrado
displayHTML(f'<a href="/explore/data/models/{catalog}/{dbName}/{MODEL_NAME}" target="_blank">Abrir Unity Catalog para ver el agente registrado</a>')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2/ Desplegar el agente
# MAGIC
# MAGIC Iniciemos ahora el endpoint del modelo. El despliegue puede tardar hasta ~15-20 minutos.
# MAGIC
# MAGIC *En Free Edition hay un límite de endpoints activos. Si el despliegue falla por ese motivo, elimina endpoints que no uses
# MAGIC o pon `DEPLOY_AGENT_ENDPOINT = False` en `config` para continuar sin endpoint (los notebooks 03, 05 y 07 funcionan igual;
# MAGIC el notebook 04 sí necesita el endpoint).*

# COMMAND ----------

from databricks import agents
from databricks.sdk import WorkspaceClient
from datetime import timedelta

if DEPLOY_AGENT_ENDPOINT:
    try:
        # Despliega el modelo en un endpoint de Model Serving (+ review app) solo si esta versión no está desplegada
        if len(agents.get_deployments(model_name=UC_MODEL_NAME, model_version=uc_registered_model_info.version)) == 0:
            agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, endpoint_name=ENDPOINT_NAME, tags={"project": "dbdemos"})

        # Esperamos a que el endpoint esté listo antes de pasar al siguiente notebook
        WorkspaceClient().serving_endpoints.wait_get_serving_endpoint_not_updating(name=ENDPOINT_NAME, timeout=timedelta(minutes=30))
        print(f"Endpoint {ENDPOINT_NAME} listo.")
    except Exception as e:
        print(f"⚠️ No se pudo desplegar el endpoint (posible límite de Free Edition): {str(e)[:500]}")
else:
    print("DEPLOY_AGENT_ENDPOINT = False: se omite el despliegue del endpoint.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Siguiente paso: agregar una herramienta para responder preguntas sobre la base de conocimiento (RAG + Vector Search sobre PDF)
# MAGIC
# MAGIC Nuestro modelo funciona bien, pero no puede responder preguntas específicas que los clientes tienen sobre sus servicios.
# MAGIC
# MAGIC Por ejemplo, si le preguntamos al agente cómo resolver un código de error de nuestro router WIFI, fallará porque no tiene información sobre ello.
# MAGIC
# MAGIC Abre [03-knowledge-base-rag/03.1-pdf-rag-tool]($../03-knowledge-base-rag/03.1-pdf-rag-tool)
