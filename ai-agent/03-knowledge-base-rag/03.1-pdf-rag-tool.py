# Databricks notebook source
# MAGIC %md
# MAGIC # Construir una base de conocimiento a partir de documentación PDF con Databricks Vector Search (RAG)
# MAGIC
# MAGIC Como vimos, nuestro agente no responde bien preguntas técnicas específicas, como los códigos de error del router WIFI.
# MAGIC
# MAGIC Eso se debe a que no tiene conocimiento sobre nuestros sistemas y productos internos.
# MAGIC
# MAGIC Por suerte, toda esa información está disponible en PDF, guardados en nuestro volumen.
# MAGIC
# MAGIC Los vamos a parsear, guardar en Vector Search y luego agregaremos un *retriever* al agente para mejorar sus capacidades.
# MAGIC
# MAGIC <div style="background-color: #fff4d4; padding: 10px; border-radius: 15px;">
# MAGIC <strong>Free Edition:</strong> solo se permite <strong>un</strong> endpoint de Vector Search. Si ya tienes uno, este notebook lo reutiliza.
# MAGIC Los manuales PDF están en inglés (el modelo de embeddings <code>gte-large-en</code> es para inglés); el agente igualmente responde en español.
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Instalación de librerías
# MAGIC %pip install "mlflow>=3.10.1" langchain==0.3.27 langgraph==0.6.11 databricks-langchain databricks-agents "unitycatalog-langchain[databricks]" databricks-feature-engineering==0.14.0 databricks-mcp "mcp<2" "typing_extensions>=4.13" pypdf
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/01-setup

# COMMAND ----------

# Carpeta del volumen donde están los PDF y número máximo de PDF a procesar (para limitar el costo)
pdf_folder = f"/Volumes/{catalog}/{dbName}/{volume_name}/pdf_documentation/"
MAX_PDFS = 8

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 1. Extraer la información de los PDF
# MAGIC Databricks ofrece la función integrada `ai_parse_document`, que usa IA para analizar los PDF y extraer su información como texto. ¡Así es muy fácil ingerir información no estructurada!

# COMMAND ----------

# Listamos algunos PDF del volumen con READ_FILES (formato binaryFile = el archivo completo en binario)
display(spark.sql(f"SELECT path FROM READ_FILES('{pdf_folder}', format => 'binaryFile') LIMIT 2"))

# COMMAND ----------

# DBTITLE 1,Probemos la función ai_parse_document
# ai_parse_document está disponible en serverless (y en DBR 17.1+)
try:
    display(spark.sql(f"""
        SELECT ai_parse_document(content) AS parsed_document
        FROM READ_FILES('{pdf_folder}', format => 'binaryFile') LIMIT 2"""))
except Exception as e:
    print(f"ai_parse_document no está disponible en este workspace: {str(e)[:300]}\nMás abajo se usará pypdf como alternativa.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1/ Crear la tabla de la base de conocimiento
# MAGIC
# MAGIC Primero creamos la tabla. Activamos el Change Data Feed para poder crear el índice de Vector Search sobre ella.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- id autoincremental (clave primaria del índice) y Change Data Feed activado (requerido por el índice Delta Sync)
# MAGIC CREATE TABLE IF NOT EXISTS knowledge_base (
# MAGIC   id BIGINT GENERATED ALWAYS AS IDENTITY,
# MAGIC   product_name STRING,
# MAGIC   title STRING,
# MAGIC   content STRING,
# MAGIC   doc_uri STRING)
# MAGIC   TBLPROPERTIES (delta.enableChangeDataFeed = true);

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 1.2/ PDF a texto con ai_parse_document
# MAGIC
# MAGIC Usamos la función integrada `ai_parse_document` para parsear automáticamente los PDF y `ai_extract` para extraer el nombre del producto y el título.
# MAGIC
# MAGIC *Nota: en este caso los PDF son relativamente pequeños, así que unimos todas las páginas de cada documento en un único campo de texto. Documentos más grandes necesitarían pasos de preprocesamiento (chunking) para reducir el tamaño del contexto y poder recuperar más documentos, por ejemplo asegurando que el modelo del router WIFI aparezca en todos los fragmentos para que la búsqueda vectorial sea más relevante.*
# MAGIC
# MAGIC *Ordenamos los PDF para procesar primero los usados en los ejemplos (router ADSL-R500, encuestas, códigos de error...) y limitamos la cantidad para controlar el costo.*

# COMMAND ----------

# Orden de prioridad: primero los manuales usados en los ejemplos de la demo
priority_order = """CASE
    WHEN path ILIKE '%adsl_r500%' THEN 0
    WHEN path ILIKE '%err_ref%' THEN 1
    WHEN path ILIKE '%sat_survey%' THEN 2
    WHEN path ILIKE '%firmware%' THEN 3
    WHEN path ILIKE '%return_policy%' THEN 4
    ELSE 9 END"""

try:
    spark.sql(f"""
    INSERT OVERWRITE TABLE knowledge_base (product_name, title, content, doc_uri)
    SELECT ai_extract.product_name, ai_extract.title, content, doc_uri
    FROM (
      SELECT
        ai_extract(content, array('product_name', 'title')) AS ai_extract,  -- extrae nombre de producto y título
        content,
        doc_uri
      FROM (
        -- Unimos el contenido de todos los elementos del documento parseado en un solo texto
        SELECT array_join(
                transform(parsed_document:document.elements::ARRAY<STRUCT<content:STRING>>, x -> x.content), '\\n') AS content,
               path AS doc_uri
        FROM (
          SELECT ai_parse_document(content) AS parsed_document, path
          FROM (
            SELECT * FROM READ_FILES('{pdf_folder}', format => 'binaryFile')
            ORDER BY {priority_order}, path
            LIMIT {MAX_PDFS} -- LÍMITE PARA CONTROLAR EL COSTO DE LA DEMO - QUÍTALO EN UN CASO REAL
          )
        )
      )
      WHERE content IS NOT NULL AND length(content) > 0  -- descartamos documentos que no se pudieron parsear
    )""")
    print("Base de conocimiento creada con ai_parse_document + ai_extract")
except Exception as e:
    # Alternativa si las funciones de IA no están disponibles: extraemos el texto con pypdf
    print(f"ai_parse_document/ai_extract no disponibles ({str(e)[:200]}). Usando pypdf como alternativa...")
    import os
    import pandas as pd
    from pypdf import PdfReader

    priority = ["adsl_r500", "err_ref", "sat_survey", "firmware", "return_policy"]
    def rank(p):
        return next((i for i, k in enumerate(priority) if k in p.lower()), 9), p

    paths = sorted([f.path.replace("dbfs:", "") for f in dbutils.fs.ls(pdf_folder) if f.path.lower().endswith(".pdf")], key=rank)[:MAX_PDFS]
    rows = []
    for p in paths:
        text = "\n".join((page.extract_text() or "") for page in PdfReader(p).pages)
        first_line = next((l.strip() for l in text.splitlines() if l.strip()), os.path.basename(p))
        product = os.path.basename(p).replace("_manual.pdf", "").replace("_", "-").upper()  # p. ej. ADSL-R500
        rows.append((product, first_line[:200], text, "dbfs:" + p))
    spark.createDataFrame(pd.DataFrame(rows, columns=["product_name", "title", "content", "doc_uri"])) \
         .createOrReplaceTempView("kb_fallback")
    spark.sql("INSERT OVERWRITE TABLE knowledge_base (product_name, title, content, doc_uri) SELECT * FROM kb_fallback")
    print(f"Base de conocimiento creada con pypdf ({len(rows)} documentos)")

# COMMAND ----------

# DBTITLE 1,Revisamos los resultados
# MAGIC %sql
# MAGIC SELECT * FROM knowledge_base;

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## 2/ Crear nuestra tabla de búsqueda vectorial
# MAGIC
# MAGIC ### 2.1/ Endpoints de Vector Search
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-basic-prep-2.png?raw=true" style="float: right; margin-left: 10px" width="400px">
# MAGIC
# MAGIC Los endpoints de Vector Search son las entidades donde viven los índices. Piensa en ellos como el punto de entrada que atiende las búsquedas.
# MAGIC
# MAGIC Creemos nuestro endpoint de Vector Search. Una vez creado, puedes verlo en la sección *Compute > Vector Search* del workspace. Haz clic en el nombre del endpoint para ver todos los índices que atiende.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient(disable_notice=True)

# Free Edition permite un solo endpoint: si ya existe otro, lo reutilizamos en lugar de crear uno nuevo
existing_endpoints = [e['name'] for e in vsc.list_endpoints().get('endpoints', [])]
if VECTOR_SEARCH_ENDPOINT_NAME not in existing_endpoints and len(existing_endpoints) > 0:
    print(f"Ya existe el endpoint '{existing_endpoints[0]}': se reutilizará (Free Edition solo permite uno).")
    VECTOR_SEARCH_ENDPOINT_NAME = existing_endpoints[0]

if not endpoint_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME):
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

# Esperamos a que el endpoint esté ONLINE (la primera vez puede tardar varios minutos)
wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
print(f"El endpoint {VECTOR_SEARCH_ENDPOINT_NAME} está listo.")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-basic-prep-3.png?raw=true" style="float: right; margin-left: 10px" width="400px">
# MAGIC
# MAGIC
# MAGIC ### 2.2/ Crear el índice de Vector Search
# MAGIC
# MAGIC Con el endpoint creado, solo tenemos que pedirle a Databricks que cree el índice sobre la tabla existente.
# MAGIC
# MAGIC Basta con indicar la columna de texto y el modelo fundacional de embeddings (`GTE`). Databricks construye y sincroniza el índice automáticamente.
# MAGIC
# MAGIC Databricks ofrece 3 tipos de búsqueda vectorial:
# MAGIC
# MAGIC * **Embeddings gestionados**: Databricks calcula los embeddings a partir de un campo de texto y sincroniza la tabla Delta con el índice (lo que usaremos)
# MAGIC * **Embeddings propios**: tú calculas los embeddings y los guardas en la tabla Delta; Databricks sincroniza la tabla con el índice
# MAGIC * **Acceso directo**: tú gestionas la indexación (sin tabla Delta) — *no disponible en Free Edition*
# MAGIC
# MAGIC Se puede hacer con la API o con unos pocos clics desde el Catalog Explorer:
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/index_creation.gif?raw=true" width="600px">
# MAGIC

# COMMAND ----------

# DBTITLE 1,Crear (o sincronizar) el índice
from databricks.sdk import WorkspaceClient

# Tabla que queremos indexar
source_table_fullname = f"{catalog}.{dbName}.knowledge_base"
# Nombre del índice
vs_index_fullname = f"{catalog}.{dbName}.knowledge_base_vs_index"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
  print(f"Creando el índice {vs_index_fullname} en el endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
  vsc.create_delta_sync_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED",                          # se sincroniza bajo demanda (más económico)
    primary_key="id",
    embedding_source_column='content',                  # columna con el texto
    embedding_model_endpoint_name=EMBEDDING_ENDPOINT_NAME  # endpoint de embeddings (ver config)
  )
  # Esperamos a que el índice esté listo y todos los embeddings creados e indexados
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
else:
  # Si ya existe, lanzamos una sincronización para incluir los datos nuevos de la tabla
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
  vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).sync()

print(f"El índice {vs_index_fullname} sobre la tabla {source_table_fullname} está listo")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3/ Probar el índice: buscar contenido relevante
# MAGIC
# MAGIC Es todo lo que hay que hacer. Databricks capturará y sincronizará automáticamente las nuevas filas de la tabla con el índice.
# MAGIC
# MAGIC Según el tamaño del dataset y del modelo, la creación del índice puede tardar unos segundos en empezar a indexar los embeddings.
# MAGIC
# MAGIC Probemos a buscar contenido similar.
# MAGIC
# MAGIC *Nota: `similarity_search` también admite un parámetro `filters`. Es útil para agregar una capa de seguridad al sistema RAG: puedes filtrar contenido sensible según quién hace la llamada (por ejemplo, filtrar por departamento según el usuario).*

# COMMAND ----------

# DBTITLE 1,Búsqueda por similitud
# La documentación está en inglés, por eso la consulta de prueba se hace en inglés
question = "My wifi router gives me error 01, what should I do?"

results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
  query_text=question,
  columns=["id", "content"],
  num_results=1)
docs = results.get('result', {}).get('data_array', [])
docs

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3/ Actualizar el agente para agregar el retriever como nueva herramienta
# MAGIC
# MAGIC Con el índice listo, solo tenemos que agregarlo como retriever a nuestro agente.
# MAGIC
# MAGIC Reutilizamos `agent.py` y `agent_config.yaml`: basta con agregar la configuración del retriever y el agente lo sumará a sus herramientas.

# COMMAND ----------

import mlflow
import yaml, sys, os
import mlflow.models
# Agregamos la carpeta ../02-agent-eval al path para poder importar agent.py
agent_eval_path = os.path.abspath(os.path.join(os.getcwd(), "../02-agent-eval"))
sys.path.append(agent_eval_path)
# Usamos el mismo experimento que el notebook anterior para tener todas las trazas en un solo lugar
mlflow.set_experiment(agent_eval_path + "/02.1_agent_evaluation")
conf_path = os.path.join(agent_eval_path, 'agent_config.yaml')

try:
    config = yaml.safe_load(open(conf_path))
    config["config_version_name"] = "model_with_retriever"
    config["retriever_config"] = {
        "index_name": vs_index_fullname,
        "tool_name": "product_technical_docs_retriever",
        "num_results": 1,
        # La descripción le indica al LLM cuándo usar la herramienta. Como los documentos están en inglés,
        # le pedimos que escriba la consulta (query) en inglés para que la búsqueda vectorial funcione mejor.
        "description": "Recupera documentación interna sobre nuestros productos, infraestructura, routers y otros, incluyendo funcionalidades, uso y solución de problemas. Usa esta herramienta para cualquier pregunta sobre documentación de productos o problemas con productos. IMPORTANTE: escribe siempre la consulta (query) en inglés, porque la documentación está en inglés."
    }
    yaml.dump(config, open(conf_path, "w"))
except Exception as e:
    print(f"Se omite la actualización (ignorar en ejecución como job) - {e}")

model_config = mlflow.models.ModelConfig(development_config=conf_path)

# COMMAND ----------

from agent import AGENT

# Probamos el retriever para confirmar que el agente ya tiene acceso a la guía PDF del router WIFI
request_example = "¿Cómo reinicio mi router WIFI ADSL-R500?"
answer = AGENT.predict({"input":[{"role": "user", "content": request_example}]})
answer

# COMMAND ----------

# MAGIC %md
# MAGIC Ahora registramos el nuevo agente en MLflow con `mlflow.pyfunc.log_model()`, como en el notebook `02.1_agent_evaluation`.

# COMMAND ----------

# El agente captura los recursos que necesita; ahora incluye el índice de Vector Search
for r in AGENT.get_resources():
  print(f"Recurso: {type(r).__name__}:{r.name}")

# COMMAND ----------

with mlflow.start_run(run_name=model_config.get('config_version_name')):
  logged_agent_info = mlflow.pyfunc.log_model(
    name="agent",
    python_model=agent_eval_path + "/agent.py",
    model_config=conf_path,
    input_example={"input": [{"role": "user", "content": request_example}]},
    # Recursos (endpoints, funciones, VS...) para la autenticación automática al desplegar
    resources=AGENT.get_resources(),
    extra_pip_requirements=["databricks-connect", "mcp==1.29.0"]
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4/ Evaluar el agente con nuestra base de documentos
# MAGIC
# MAGIC ¡El nuevo modelo está disponible! Como siempre, el siguiente paso es evaluarlo para asegurarnos de que las respuestas mejoran.
# MAGIC
# MAGIC
# MAGIC ### 4.1/ Generar datos de evaluación sintéticos
# MAGIC
# MAGIC Nuestro dataset de evaluación no tiene ninguna pregunta sobre los PDF.
# MAGIC
# MAGIC Con Databricks es fácil iniciar el dataset de evaluación con datos sintéticos y luego mejorarlo con el tiempo.

# COMMAND ----------

import pandas as pd

docs = spark.table('knowledge_base')
# Describimos lo que hace nuestro agente
agent_description = """
El agente es un chatbot RAG que responde preguntas técnicas sobre productos como routers wifi, instalación de fibra e información de red, pero también sobre estrategias de retención de clientes o pautas de atención en redes sociales. El agente tiene acceso a un corpus de documentos y su tarea es responder las preguntas del usuario recuperando los documentos relevantes y sintetizando una respuesta útil y precisa.
"""

question_guidelines = """
# Perfiles de usuario
- Un cliente que pregunta cómo solucionar un problema del sistema, paso a paso
- Un agente interno que pregunta sobre políticas internas

# Preguntas de ejemplo
- ¿Cómo soluciono el código de error 1001: autorización de devolución inválida cuando un cliente no puede enviar su solicitud de devolución?
- Me aparece el código de error 1001 al intentar publicar una encuesta. ¿Qué lo puede causar y cómo lo soluciono?

# Pautas adicionales
- Las preguntas deben ser breves y sonar naturales
- Las preguntas deben estar escritas en español
"""

try:
    from databricks.agents.evals import generate_evals_df
    # Generamos el dataset de evaluación sintético
    evals = generate_evals_df(
        docs,
        # Número total de evaluaciones a generar. El método intenta cubrir todos los documentos;
        # si este número es menor que el número de documentos, algunos documentos no tendrán evaluaciones.
        num_evals=10,
        # Pautas en texto libre que guían la generación sintética
        agent_description=agent_description,
        question_guidelines=question_guidelines
    )
    # Adaptamos el formato de entrada al que usa nuestro predict_wrapper: {"question": ...}
    evals["inputs"] = evals["inputs"].apply(lambda x: {"question": x["messages"][0]["content"]})
except Exception as e:
    # Alternativa si la generación sintética no está disponible: unas preguntas escritas a mano
    print(f"generate_evals_df no disponible ({str(e)[:200]}). Se usan preguntas manuales.")
    evals = pd.DataFrame({"inputs": [{"question": q} for q in [
        "¿Cómo reinicio mi router ADSL-R500?",
        "Mi router ADSL-R500 muestra el error 01, ¿qué debo hacer?",
        "¿Cuáles son los pasos para actualizar el firmware del router?",
        "¿Cuál es la política de devolución de equipos?",
        "¿Cómo se administra una encuesta de satisfacción al cliente (SAT-SURVEY-2024)?",
    ]]})
display(evals)

# COMMAND ----------

# Agregamos el dataset sintético a nuestro dataset de evaluación de MLflow
eval_dataset_table_name = f"{catalog}.{dbName}.ai_agent_mlflow_eval"

eval_dataset = mlflow.genai.datasets.get_dataset(eval_dataset_table_name)
eval_dataset.merge_records(evals)
print("Registros agregados al dataset de evaluación.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2/ Ejecutar la evaluación
# MAGIC Igual que antes, evaluamos con el dataset de MLflow. Verificamos que el modelo se siga comportando bien con las preguntas de clientes y que ahora responda bien las preguntas de la base de conocimiento.

# COMMAND ----------

from mlflow.genai.scorers import RetrievalGroundedness, RelevanceToQuery, Safety, Guidelines
import pandas as pd

eval_dataset = mlflow.genai.datasets.get_dataset(f"{catalog}.{dbName}.ai_agent_mlflow_eval")

# Usamos los mismos jueces que antes (la función está definida en _resources/01-setup)
scorers = get_scorers(JUDGE_MODEL)

# Cargamos el modelo y creamos la función de predicción
loaded_model = mlflow.pyfunc.load_model(f"runs:/{logged_agent_info.run_id}/agent")
def predict_wrapper(question):
    # Formato estilo chat
    model_input = pd.DataFrame({
        "input": [[{"role": "user", "content": question}]]
    })
    response = loaded_model.predict(model_input)
    return response['output'][-1]['content'][-1]['text']

print("Ejecutando la evaluación...")
with mlflow.start_run(run_name='eval_with_retriever'):
    results = mlflow.genai.evaluate(data=eval_dataset, predict_fn=predict_wrapper, scorers=scorers)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3/ ¡Desplegar el modelo final!
# MAGIC
# MAGIC Todo listo. Registramos el modelo en UC y actualizamos el endpoint con la última versión.

# COMMAND ----------

from mlflow import MlflowClient
UC_MODEL_NAME = f"{catalog}.{dbName}.{MODEL_NAME}"

# Registramos el modelo en Unity Catalog (nueva versión)
client = MlflowClient()
uc_registered_model_info = mlflow.register_model(model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME, tags={"model": "customer_support_agent", "model_version": "with_retriever"})

client.set_registered_model_alias(name=UC_MODEL_NAME, alias="model-to-deploy", version=uc_registered_model_info.version)
displayHTML(f'<a href="/explore/data/models/{catalog}/{dbName}/{MODEL_NAME}" target="_blank">Abrir Unity Catalog para ver el agente registrado</a>')

# COMMAND ----------

from databricks import agents
from databricks.sdk import WorkspaceClient
from datetime import timedelta

if DEPLOY_AGENT_ENDPOINT:
    try:
        # Al desplegar con el mismo nombre de modelo, la nueva versión se agrega al endpoint existente
        if len(agents.get_deployments(model_name=UC_MODEL_NAME, model_version=uc_registered_model_info.version)) == 0:
            agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, endpoint_name=ENDPOINT_NAME, tags={"project": "dbdemos"})

        # Esperamos a que el endpoint esté listo antes de pasar al siguiente notebook
        WorkspaceClient().serving_endpoints.wait_get_serving_endpoint_not_updating(name=ENDPOINT_NAME, timeout=timedelta(minutes=30))
        print(f"Endpoint {ENDPOINT_NAME} actualizado y listo.")
    except Exception as e:
        print(f"⚠️ No se pudo desplegar el endpoint (posible límite de Free Edition): {str(e)[:500]}")
else:
    print("DEPLOY_AGENT_ENDPOINT = False: se omite el despliegue del endpoint.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Siguiente paso: desplegar el chatbot en una Databricks App
# MAGIC
# MAGIC Con el agente listo, despleguemos una aplicación Gradio para servirlo a los usuarios finales.
# MAGIC
# MAGIC Abre [04-deploy-app/04-Deploy-Frontend-Lakehouse-App]($../04-deploy-app/04-Deploy-Frontend-Lakehouse-App)
