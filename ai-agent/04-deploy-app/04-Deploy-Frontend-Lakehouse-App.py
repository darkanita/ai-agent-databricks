# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # 4/ Desplegar el frontend con Databricks Apps
# MAGIC
# MAGIC
# MAGIC La review app de Agent Bricks sirve para recoger el feedback de los stakeholders durante el desarrollo.
# MAGIC
# MAGIC ¡Pero todavía necesitas desplegar tu propia aplicación para los usuarios finales!
# MAGIC
# MAGIC Usemos Databricks Apps para construir y desplegar un primer chatbot sencillo.
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-frontend-app.png?raw=true" width="1200px">
# MAGIC
# MAGIC
# MAGIC <div style="background-color: #d4e7ff; padding: 10px; border-radius: 15px;">
# MAGIC <strong>Nota:</strong> en este ejemplo la app usa el endpoint del agente. Sin embargo, si la app es el único caso de uso, también podrías empaquetar el agente de MLflow dentro de la aplicación y prescindir del endpoint.
# MAGIC </div>
# MAGIC
# MAGIC <div style="background-color: #fff4d4; padding: 10px; border-radius: 15px; margin-top: 10px">
# MAGIC <strong>Free Edition:</strong> se permiten hasta 3 Databricks Apps, y cada app se detiene automáticamente 24 horas después de iniciarse o desplegarse (puedes reiniciarla cuando quieras).
# MAGIC Este notebook necesita que el endpoint del agente se haya desplegado en el notebook 02 o 03.
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Instalación de librerías
# MAGIC %pip install "mlflow[databricks]>=3.10.1" "databricks-sdk>=0.59.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/01-setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuración de la aplicación
# MAGIC
# MAGIC Databricks Apps funciona con cualquier framework de Python. En la demo crearemos un archivo de configuración sencillo con el nombre del endpoint de serving y lo guardaremos como `chatbot_app/app.yaml`.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

# Verificamos que el endpoint del agente exista antes de crear la app
w = WorkspaceClient()
existing_endpoints = [e.name for e in w.serving_endpoints.list()]
assert ENDPOINT_NAME in existing_endpoints, (
    f"No se encontró el endpoint '{ENDPOINT_NAME}'. Despliega el agente en el notebook 02 o 03 "
    "(DEPLOY_AGENT_ENDPOINT = True en config) antes de ejecutar este notebook."
)
print(f"La Databricks App usará el siguiente endpoint de serving: {ENDPOINT_NAME}")

# COMMAND ----------

import yaml

# La app llamará al endpoint del agente que desplegamos.
# Como el catálogo y el esquema se pueden cambiar en config, escribimos el nombre correcto del endpoint en app.yaml
yaml_app_config = {"command": ["uvicorn", "main:app", "--workers", "1"],
                    "env": [{"name": "MODEL_SERVING_ENDPOINT", "value": ENDPOINT_NAME}]
                  }
try:
    with open('chatbot_app/app.yaml', 'w') as f:
        yaml.dump(yaml_app_config, f)
except Exception as e:
    print(f'Se omite la escritura (ejecución como job de build) - {e}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Capturar feedback con MLflow Tracing y la Feedback API
# MAGIC
# MAGIC Con MLflow 3 es muy fácil capturar el feedback (pulgar arriba/abajo) directamente desde la aplicación.
# MAGIC
# MAGIC ```
# MAGIC client = mlflow.deployments.get_deploy_client("databricks")
# MAGIC
# MAGIC input_message = [{"content": "test", "role": "user", "type": "message"}]
# MAGIC
# MAGIC response = client.predict(
# MAGIC   endpoint=ENDPOINT_NAME,
# MAGIC   inputs={'input': input_message, "databricks_options": {
# MAGIC       # Devuelve la traza para obtener el trace_id con el que se registra el feedback
# MAGIC       "return_trace": True
# MAGIC     }}
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC
# MAGIC Luego, basta con usar `mlflow-tracing` en el backend del chatbot para registrar el feedback del usuario asociado a la traza:
# MAGIC
# MAGIC
# MAGIC ```
# MAGIC mlflow.log_feedback(
# MAGIC                 trace_id=trace_id, # el trace id presente en la respuesta, normalmente tr-xxxxx
# MAGIC                 name='user_feedback',
# MAGIC                 value=True if like_data.liked else False,
# MAGIC                 rationale=None,
# MAGIC                 source=mlflow.entities.AssessmentSource(source_type='HUMAN', source_id='user')
# MAGIC             )
# MAGIC ```
# MAGIC
# MAGIC *Nota: también puedes gestionar tus propios IDs - ver la [documentación de feedback](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/collect-user-feedback/) para más detalles.*

# COMMAND ----------

# MAGIC %md
# MAGIC ## Desplegar la aplicación
# MAGIC
# MAGIC La aplicación (un chatbot con Gradio) está formada por 2 archivos en la carpeta `chatbot_app`:
# MAGIC - `main.py`, con el código Python
# MAGIC - `app.yaml`, con la configuración
# MAGIC
# MAGIC Solo tenemos que llamar a la API para crear una nueva app y desplegarla usando la ruta `chatbot_app`:

# COMMAND ----------

from databricks.sdk.service.apps import App, AppResource, AppResourceServingEndpoint, AppResourceServingEndpointServingEndpointPermission, AppDeployment

app_name = "dbdemos-ai-agent-app"

# COMMAND ----------

# MAGIC %md
# MAGIC Las Databricks Apps incluyen un Service Principal creado automáticamente. Antes de desplegar, le damos acceso al endpoint del modelo...

# COMMAND ----------

# Recurso de la app: permiso CAN_QUERY sobre el endpoint del agente para el service principal de la app
serving_endpoint = AppResourceServingEndpoint(name=ENDPOINT_NAME,
                                              permission=AppResourceServingEndpointServingEndpointPermission.CAN_QUERY
                                              )

rag_endpoint = AppResource(name="rag-endpoint", serving_endpoint=serving_endpoint)

rag_app = App(name=app_name,
              description="Tu asistente de Databricks",
              default_source_code_path=os.path.join(os.getcwd(), 'chatbot_app'),
              resources=[rag_endpoint])
try:
  # Creamos la app y esperamos a que su cómputo esté listo
  app_details = w.apps.create_and_wait(app=rag_app)
  print(app_details)
except Exception as e:
  if "already exists" in str(e):
    print("La app ya existe; verificamos que esté en ejecución antes de (re)desplegarla")
    # Una app existente puede estar detenida (por ejemplo, tras las 24 h de Free Edition). deploy_and_wait
    # requiere que la app esté en ejecución, así que la iniciamos si hace falta.
    from databricks.sdk.service.apps import ComputeState
    app = w.apps.get(name=app_name)
    state = app.compute_status.state if app.compute_status else None
    if state != ComputeState.ACTIVE:
      try:
        w.apps.start_and_wait(name=app_name)
        print("App iniciada.")
      except Exception as start_err:
        print(f"No se pudo iniciar la app (continuamos, el despliegue puede iniciarla): {start_err}")
  else:
    raise e

# COMMAND ----------

# MAGIC %md
# MAGIC Una vez creada la app, podemos (re)desplegar el código así:

# COMMAND ----------

import mlflow

# Usamos el experimento del notebook 02 (el mismo donde el endpoint escribe sus trazas)
xp_name = os.getcwd().rsplit("/", 1)[0] + "/02-agent-eval/02.1_agent_evaluation"
mlflow.set_experiment(xp_name)

# COMMAND ----------

# Desplegamos el código de la carpeta chatbot_app y esperamos a que termine
deployment = AppDeployment(source_code_path=os.path.join(os.getcwd(), 'chatbot_app'))

app_details = w.apps.deploy_and_wait(app_name=app_name, app_deployment=deployment)

# COMMAND ----------

# URL de acceso a la aplicación
w.apps.get(name=app_name).url

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## ¡Tu Databricks App está lista y desplegada!
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-gradio-app.png?raw=true" width="750px" style="float: right; margin-left:10px">
# MAGIC
# MAGIC Abre la interfaz y empieza a hacerle preguntas al chatbot.
# MAGIC
# MAGIC Como mejora, podríamos ampliar la interfaz para enviar el feedback y revisar las respuestas incorrectas para mejorarlas.
# MAGIC
# MAGIC ## Conclusión
# MAGIC
# MAGIC Vimos cómo Databricks ofrece una plataforma de punta a punta:
# MAGIC - Construir y desplegar un endpoint
# MAGIC - Soluciones integradas para revisar, analizar y mejorar el chatbot
# MAGIC - ¡Desplegar aplicaciones de IA generativa con Databricks Apps!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Siguiente paso: ¿listo para el siguiente nivel? Monitoreemos el desempeño del agente en producción
# MAGIC
# MAGIC Abre el notebook [05-production-monitoring/05.production-monitoring]($../05-production-monitoring/05.production-monitoring) para aprender a monitorear el endpoint y evaluarlo en producción.
