# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Construye y evalúa sistemas de agentes de IA con Databricks y MLflow
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/ai-agent/agent-demo-0.png?raw=true" width="800px" style="float: right">
# MAGIC
# MAGIC Databricks facilita crear y registrar tus propios agentes y herramientas para manipular tus datos y conectarte a cualquier sistema externo (desde el OpenAI Agent SDK y los agentes de Claude hasta servidores MCP).
# MAGIC
# MAGIC ¡Pero construir el agente es solo el primer paso para llevarlo a producción y generar impacto en el negocio!
# MAGIC
# MAGIC En esta demo construiremos un agente para el equipo de atención al cliente. Podrá procesar las solicitudes de soporte y dar orientación, **reduciendo el tiempo de resolución, aumentando la satisfacción y bajando los costos.**
# MAGIC
# MAGIC Necesitarás los siguientes pasos, y Databricks los hace más fáciles que nunca:
# MAGIC
# MAGIC <div style="background-color: #fff4d4; padding: 10px; border-radius: 15px;">
# MAGIC <strong>Versión adaptada a Databricks Free Edition (serverless):</strong> usa el catálogo <code>workspace</code>, genera los datos localmente si no hay acceso a internet,
# MAGIC reutiliza el único endpoint de Vector Search permitido y elige automáticamente un modelo fundacional disponible.
# MAGIC Revisa y ajusta los parámetros en el notebook <a href="$./config">config</a>. Agent Bricks no está disponible en Free Edition.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 1/ Crea tus herramientas
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/ai-agent/agent-demo-1.png?raw=true" width="500px" style="float: right">
# MAGIC
# MAGIC Empecemos creando nuestras primeras herramientas. Crearemos algunas funciones que ayudan al equipo de soporte a encontrar información de los clientes y las registraremos en tu catalog.schema.
# MAGIC
# MAGIC Recuerda: ¡puede ser cualquier función!

# COMMAND ----------

# MAGIC %md
# MAGIC Abre [01-create-tools/01_create_first_billing_agent]($./01-create-tools/01_create_first_billing_agent) para comenzar.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 2/ Despliega y evalúa tu agente
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/ai-agent/mlflow-evaluate-0.png?raw=true?raw=true" width="500px" style="float: right">
# MAGIC
# MAGIC Con las herramientas listas, crearemos un agente con LangChain/LangGraph y le daremos las herramientas que creamos.
# MAGIC
# MAGIC Una vez creado el agente, ejecutaremos una primera ronda de evaluación con un dataset de evaluación y mejoraremos el prompt para obtener mejores métricas.

# COMMAND ----------

# MAGIC %md
# MAGIC Abre [02-agent-eval/02.1_agent_evaluation]($./02-agent-eval/02.1_agent_evaluation) para desplegar y evaluar tu agente.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3/ Agrega una base de conocimiento
# MAGIC
# MAGIC Nuestro agente funciona bien, pero no puede responder preguntas sobre nuestros productos y procedimientos internos.
# MAGIC
# MAGIC Es momento de agregar una base de conocimiento: extraer la información de los PDF y usar Vector Search para enriquecer el contexto del prompt con información propia.
# MAGIC
# MAGIC Y, por supuesto, cerraremos el ciclo volviendo a evaluar el agente para asegurarnos de que sigue mejorando.
# MAGIC
# MAGIC Abre [03-knowledge-base-rag/03.1-pdf-rag-tool]($./03-knowledge-base-rag/03.1-pdf-rag-tool) para agregar tu base de conocimiento.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4/ Despliega tu agente en una Databricks App y captura el feedback de los usuarios en MLflow
# MAGIC
# MAGIC Con el agente listo, podemos desplegarlo con una Databricks App.
# MAGIC
# MAGIC En la demo publicaremos un chatbot sencillo con Gradio para mostrar cómo capturar el feedback de los usuarios en MLflow y usarlo para mejorar el dataset de evaluación y el desempeño del modelo con el tiempo.
# MAGIC
# MAGIC Abre [04-deploy-app/04-Deploy-Frontend-Lakehouse-App]($./04-deploy-app/04-Deploy-Frontend-Lakehouse-App) para comenzar.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5/ Agrega monitoreo en producción a tu agente
# MAGIC
# MAGIC ¡La aplicación ya está publicada! El siguiente paso es agregar un proceso de monitoreo que evalúe todas las solicitudes entrantes y verifique que se mantienen alineadas con nuestras métricas de evaluación.
# MAGIC
# MAGIC Abre [05-production-monitoring/05.production-monitoring]($./05-production-monitoring/05.production-monitoring) para agregar monitoreo en producción.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6/ Mide cómo tu agente mejora los indicadores del negocio
# MAGIC
# MAGIC El último paso es medir el impacto del agente en el negocio.
# MAGIC
# MAGIC Databricks te permite gestionar todo tu stack de datos. Por ejemplo, puedes sincronizar los datos de tu CRM o de tu sistema de tickets y seguir cómo sube el CSAT con el tiempo, comparando un equipo con agente y otro sin él.
# MAGIC
# MAGIC Abre [06-improving-business-kpis/06-business-dashboard]($./06-improving-business-kpis/06-business-dashboard) para seguir tus KPI de negocio con un dashboard listo para usar.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7/ (Opcional) Agrega herramientas desde servidores MCP
# MAGIC
# MAGIC Abre [07-adding-mcp/07-agent_mcp]($./07-adding-mcp/07-agent_mcp) para darle al agente herramientas de un servidor MCP gestionado por Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC ¡Felicitaciones!
# MAGIC Ya estás listo para construir y desplegar tu agente de IA con Databricks y MLflow 3.
# MAGIC
# MAGIC ## Siguiente paso: agentes listos para usar con Agent Bricks
# MAGIC
# MAGIC Muchas tareas de IA son comunes, y Databricks ofrece agentes listos para usar que puedes combinar *(no disponible en Free Edition)*:
# MAGIC
# MAGIC - **Information Extraction**: extrae información clave, clasifica contenido o resume texto de documentos en un JSON estructurado.
# MAGIC - **Custom LLM**: especializa un endpoint LLM para tareas de texto (generación de contenido, chat) alineadas con las pautas de tu dominio.
# MAGIC - **Multi-Agent Supervisor**: diseña un sistema de IA que integre tus agentes. Incluso puedes incluir Genie Spaces.
# MAGIC - **Knowledge Assistant**: convierte tu documentación en un chatbot experto.
# MAGIC - ...
# MAGIC
# MAGIC Abre [Agent Bricks](/ml/agents) y configura estos agentes en pocos clics.
