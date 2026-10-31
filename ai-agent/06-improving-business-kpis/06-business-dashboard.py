# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Mide el impacto de tu agente de IA con Databricks Lakeflow, AI/BI Dashboards y Genie
# MAGIC
# MAGIC Desplegar un nuevo proyecto de IA es genial, pero al final lo que quieres es medir cómo impacta tu negocio.
# MAGIC
# MAGIC Normalmente, un asistente de soporte al cliente debería:
# MAGIC
# MAGIC - Reducir el tiempo de resolución
# MAGIC - Aumentar la satisfacción del cliente
# MAGIC - Generar más resultados de negocio
# MAGIC - Permitir que el equipo de soporte haga más, con interacciones de mayor calidad...
# MAGIC
# MAGIC Databricks te permite seguir todas estas métricas: conecta tu CRM o sistema de tickets con Lakeflow Connect (o cualquier otra ingesta), prepara y transforma los datos si hace falta, despliega un dashboard orientado al negocio y haz cualquier pregunta adicional sobre tus datos con Databricks Genie.
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/ai-agent/customer-support-dashboard.png?raw=true" width="900px">

# COMMAND ----------

# MAGIC %md
# MAGIC ## ¿Quieres probar una configuración AI/BI de soporte al cliente?
# MAGIC
# MAGIC Ejecuta las siguientes celdas para instalar la demo de dbdemos de soporte al cliente.
# MAGIC
# MAGIC *Free Edition: la instalación usa el catálogo `workspace`. Algunos recursos de la demo (por ejemplo, pipelines o jobs)
# MAGIC pueden verse afectados por los límites de Free Edition (un pipeline activo por tipo, 5 tareas de job concurrentes).*

# COMMAND ----------

# DBTITLE 1,Instalación de dbdemos
# MAGIC %pip install dbdemos -U
# MAGIC %restart_python

# COMMAND ----------

import dbdemos

# Instala la demo de AI/BI de soporte al cliente en el catálogo "workspace" (Free Edition no tiene catálogo "main")
dbdemos.install('aibi-customer-support', catalog='workspace', schema='dbdemos_aibi_customer_support')
