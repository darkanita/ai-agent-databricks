# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Construye tu primer sistema de agentes con Databricks
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/llm-tools-functions/ai-agent-functions.png?raw=true" style="float: right" width="700px">
# MAGIC Los LLM son muy buenos respondiendo preguntas de conocimiento general, pero no tienen información sobre tu propio negocio.
# MAGIC
# MAGIC Databricks facilita desarrollar herramientas propias que el LLM puede llamar, creando un agente completo con capacidad de razonamiento.
# MAGIC
# MAGIC
# MAGIC ### Construir herramientas sencillas en Unity Catalog
# MAGIC En este notebook registraremos en Unity Catalog:
# MAGIC - **Funciones SQL**: consultas que acceden a la información de clientes, pedidos y suscripciones.
# MAGIC - **Una función Python sencilla**: registraremos una función Python que hace operaciones matemáticas, para superar una limitación común de los modelos de lenguaje.

# COMMAND ----------

# MAGIC %md
# MAGIC Nota: ejecuta esta demo con cómputo **Serverless** (el único disponible en Databricks Free Edition). En el panel *Environment* usa la versión de entorno 2 o superior.

# COMMAND ----------

# DBTITLE 1,Instalación de librerías
# MAGIC %pip install "mlflow>=3.10.1" databricks-langchain databricks-agents "unitycatalog-langchain[databricks]" databricks-feature-engineering==0.14.0 "typing_extensions>=4.13"
# MAGIC # Reiniciamos Python para cargar los paquetes recién instalados
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Setup: configuración, esquema, volumen, datos y tablas (ver `_resources/01-setup`).
# MAGIC La primera ejecución puede tardar varios minutos si hay que generar los datos.

# COMMAND ----------

# MAGIC %run ../_resources/01-setup

# COMMAND ----------

# MAGIC %md
# MAGIC Los datos sintéticos que usarán nuestros agentes ya están listos en las tablas `customers`, `subscriptions` y `billing`.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 1. Obtener los datos de un cliente a partir de su correo
# MAGIC
# MAGIC Agreguemos una función que devuelva los datos de un cliente a partir de su correo electrónico.
# MAGIC
# MAGIC *Nota: para consultas puntuales de baja latencia como esta se podrían usar tablas Postgres en tiempo real (Lakebase). Para mantener la demo sencilla usaremos SQL. ¡Usar SQL también es una muy buena forma de darle capacidades analíticas a tu agente!*
# MAGIC

# COMMAND ----------

# DBTITLE 1,Crear una función registrada en Unity Catalog
# MAGIC %sql
# MAGIC -- Función SQL de tabla (UDTF) que devuelve el registro del cliente.
# MAGIC -- Los COMMENT son importantes: el LLM los usa como descripción de la herramienta y de sus parámetros.
# MAGIC CREATE OR REPLACE FUNCTION get_customer_by_email(email_input STRING COMMENT 'correo del cliente usado para obtener su información')
# MAGIC RETURNS TABLE (
# MAGIC     customer_id BIGINT,
# MAGIC     first_name STRING,
# MAGIC     last_name STRING,
# MAGIC     email STRING,
# MAGIC     phone STRING,
# MAGIC     address STRING,
# MAGIC     city STRING,
# MAGIC     state STRING,
# MAGIC     zip_code STRING,
# MAGIC     customer_segment STRING,
# MAGIC     registration_date DATE,
# MAGIC     customer_status STRING,
# MAGIC     loyalty_tier STRING,
# MAGIC     tenure_years DOUBLE,
# MAGIC     churn_risk_score BIGINT,
# MAGIC     customer_value_score BIGINT
# MAGIC )
# MAGIC COMMENT 'Devuelve el registro del cliente que coincide con el correo indicado. Incluye su ID, nombre, apellido y más.'
# MAGIC RETURN (
# MAGIC     SELECT * FROM customers
# MAGIC     WHERE email = email_input
# MAGIC     LIMIT 1
# MAGIC );
# MAGIC

# COMMAND ----------

# DBTITLE 1,Probamos la función
# MAGIC %sql SELECT * FROM get_customer_by_email('john21@example.net');

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 2. Obtener toda la información de facturación
# MAGIC Agreguemos una función que devuelva los pedidos y suscripciones de un cliente. Recibe el ID del cliente y devuelve agregados de toda la facturación pasada y las suscripciones actuales.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Une las suscripciones del cliente con un resumen de su facturación (total facturado, pagado, pagos atrasados...)
# MAGIC CREATE OR REPLACE FUNCTION get_customer_billing_and_subscriptions(customer_id_input BIGINT COMMENT 'ID del cliente usado para obtener pedidos, facturación y suscripciones')
# MAGIC RETURNS TABLE (
# MAGIC     customer_id BIGINT,
# MAGIC     subscription_id BIGINT,
# MAGIC     service_type STRING,
# MAGIC     plan_name STRING,
# MAGIC     plan_tier STRING,
# MAGIC     monthly_charge BIGINT,
# MAGIC     start_date DATE,
# MAGIC     contract_length_months BIGINT,
# MAGIC     status STRING,
# MAGIC     autopay_enabled BOOLEAN,
# MAGIC     total_billed DOUBLE,
# MAGIC     total_paid DOUBLE,
# MAGIC     total_late_payments BIGINT,
# MAGIC     total_late_fees DOUBLE,
# MAGIC     latest_payment_status STRING
# MAGIC )
# MAGIC COMMENT 'Devuelve los detalles de suscripciones y facturación de un cliente.'
# MAGIC RETURN (
# MAGIC     SELECT
# MAGIC         s.customer_id, s.subscription_id, s.service_type, s.plan_name, s.plan_tier,
# MAGIC         s.monthly_charge, s.start_date, s.contract_length_months, s.status, s.autopay_enabled,
# MAGIC         COALESCE(b.total_billed, 0), COALESCE(b.total_paid, 0),
# MAGIC         COALESCE(b.total_late_payments, 0), COALESCE(b.total_late_fees, 0),
# MAGIC         COALESCE(b.latest_payment_status, 'N/A')
# MAGIC     FROM subscriptions s
# MAGIC     LEFT JOIN (
# MAGIC         -- Agregados de facturación por suscripción
# MAGIC         SELECT
# MAGIC             subscription_id, customer_id,
# MAGIC             SUM(total_amount) AS total_billed,
# MAGIC             SUM(payment_amount) AS total_paid,
# MAGIC             COUNT_IF(payment_date > due_date OR payment_status = 'Late') AS total_late_payments,
# MAGIC             SUM(CASE WHEN payment_date > due_date OR payment_status = 'Late' THEN total_amount - payment_amount ELSE 0 END) AS total_late_fees,
# MAGIC             MAX(payment_status) AS latest_payment_status
# MAGIC         FROM billing
# MAGIC         WHERE customer_id = customer_id_input
# MAGIC         GROUP BY subscription_id, customer_id
# MAGIC     ) b ON s.subscription_id = b.subscription_id
# MAGIC     WHERE s.customer_id = customer_id_input
# MAGIC );

# COMMAND ----------

# DBTITLE 1,Probamos encadenando las dos funciones
# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM get_customer_billing_and_subscriptions(
# MAGIC   (SELECT customer_id FROM get_customer_by_email('john21@example.net'))
# MAGIC );
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 3. Darle al LLM una función Python para hacer cálculos
# MAGIC Los LLM suelen tener dificultades con las operaciones matemáticas. Agreguemos una herramienta que permita al LLM calcular cualquier expresión matemática usando Python directamente.
# MAGIC
# MAGIC Databricks lo hace fácil, ejecutando funciones Python de forma segura y aislada (sandbox):

# COMMAND ----------

# DBTITLE 1,Función Python muy sencilla
# -----------------------
# HERRAMIENTA 2: evaluar una expresión matemática
# -----------------------
def calculate_math_expression(expression: str) -> float:
    """
    Evalúa de forma segura una expresión matemática básica.

    Args:
        expression (str): Una expresión matemática (por ejemplo, "sqrt(2 + 3 * (4 - 1))"), usando funciones del módulo math de Python.

    Returns:
        float: El resultado de evaluar la expresión.
    """
    # Nota: el docstring se convierte en la descripción de la función en Unity Catalog.
    # Las palabras clave "Args:" y "Returns:" deben quedar en inglés para que el parser las reconozca.
    import math
    # Solo permitimos las funciones del módulo math (sqrt, sin, log...) más abs y round
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    allowed_names.update({"abs": abs, "round": round})

    # Una herramienta nunca debe lanzar excepciones: el agente a veces la llama con una expresión
    # vacía o inválida, y un error (UDF_USER_CODE_ERROR) abortaría toda la ejecución/evaluación.
    # Devolvemos NaN ante una entrada inválida para que el agente pueda recuperarse.
    if expression is None or str(expression).strip() == "":
        return float("nan")
    try:
        # eval sin builtins: solo se pueden usar los nombres permitidos
        result = eval(str(expression), {"__builtins__": None}, allowed_names)
        return float(result)
    except Exception:
        return float("nan")

# Prueba local de la función
print(calculate_math_expression("6 + 3 * (13 - 1)"))

# COMMAND ----------

# DBTITLE 1,Registrar la función Python en Unity Catalog
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

client = DatabricksFunctionClient()

# Despliega la herramienta en UC; los metadatos se toman automáticamente del docstring y de las anotaciones de tipo
python_tool_uc_info = client.create_python_function(func=calculate_math_expression, catalog=catalog, schema=dbName, replace=True)

# La herramienta queda registrada como `{catalog}.{schema}.{func}`, donde {func} es el nombre de la función
print(f"Nombre de la función desplegada en Unity Catalog: {python_tool_uc_info.full_name}")
# Enlace HTML a las funciones creadas
displayHTML(f'<a href="/explore/data/functions/{catalog}/{dbName}/calculate_math_expression" target="_blank">Ir a Unity Catalog para ver las funciones registradas</a>')

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Nota: también existe la función `system.ai.python_exec`, que permite al LLM ejecutar código generado en un entorno aislado:
# MAGIC
# MAGIC ```
# MAGIC
# MAGIC %sql
# MAGIC SELECT system.ai.python_exec("""
# MAGIC from datetime import datetime
# MAGIC print(datetime.now().strftime('%Y-%m-%d'))
# MAGIC """) as current_date
# MAGIC
# MAGIC ```
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## 4: Vamos al AI Playground para usar estas funciones y armar nuestro primer agente
# MAGIC
# MAGIC Abre el [Playground](/ml/playground) y selecciona las herramientas que creamos para probar tu agente.
# MAGIC
# MAGIC <div style="float: right; width: 70%;">
# MAGIC   <img
# MAGIC     src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/refs/heads/main/images/\
# MAGIC cross_demo_assets/AI_Agent_GIFs/AI_agent_function_selection.gif"
# MAGIC     alt="Selección de funciones"
# MAGIC     width="100%"
# MAGIC   >
# MAGIC </div>
# MAGIC
# MAGIC ### Guía de ubicación
# MAGIC
# MAGIC Tus funciones se organizan en Unity Catalog con esta estructura:
# MAGIC
# MAGIC #### Ruta de ejemplo:
# MAGIC `workspace.dbdemos_ai_agent.get_customer_by_email`
# MAGIC
# MAGIC 💡 Nota: si cambiaste el catálogo o el esquema en `config`, usa tus propios nombres.
# MAGIC
# MAGIC Prueba por ejemplo: *"Dame la información del cliente john21@example.net"*.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Siguiente paso: evaluación
# MAGIC
# MAGIC Nuestro agente ya está listo y usa nuestras herramientas para responder correctamente.
# MAGIC
# MAGIC Pero, ¿cómo asegurarnos de que funciona bien y, sobre todo, de que seguirá funcionando con preguntas y cambios futuros?
# MAGIC
# MAGIC Para eso necesitamos construir un dataset de evaluación y usar MLflow para analizar el agente automáticamente.
# MAGIC Abre el notebook [02-agent-eval/02.1_agent_evaluation]($../02-agent-eval/02.1_agent_evaluation) para ver cómo desplegar el agente con LangChain y ejecutar las primeras evaluaciones.
