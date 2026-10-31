# Databricks notebook source
# MAGIC %md
# MAGIC ## Generación del dataset de evaluación
# MAGIC
# MAGIC Crea un dataset sintético de preguntas (en español) sobre datos de clientes y facturación,
# MAGIC junto con los hechos esperados (`expected_facts`) que debería contener una respuesta correcta.
# MAGIC Los hechos esperados se generan con la función SQL `AI_QUERY` y el LLM configurado.

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

# Usamos el catálogo y el esquema de la demo
spark.sql(f'use catalog {catalog}')
spark.sql(f'use schema {dbName}')

# COMMAND ----------

# MAGIC %md
# MAGIC Generamos un dataset de evaluación sintético con preguntas sobre datos de clientes y facturación.

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id, pandas_udf, concat_ws, lit, col, from_json, struct, expr
from pyspark.sql.types import StringType, ArrayType
import pandas as pd

# Paso 1: tomamos 20 clientes y les asignamos un identificador de fila
df = spark.table("customers").limit(20).withColumn("row_id", monotonically_increasing_id())

# Paso 2: generamos una pregunta por cliente a partir de plantillas
@pandas_udf(StringType())
def generate_question(email: pd.Series, row_id: pd.Series) -> pd.Series:
    templates = [
        "¿Cuál es el número de teléfono de {email}?",
        "Lista todos los pedidos realizados por {email}.",
        "¿Cuál es el estado actual de la suscripción de {email}?",
        "Muestra los detalles de facturación de {email}.",
        "¿Tiene {email} facturas pendientes de pago?",
        "¿Qué productos compró {email}?",
        "¿Cuál es el nivel de fidelidad de {email}?",
        "¿Cuándo se registró {email} como cliente?",
        "Resume todas las suscripciones de {email}.",
        "¿En qué ciudad vive {email}?",
        "¿Cuál es el estado actual de la cuenta de {email}?",
        "¿Cuántos años lleva {email} como cliente?",
        "¿Cuál es el puntaje de riesgo de abandono (churn) de {email}?",
        "¿Cuál es el puntaje de valor de cliente de {email}?",
        "¿Tiene activado el pago automático la cuenta de {email}?",
        "¿Cuántos pagos atrasados ha tenido {email}?",
        "¿Cuál es el código postal de {email}?",
        "¿Qué tipo de cliente es {email} (por ejemplo, Individual o Business)?",
        "¿Cuál es la dirección completa de {email}?"
    ]
    # Asignamos las plantillas de forma rotativa según el row_id
    return pd.Series([
        templates[int(i) % len(templates)].format(email=e)
        for i, e in zip(row_id, email)
    ])

df = df.withColumn("question", generate_question("email", "row_id"))

# Paso 3: pasamos a pandas y volvemos a Spark para materializar el resultado de la UDF
df_pd = df.toPandas()
df_pd.attrs = {}  # En Spark Connect (serverless) attrs puede tener metadatos no serializables
df_clean = spark.createDataFrame(df_pd)

# Paso 4: construimos el prompt para AI_QUERY con todos los datos del cliente
df_clean = df_clean.withColumn(
    "prompt",
    concat_ws(
        " ",
        lit("Estás evaluando un sistema de IA."),
        lit("Con base en el siguiente registro de cliente:"),
        concat_ws(", ",
            df_clean.first_name, df_clean.last_name, df_clean.email, df_clean.phone,
            df_clean.address, df_clean.city, df_clean.state, df_clean.zip_code,
            df_clean.customer_segment, df_clean.registration_date.cast("string"),
            df_clean.customer_status, df_clean.loyalty_tier,
            df_clean.tenure_years.cast("string"), df_clean.churn_risk_score.cast("string"),
            df_clean.customer_value_score.cast("string")
        ),
        lit("Genera un arreglo JSON de afirmaciones factuales (expected_facts) que deberían incluirse en la respuesta correcta a la siguiente pregunta. Cada elemento debe ser una oración completa en español. Devuelve únicamente un arreglo JSON válido de strings, nada más."),
        lit("Pregunta:"), df_clean.question
    )
)

# Paso 5: registramos una vista temporal y llamamos a AI_QUERY con el LLM configurado
df_clean.createOrReplaceTempView("customer_test_questions")

final_df_raw = spark.sql(f"""
SELECT
  question,
  AI_QUERY("{LLM_ENDPOINT_NAME}", prompt) AS expected_facts_json
FROM customer_test_questions
""")

# Paso 6: convertimos el string JSON a Array<String>
final_df = final_df_raw.withColumn(
    "expected_facts",
    from_json(col("expected_facts_json"), ArrayType(StringType()))
)

# Paso 7: construimos el formato estructurado que espera MLflow (inputs / expectations)
eval_df = final_df.withColumn("inputs", struct("question")) \
                  .withColumn("predictions", lit("")) \
                  .withColumn("expectations", struct("expected_facts")) \
                  .select("inputs", "predictions", "expectations")

# Paso 8: guardamos el dataset como JSON en el volumen
eval_df.write.format('json').mode("overwrite").save(f"/Volumes/{catalog}/{dbName}/{volume_name}/eval_dataset")

# COMMAND ----------

# Mostramos el dataset leyendo el archivo guardado (así no se vuelve a ejecutar AI_QUERY)
display(spark.read.json(f"/Volumes/{catalog}/{dbName}/{volume_name}/eval_dataset"))
