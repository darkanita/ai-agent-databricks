# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC # Monitoreo en producción: calidad automatizada a escala
# MAGIC
# MAGIC El monitoreo en producción de MLflow ejecuta automáticamente evaluaciones de calidad sobre una muestra del tráfico de producción, para que tu aplicación de IA generativa mantenga altos estándares de calidad sin intervención manual. MLflow permite usar en producción las mismas métricas que definiste para la evaluación offline, logrando una evaluación consistente en todo el ciclo de vida: de desarrollo a producción.
# MAGIC
# MAGIC **Beneficios clave:**
# MAGIC
# MAGIC - Evaluación automatizada: ejecuta jueces LLM sobre las trazas de producción con tasas de muestreo configurables
# MAGIC - Evaluación continua de la calidad: monitorea métricas de calidad en tiempo real sin afectar la experiencia del usuario
# MAGIC - Monitoreo costo-eficiente: estrategias de muestreo para equilibrar cobertura y costo de cómputo
# MAGIC
# MAGIC El monitoreo en producción te permite desplegar con confianza, sabiendo que detectarás los problemas de forma proactiva antes de que tengan un gran impacto en los usuarios.
# MAGIC
# MAGIC Para más detalles, consulta la documentación [Monitor served models using AI Gateway-enabled inference tables](https://docs.databricks.com/aws/en/ai-gateway/inference-tables) y [Production quality monitoring](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/production-monitoring).
# MAGIC
# MAGIC <img src="https://i.imgur.com/wv4p562.gif">
# MAGIC
# MAGIC *Free Edition: el monitoreo crea un job programado. Free Edition permite hasta 5 tareas de job concurrentes y el consumo cuenta para la cuota diaria.*

# COMMAND ----------

# DBTITLE 1,Instalación de librerías
# MAGIC %pip install "mlflow[databricks]>=3.10.1"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/01-setup

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Creemos nuestro monitor de nivel productivo
# MAGIC
# MAGIC Puedes crear el monitor fácilmente desde la interfaz o directamente con el SDK:
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Elige el tipo de juez (scorer) según cuánta personalización y control necesites. Cada enfoque se apoya en el anterior, agregando complejidad y control. Empieza con los [jueces integrados](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/scorers) y los [jueces Guidelines](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/judges/guidelines) para una evaluación rápida. A medida que tus necesidades evolucionen, crea [jueces LLM personalizados](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/custom-judge/create-custom-judge) para criterios del dominio y [scorers basados en código](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/custom-scorers) para lógica de negocio determinista.
# MAGIC
# MAGIC | Enfoque                     | Nivel de personalización | Caso de uso                                                                                 |
# MAGIC |-----------------------------|--------------------------|---------------------------------------------------------------------------------------------|
# MAGIC | Jueces integrados           | Mínimo                   | Probar rápidamente la evaluación con LLM usando scorers integrados como `Safety` y `RetrievalGroundedness`. |
# MAGIC | Jueces Guidelines           | Moderado                 | Juez integrado que verifica si las respuestas cumplen reglas personalizadas en lenguaje natural, como pautas de estilo o de veracidad. |
# MAGIC | Jueces LLM personalizados   | Total                    | Jueces LLM totalmente personalizados con criterios de evaluación detallados.<br><br>Pueden devolver puntajes numéricos, categorías o valores booleanos. |
# MAGIC | Scorers basados en código   | Total                    | Scorers programáticos y deterministas que evalúan coincidencias exactas, validación de formato y métricas de rendimiento. |
# MAGIC
# MAGIC Esta demo muestra el uso de jueces integrados, jueces Guidelines y jueces LLM personalizados.

# COMMAND ----------

from mlflow.genai.scorers import RetrievalGroundedness, RelevanceToQuery, Safety
import mlflow

# Reutilizamos el experimento existente del notebook 02 (donde están las trazas del agente)
xp_name = os.getcwd().rsplit("/", 1)[0] + "/02-agent-eval/02.1_agent_evaluation"
mlflow.set_experiment(xp_name)

# Registramos los jueces integrados en el experimento (registrar no los activa todavía)
safety_judge = Safety().register(name="safety")
groundedness_judge = RetrievalGroundedness().register(name="groundedness")
relevance_judge = RelevanceToQuery().register(name="relevance_to_query")

# COMMAND ----------

# MAGIC %md
# MAGIC Los [jueces LLM Guidelines](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/judges/guidelines#1-built-in-guidelines-judge-global-guidelines) usan criterios de aprobado/reprobado en lenguaje natural para evaluar las salidas de IA generativa.
# MAGIC Un juez `Guidelines()` aplica pautas globales de forma uniforme a todas las filas, evaluando solo las entradas/salidas de la app. Funciona tanto en evaluación offline como en monitoreo en producción.
# MAGIC <br><br>Los jueces Guidelines usan un LLM ajustado especialmente para evaluar si el texto cumple los criterios indicados. El juez:
# MAGIC * **Recibe contexto:** cualquier diccionario JSON con los datos a evaluar (por ejemplo, request, response, retrieved_documents, user_preferences). Puedes referirte a estas claves por nombre en tus pautas
# MAGIC * **Aplica las pautas:** tus reglas en lenguaje natural que definen cuándo aprueba o reprueba
# MAGIC * **Emite un juicio:** devuelve un resultado binario (aprobado/reprobado) con una justificación detallada

# COMMAND ----------

from mlflow.genai.scorers import Guidelines

# Pautas de exactitud adaptadas al contexto de esta demo (soporte telco)
accuracy_guidelines = [
  """
  La respuesta cita correctamente toda la información factual de los datos proporcionados, según estas reglas:
    - Toda la información factual debe provenir directamente de los datos proporcionados, SIN inventar nada
    - Los nombres, fechas, números e identificadores (cliente, suscripción, factura) deben ser 100% exactos
    - Los montos de facturación, estados de pago y detalles de suscripción deben coincidir exactamente con los datos
    - Los procedimientos técnicos y códigos de error deben corresponder a la documentación recuperada
    - FALLA AUTOMÁTICA si se menciona cualquier información que no esté explícitamente en los datos
  """,
]

accuracy_judge = Guidelines(name="accuracy_guidelines", guidelines=accuracy_guidelines).register(name="accuracy")


# COMMAND ----------

# MAGIC %md
# MAGIC Los [jueces LLM personalizados](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/custom-judge/) permiten definir criterios de puntuación complejos y matizados para aplicaciones de IA generativa usando lenguaje natural.
# MAGIC
# MAGIC Aunque los [jueces LLM integrados](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/scorers) de MLflow son un excelente punto de partida para las dimensiones de calidad comunes, los jueces creados con `make_judge()` te dan control total sobre los criterios de evaluación.
# MAGIC
# MAGIC Para crear un juez, escribes un prompt con instrucciones en lenguaje natural sobre cómo evaluar la calidad del agente. `make_judge()` acepta variables de plantilla para acceder a las entradas, salidas, resultados o comportamientos esperados e incluso a las trazas completas del agente.
# MAGIC
# MAGIC Las instrucciones deben incluir al menos una variable de plantilla, pero no es necesario usarlas todas.
# MAGIC
# MAGIC * `{{ inputs }}` - datos de entrada del agente
# MAGIC * `{{ outputs }}` - salida generada por el agente
# MAGIC * `{{ expectations }}` - verdades de referencia o resultados esperados
# MAGIC * `{{ trace }}` - la traza completa de ejecución del agente
# MAGIC
# MAGIC Son las únicas variables permitidas. Variables personalizadas como `{{ question }}` producen errores de validación, para garantizar un comportamiento consistente y evitar problemas de inyección en las plantillas.

# COMMAND ----------

from mlflow.genai import make_judge

# Juez personalizado que verifica que la respuesta no muestre razonamiento ni herramientas
steps_and_reasoning_judge = make_judge(
  name="steps_and_reasoning_validator",
  instructions=(
    "Analiza la respuesta del agente: {{ outputs }} para verificar si contiene pasos de razonamiento.\n"
    "La respuesta debe darse sin mostrar el razonamiento:\n"
    "  - no mencionar que necesita buscar o consultar información\n"
    "  - no mencionar las herramientas o funciones utilizadas\n"
    "  - no describir los pasos intermedios ni el razonamiento\n"
    "Califica como true: la respuesta no contiene pasos ni razonamiento\n"
    "Califica como false: la respuesta contiene pasos o razonamiento\n"
    "Tu respuesta debe ser un booleano: true o false"
  ),
  feedback_value_type=bool,
)

steps_and_reasoning_judge = steps_and_reasoning_judge.register(name="steps_and_reasoning")

# COMMAND ----------

# MAGIC %md
# MAGIC Los jueces basados en trazas analizan las trazas de ejecución para entender qué pasó durante la ejecución del agente. Exploran las trazas de forma autónoma usando herramientas de Model Context Protocol (MCP) y pueden:
# MAGIC
# MAGIC * Validar patrones de uso de herramientas
# MAGIC * Identificar cuellos de botella de rendimiento
# MAGIC * Investigar fallas de ejecución
# MAGIC * Verificar flujos de trabajo de varios pasos
# MAGIC
# MAGIC El siguiente ejemplo define un juez que evalúa si las llamadas a herramientas fueron correctas analizando las trazas:

# COMMAND ----------

# Juez basado en trazas que valida las llamadas a herramientas
tool_call_judge = make_judge(
    name="tool_call_correctness",
    instructions=(
        "Analiza la traza de ejecución {{ trace }} para determinar si el agente llamó a las herramientas adecuadas para la solicitud del usuario.\n"
        "Examina la traza para:\n"
        "1. Identificar qué herramientas estaban disponibles y su propósito\n"
        "2. Determinar qué herramientas se llamaron realmente\n"
        "3. Evaluar si las llamadas a herramientas fueron razonables para responder la pregunta del usuario\n"
        "Evalúa el uso de herramientas y responde con un valor booleano:\n"
        "- true: el agente llamó a las herramientas correctas para atender la solicitud\n"
        "- false: el agente llamó herramientas incorrectas, omitió herramientas necesarias o llamó herramientas innecesarias\n"
        "Tu respuesta debe ser un booleano: true o false."
    ),
    feedback_value_type=bool,
    # Para analizar una traza completa, el juez basado en trazas necesita un modelo explícito.
    # Usamos el LLM elegido en config (en Free Edition no todos los modelos están disponibles).
    model=f"databricks:/{LLM_ENDPOINT_NAME}"
    )

tool_call_judge = tool_call_judge.register(name="tool_call_correctness")

# COMMAND ----------

# Iniciar el monitoreo en producción crea un job que se ejecuta periódicamente (genera un pequeño costo)
# Descomenta para iniciar el monitoreo en producción sobre tu experimento.
from mlflow.genai.scorers import list_scorers, ScorerSamplingConfig
scorers = list_scorers()
print("Jueces registrados:", [s.name for s in scorers])
# for scorer in scorers:
#   scorer.start(sampling_config=ScorerSamplingConfig(sample_rate=1.0)) # evalúa todas las trazas (sample_rate=1.0)

# COMMAND ----------

# Descomenta para detener y eliminar todos los jueces del experimento
from mlflow.genai.scorers import list_scorers, delete_scorer
import mlflow
import os

xp_name = os.getcwd().rsplit("/", 1)[0] + "/02-agent-eval/02.1_agent_evaluation"
mlflow.set_experiment(xp_name)
scorers = list_scorers()
# for scorer in scorers:
#   scorer.stop()
#   delete_scorer(name=scorer.name)

# COMMAND ----------

# MAGIC %md
# MAGIC Si iniciaste el monitoreo, recuerda eliminar manualmente el job de monitoreo en *Jobs & Pipelines* cuando ya no lo necesites.

# COMMAND ----------

# MAGIC %md
# MAGIC El job de monitoreo tarda unos 15-30 minutos en ejecutarse por primera vez. Después de la ejecución inicial, se ejecuta cada 15 minutos. Si tienes un gran volumen de tráfico en producción, el job puede tardar más.
# MAGIC
# MAGIC Cada vez que el job se ejecuta:
# MAGIC
# MAGIC 1. Ejecuta cada juez configurado sobre la muestra de trazas.
# MAGIC   Si hay distintas tasas de muestreo por juez, el job intenta evaluar la mayor cantidad posible de las mismas trazas. Por ejemplo, si el juez A tiene una tasa del 20% y el juez B del 40%, se usará el mismo 20% de trazas para A y B.
# MAGIC 2. Adjunta el feedback del juez a cada traza en el experimento de MLflow indicado
# MAGIC 3. Escribe una copia de TODAS las trazas (no solo las muestreadas) en la tabla Delta `trace_logs_<MLflow_experiment_id>`
# MAGIC   Puedes ver los resultados del monitoreo en la pestaña Traces del experimento de MLflow, o consultar las trazas con SQL o Spark en la tabla Delta generada.
