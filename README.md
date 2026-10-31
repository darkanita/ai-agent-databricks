# Agente de IA con Databricks y MLflow 3 — versión en español para Free Edition

Traducción al español, con código comentado, de la demo **`ai-agent`** de [dbdemos](https://github.com/databricks-demos/dbdemos), adaptada para ejecutarse en **Databricks Free Edition (cómputo serverless)**.

Construiremos un agente de atención al cliente para una empresa de telecomunicaciones que:

1. Usa **funciones de Unity Catalog** (SQL y Python) como herramientas.
2. Se construye con **LangGraph** y se registra en **MLflow 3** como `ResponsesAgent`.
3. Se **evalúa** con jueces LLM de MLflow (`mlflow.genai.evaluate`) y se mejora con un mejor *system prompt*.
4. Incorpora una **base de conocimiento (RAG)** a partir de manuales PDF con `ai_parse_document` y **Vector Search**.
5. Se despliega como **endpoint de Model Serving** y como **chatbot (Gradio) en Databricks Apps**, con captura de feedback en MLflow.
6. Se **monitorea en producción** con jueces registrados en MLflow.
7. Usa herramientas de un **servidor MCP** gestionado por Databricks.

## Estructura

```
ai-agent/
├── config.py                                  # Catálogo, esquema, modelos y opciones (edita aquí)
├── 01-ai-agent-introduction.py                # Introducción y mapa de la demo
├── 01-create-tools/01_create_first_billing_agent.py
├── 02-agent-eval/02.1_agent_evaluation.py     # Agente, evaluación y despliegue
├── 02-agent-eval/agent.py                     # Código del agente (LangGraph + ResponsesAgent)
├── 03-knowledge-base-rag/03.1-pdf-rag-tool.py # RAG con PDFs y Vector Search
├── 04-deploy-app/04-Deploy-Frontend-Lakehouse-App.py
├── 04-deploy-app/chatbot_app/                 # App Gradio (main.py, app.yaml, requirements.txt)
├── 05-production-monitoring/05.production-monitoring.py
├── 06-improving-business-kpis/06-business-dashboard.py
├── 07-adding-mcp/07-agent_mcp.py
└── _resources/                                # Setup, generación de datos, PDFs y dataset de evaluación
```

## Cómo usarlo

1. Crea una cuenta gratuita en [Databricks Free Edition](https://www.databricks.com/learn/free-edition).
2. Importa el repositorio en tu workspace: **Workspace → Create → Git folder** y pega la URL de este repositorio.
   (También puedes descargar el ZIP e importarlo con **Import**.)
3. Abre cada notebook, conéctalo a **Serverless** y en el panel **Environment** elige la versión de entorno **2 o superior**.
4. Revisa `ai-agent/config.py` y ejecuta los notebooks **en orden**: 01 → 02 → 03 → 04 → 05 → (06, 07 opcionales).

> La primera ejecución del notebook 01 puede tardar varios minutos si hay que generar los datos y los manuales PDF.

## Adaptaciones para Free Edition

| Tema | Original | Esta versión |
|---|---|---|
| Catálogo | `main` | `workspace` (el catálogo por defecto en Free Edition) |
| Datos | Descarga desde GitHub/S3 | Intenta descargar; si el acceso a internet está restringido, los **genera localmente** (Faker) |
| PDFs | Generados con una URL de otro workspace + WeasyPrint | Generados con el LLM de **tu** workspace + `fpdf2` (100% Python); 8 manuales para cuidar la cuota |
| Modelos | Fijos | Se elige automáticamente el primer LLM / modelo de embeddings **disponible** |
| Vector Search | Crea `dbdemos_vs_endpoint` | Si ya tienes un endpoint (máximo 1 en Free Edition), **lo reutiliza** |
| `ai_parse_document` | Obligatorio | Si no está disponible, extrae el texto con `pypdf` |
| Datos sintéticos de evaluación | `generate_evals_df` | Si no está disponible, usa preguntas escritas a mano |
| Despliegue del endpoint | Obligatorio | Opcional (`DEPLOY_AGENT_ENDPOINT` en `config`); si falla por límites, el notebook continúa |
| Nombres de herramientas | `get_customer_billing_and_subscriptions` | `get_billing_and_subscriptions`: con el catálogo `workspace` el nombre completo `catalogo__esquema__funcion` superaba los 64 caracteres que aceptan los LLM y la herramienta fallaba |
| Idioma | Inglés | Notebooks, prompts, preguntas de evaluación y app en **español**. Los manuales PDF siguen en inglés (el modelo `gte-large-en` es para inglés) y el retriever pide la consulta en inglés |

### Límites de Free Edition a tener en cuenta

- Solo cómputo serverless; acceso a internet restringido a dominios de confianza.
- 1 endpoint de Vector Search, límite de endpoints de Model Serving activos.
- Hasta 3 Databricks Apps; cada app se detiene a las 24 h (se puede reiniciar).
- Agent Bricks no está disponible.
- Si superas la cuota diaria, el cómputo se detiene hasta que se restablezca.

El notebook **04** necesita el endpoint del agente desplegado en el notebook 02 o 03.

## Créditos y licencia

Basado en la demo `ai-agent` de [databricks-demos/dbdemos](https://github.com/databricks-demos/dbdemos) (Copyright Databricks, Inc.). Consulta `ai-agent/_resources/LICENSE.py` para los términos de la licencia original.
