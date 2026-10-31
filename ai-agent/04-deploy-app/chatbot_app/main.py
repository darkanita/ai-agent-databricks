from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from enum import Enum
from typing import Optional
import gradio as gr
import os
from gradio.themes.utils import sizes
from databricks.sdk import WorkspaceClient
import mlflow
from mlflow.deployments import get_deploy_client
import json

# Chatbot Gradio servido con FastAPI en una Databricks App.
# Llama al endpoint del agente y permite enviar feedback (pulgar arriba/abajo) a MLflow.

# Las trazas y el feedback se registran en el tracking server de MLflow del workspace
mlflow.set_tracking_uri("databricks")

# --- Configuración de FastAPI ---
app = FastAPI()
router = APIRouter()
app.include_router(router)

# --- Configuración de Databricks Model Serving ---
# Dentro de la app, WorkspaceClient se autentica automáticamente con el service principal de la app
w = WorkspaceClient()
# Lista de endpoints disponibles para el desplegable de la interfaz
available_endpoints = [x.name for x in w.serving_endpoints.list()]

# --- Lógica del chatbot ---
def respond(message, history, dropdown):
    """Envía el mensaje al endpoint seleccionado y devuelve el historial actualizado del chat."""
    if len(message.strip()) == 0:
        return [
            *history,
            gr.ChatMessage(
                content="ERROR: la pregunta no puede estar vacía.",
                role="assistant"
            )
        ]

    try:
        # Cliente de despliegue de MLflow para consultar el endpoint de serving
        client = get_deploy_client("databricks")
        input_message = [{
            "content": message,
            "role": "user",
            "type": "message"
        }]

        response = client.predict(
            endpoint=dropdown,
            inputs={'input': input_message, "databricks_options": {
                # Pedimos la traza para obtener el trace_id y poder asociarle el feedback
                "return_trace": True
            }}
        )
        print(f"Respuesta: {response}")
        output_messages = response['output']
        trace_id = response.get("databricks_output", {}).get("trace", {}).get("info", {}).get("trace_id")
        print(f"ID de traza: {trace_id}")

        thoughts = []
        final_msg = ""

        # Primera pasada: recopilamos las llamadas a funciones y sus resultados como "pensamientos" del agente
        for msg in output_messages:
            if msg.get('type') == 'function_call':
                function_name = msg.get('name', 'Unknown function')
                thoughts.append(f"🔧 Llamando a la función: {function_name}")

            elif msg.get('type') == 'function_call_output':
                try:
                    output_data = json.loads(msg.get('output', '{}'))
                    if 'value' in output_data:
                        thoughts.append(f"📊 La función devolvió datos")
                except Exception:
                    thoughts.append("📊 Función completada")

        # Segunda pasada: buscamos el mensaje final del asistente
        for msg in output_messages:
            if msg.get('role') == 'assistant' and msg.get('content'):
                for content in msg['content']:
                    if content['type'] == 'output_text':
                        text = content['text'].strip()
                        if text and text != message.strip():
                            final_msg = text
                            break
                if final_msg:
                    break

        chat_msgs = [
            *history,
            gr.ChatMessage(
                content=message,
                role="user"
            )
        ]

        # Los pensamientos se muestran en un acordeón usando metadata={'title': ...} de Gradio
        if thoughts:
            thoughts_text = "\n\n".join(thoughts)
            chat_msgs.append(
                gr.ChatMessage(
                    content=thoughts_text,
                    role="assistant",
                    metadata={"title": "🤔 Pensamientos del agente"}
                )
            )

        if final_msg:
            chat_msgs.append(
                gr.ChatMessage(
                    content=final_msg,
                    role="assistant",
                    # Guardamos el trace_id en las opciones del mensaje para usarlo al recibir feedback
                    options=[{"value": trace_id, "label": "trace_id"}] if trace_id else []
                )
            )
        print(f"Mensajes del chat: {chat_msgs}")
        return chat_msgs

    except Exception as error:
        return [
            *history,
            gr.ChatMessage(content=message, role="user"),
            gr.ChatMessage(
                content=f"ERROR al consultar el endpoint {dropdown}: {error}",
                role="assistant"
            )
        ]

# --- Interfaz Gradio con botones de feedback ---
theme = gr.themes.Soft(
    text_size=sizes.text_sm,
    radius_size=sizes.radius_sm,
    spacing_size=sizes.spacing_sm,
)

with gr.Blocks(theme=theme) as demo:
    gr.Markdown(
        """
        # Demo de agente de IA en Databricks - Asistente de datos de clientes

        Este agente de IA te ayuda a consultar datos de clientes y obtener información útil.  
        Usa herramientas para acceder a las bases de datos y puede responder preguntas sobre clientes, sus segmentos, facturación y documentación técnica de productos.  
        Es un ejemplo de demostración de las capacidades de los agentes de IA con Databricks.
        """
    )

    endpoint_dropdown = gr.Dropdown(
        choices=available_endpoints,
        value=os.environ["MODEL_SERVING_ENDPOINT"],
        label="Endpoint de serving"
    )

    chatbot = gr.Chatbot(
        show_label=False,
        container=False,
        show_copy_button=True,
        type="messages"
    )
    
    with gr.Row():
        textbox = gr.Textbox(
            placeholder="Pregunta sobre los datos de clientes...",
            container=False,
            scale=7
        )
        send_button = gr.Button("Enviar", variant="primary")

    feedback_output = gr.Markdown(visible=False)

    # --- Manejo del feedback: extrae el trace_id de las opciones del mensaje ---
    def handle_feedback(history, like_data: gr.LikeData):
        """Registra en MLflow el feedback (like/dislike) asociado a la traza de la respuesta."""
        idx = like_data.index

        print(f"Historial en handle_feedback: {history}")

        # Extraemos el trace_id de las opciones del mensaje del asistente
        trace_id = next((v.get('value') for m in history if m['role'] == 'assistant' for v in m['options'] if v.get('label') == 'trace_id'), None)

        if not trace_id:
            return gr.Markdown("No se encontró un ID de traza para este mensaje.", visible=True)
        try:
            mlflow.log_feedback(
                trace_id=trace_id,
                name='user_feedback',
                value=True if like_data.liked else False,
                rationale=None,
                source=mlflow.entities.AssessmentSource(
                    source_type='HUMAN',
                    source_id='user',
                ),
            )
            return gr.Markdown(f"¡Gracias por tu feedback! Enviado a MLflow: {trace_id}", visible=True)
        except Exception as e:
            return gr.Markdown(f"Error al enviar el feedback: {str(e)}", visible=True)

    def chat_submit(message, history, endpoint):
        return respond(message, history, endpoint)

    with gr.Row():
        chatbot
        feedback_output

    # Enviar con Enter o con el botón
    textbox.submit(
        chat_submit,
        [textbox, chatbot, endpoint_dropdown],
        chatbot
    )
    
    send_button.click(
        chat_submit,
        [textbox, chatbot, endpoint_dropdown],
        chatbot
    )

    # Asociamos el manejador de feedback a los botones like/dislike del chatbot
    chatbot.like(handle_feedback, chatbot, feedback_output)

    gr.Examples(
        examples=[
            ["Dame la información de john21@example.net"],
            ["¿Cómo reinicio mi router WIFI ADSL-R500?"],
            ["Resume todas las suscripciones de john21@example.net"]
        ],
        inputs=[textbox]
    )

# Iniciamos el servidor en el puerto 8000 (el que espera Databricks Apps)
demo.queue(default_concurrency_limit=100)
demo.launch(server_name="0.0.0.0", server_port=8000)
