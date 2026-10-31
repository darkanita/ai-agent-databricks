# Databricks notebook source
# MAGIC %md
# MAGIC # Generación de la documentación PDF (base de conocimiento)
# MAGIC
# MAGIC Genera manuales técnicos en PDF de equipos y servicios telco usando un LLM del propio workspace
# MAGIC y los guarda en el volumen `pdf_documentation`. Luego, el notebook `03.1-pdf-rag-tool` los parsea
# MAGIC y los indexa en Vector Search.
# MAGIC
# MAGIC **Cambios para Databricks Free Edition / serverless:**
# MAGIC - Se elimina la URL fija de otro workspace: el LLM se llama con el cliente de despliegue de MLflow,
# MAGIC   que usa automáticamente el host y las credenciales del workspace actual.
# MAGIC - Se reemplaza WeasyPrint (requiere librerías de sistema que no están en serverless) por `fpdf2`, 100% Python.
# MAGIC - Se generan pocos documentos (widget `num_docs`) para limitar el consumo de la cuota gratuita.
# MAGIC - Los manuales se generan en **inglés** porque el modelo de embeddings (`gte-large-en`) es para inglés.
# MAGIC   El agente responde en español igualmente.

# COMMAND ----------

# MAGIC %pip install fpdf2 "mlflow>=3.10.1"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

# Número de manuales a generar (los primeros de la lista de prioridad incluyen los usados en los ejemplos)
dbutils.widgets.text("num_docs", "8", "Número de manuales a generar")
NUM_DOCS = int(dbutils.widgets.get("num_docs"))

# COMMAND ----------

import os
import re
import shutil
import tempfile
import unicodedata
from concurrent.futures import ThreadPoolExecutor

def get_documentation_entities():
    """Lista de manuales a generar (equipos y servicios telco). Se mantienen en inglés porque
    el modelo de embeddings del índice (gte-large-en) está optimizado para inglés."""

    return [
        # Manuales de equipos (hardware)
        {
            "title": "Fiber Optic Modem User Manual",
            "category": "Hardware",
            "model": "FOM-2000X",
            "description": "High-speed fiber optic modem for residential and business use"
        },
        {
            "title": "ADSL Router Troubleshooting Guide",
            "category": "Hardware",
            "model": "ADSL-R500",
            "description": "Comprehensive troubleshooting for ADSL router connectivity issues"
        },
        {
            "title": "Mobile Hotspot Device Manual",
            "category": "Hardware",
            "model": "MHD-4G-Pro",
            "description": "4G/5G mobile hotspot device for on-the-go internet access"
        },
        {
            "title": "WiFi Extender Setup Guide",
            "category": "Hardware",
            "model": "WIFI-EXT-300",
            "description": "WiFi range extender installation and configuration manual"
        },
        {
            "title": "VoIP Phone System Manual",
            "category": "Hardware",
            "model": "VOIP-PRO-100",
            "description": "Voice over IP phone system setup and configuration"
        },
        {
            "title": "Cable Modem Troubleshooting",
            "category": "Hardware",
            "model": "CABLE-MODEM-500",
            "description": "Cable modem connectivity and performance troubleshooting"
        },
        
        # Documentación técnica
        {
            "title": "Internet Service Error Codes Reference",
            "category": "Technical",
            "model": "ERR-REF-2024",
            "description": "Complete reference guide for all internet service error codes"
        },
        {
            "title": "Network Security Configuration Manual",
            "category": "Security",
            "model": "SEC-NET-2024",
            "description": "Network security setup and configuration for telco equipment"
        },
        {
            "title": "Router Firmware Update Guide",
            "category": "Maintenance",
            "model": "FIRMWARE-UPDATE-001",
            "description": "Safe router firmware update procedures and troubleshooting"
        },
        {
            "title": "Network Monitoring Tools Manual",
            "category": "Monitoring",
            "model": "NET-MON-2024",
            "description": "Network monitoring and diagnostic tools usage guide"
        },
        
        # Instalación y configuración
        {
            "title": "Fiber Installation Guide",
            "category": "Installation",
            "model": "FIBER-INST-001",
            "description": "Step-by-step fiber optic installation and setup guide"
        },
        {
            "title": "Home Network Configuration Manual",
            "category": "Configuration",
            "model": "HOME-NET-001",
            "description": "Complete home network setup and configuration guide"
        },
        {
            "title": "Smart Home Network Setup",
            "category": "Configuration",
            "model": "SMART-HOME-NET",
            "description": "Smart home device network integration and setup"
        },
        
        # Servicios empresariales
        {
            "title": "Business Internet Setup Guide",
            "category": "Business",
            "model": "BIZ-INT-2024",
            "description": "Business-grade internet service setup and management"
        },
        {
            "title": "Internet Backup Solutions Guide",
            "category": "Business",
            "model": "BACKUP-INT-001",
            "description": "Internet backup solutions for business continuity"
        },
        {
            "title": "Internet Service Level Agreement Guide",
            "category": "Business",
            "model": "SLA-GUIDE-2024",
            "description": "Understanding and monitoring internet service level agreements"
        },
        
        # Diagnóstico y optimización
        {
            "title": "Internet Speed Test Guide",
            "category": "Diagnostics",
            "model": "SPEED-TEST-2024",
            "description": "How to test and optimize internet connection speeds"
        },
        {
            "title": "Network Performance Optimization",
            "category": "Optimization",
            "model": "PERF-OPT-2024",
            "description": "Network performance optimization techniques and best practices"
        },
        {
            "title": "Emergency Internet Connectivity Guide",
            "category": "Emergency",
            "model": "EMERG-INT-001",
            "description": "Emergency internet connectivity solutions and procedures"
        },
        {
            "title": "Internet Security Best Practices",
            "category": "Security",
            "model": "SEC-BEST-2024",
            "description": "Internet security best practices for home and business users"
        },
        
        # Documentación de atención y soporte al cliente
        {
            "title": "Customer Refund Policy Guide",
            "category": "Customer Service",
            "model": "REFUND-POLICY-2024",
            "description": "Complete guide to customer refund policies, procedures, and eligibility criteria"
        },
        {
            "title": "Equipment Return Policy Manual",
            "category": "Customer Service",
            "model": "RETURN-POLICY-2024",
            "description": "Equipment return procedures, restocking fees, and return authorization process"
        },
        {
            "title": "Service Cancellation Procedures",
            "category": "Customer Service",
            "model": "CANCEL-PROC-2024",
            "description": "Step-by-step service cancellation procedures and early termination fees"
        },
        {
            "title": "Customer Complaint Resolution Guide",
            "category": "Customer Service",
            "model": "COMPLAINT-RES-2024",
            "description": "Customer complaint handling procedures and escalation protocols"
        },
        {
            "title": "Service Upgrade and Downgrade Guide",
            "category": "Customer Service",
            "model": "SERVICE-CHANGE-2024",
            "description": "Service plan changes, upgrade procedures, and proration policies"
        },
        {
            "title": "Payment Plan and Billing Support Guide",
            "category": "Customer Service",
            "model": "BILLING-SUPPORT-2024",
            "description": "Payment plan options, billing dispute resolution, and payment methods"
        },
        {
            "title": "Customer Loyalty Program Guide",
            "category": "Customer Service",
            "model": "LOYALTY-PROG-2024",
            "description": "Loyalty program benefits, tier requirements, and reward redemption"
        },
        {
            "title": "Technical Support Escalation Procedures",
            "category": "Customer Service",
            "model": "ESCALATION-PROC-2024",
            "description": "When and how to escalate technical issues to higher support tiers"
        },
        {
            "title": "Customer Account Management Guide",
            "category": "Customer Service",
            "model": "ACCOUNT-MGMT-2024",
            "description": "Account changes, authorized users, and account security procedures"
        },
        {
            "title": "Service Outage Communication Protocol",
            "category": "Customer Service",
            "model": "OUTAGE-COMM-2024",
            "description": "How to communicate service outages and estimated restoration times"
        },
        {
            "title": "Customer Satisfaction Survey Procedures",
            "category": "Customer Service",
            "model": "SAT-SURVEY-2024",
            "description": "Customer satisfaction survey administration and follow-up procedures"
        },
        {
            "title": "Service Credit and Compensation Guide",
            "category": "Customer Service",
            "model": "SERVICE-CREDIT-2024",
            "description": "When and how to offer service credits for service issues"
        },
        {
            "title": "Customer Retention Strategies Guide",
            "category": "Customer Service",
            "model": "RETENTION-STRAT-2024",
            "description": "Customer retention strategies, win-back offers, and loyalty incentives"
        },
        {
            "title": "Multi-Language Support Procedures",
            "category": "Customer Service",
            "model": "MULTI-LANG-2024",
            "description": "Supporting customers in multiple languages and translation services"
        },
        {
            "title": "Accessibility Support Guidelines",
            "category": "Customer Service",
            "model": "ACCESSIBILITY-2024",
            "description": "Supporting customers with disabilities and accessibility requirements"
        },
        {
            "title": "Customer Privacy and Data Protection Guide",
            "category": "Customer Service",
            "model": "PRIVACY-PROT-2024",
            "description": "Customer data protection, privacy policies, and GDPR compliance"
        },
        {
            "title": "Social Media Customer Support Guide",
            "category": "Customer Service",
            "model": "SOCIAL-SUPPORT-2024",
            "description": "Handling customer inquiries and complaints on social media platforms"
        },
        {
            "title": "Customer Education and Training Materials",
            "category": "Customer Service",
            "model": "CUST-EDU-2024",
            "description": "Educational materials and training resources for customers"
        }
    ]



# Manuales que se generan primero porque se usan en los ejemplos de los notebooks
PRIORITY_MODELS = ["ADSL-R500", "ERR-REF-2024", "SAT-SURVEY-2024", "FIRMWARE-UPDATE-001",
                   "RETURN-POLICY-2024", "REFUND-POLICY-2024", "FOM-2000X", "WIFI-EXT-300"]


def get_prioritized_entities(n):
    """Devuelve los n primeros manuales, poniendo primero los de PRIORITY_MODELS."""
    entities = get_documentation_entities()
    entities.sort(key=lambda e: PRIORITY_MODELS.index(e["model"]) if e["model"] in PRIORITY_MODELS else 999)
    return entities[:n]

# COMMAND ----------

SYSTEM_PROMPT = """You are an expert technical documentation writer for telecommunications equipment and services.
Write in English, with a professional tone. Output plain text using this light markdown only:
'# ' for the document title, '## ' for sections, '### ' for sub-sections, '- ' for bullets and '1. ' for numbered steps.
Do not use tables, code fences, bold markers or HTML."""


def get_documentation_prompt(entity):
    """Construye el prompt para un manual concreto."""
    return f"""Write a complete documentation manual (about 1500-2500 words) for:

TITLE: {entity['title']}
CATEGORY: {entity['category']}
MODEL: {entity['model']}
DESCRIPTION: {entity['description']}

The manual will be used as the only source of a RAG customer-support assistant, so be explicit and self-contained.
Mention the model name ({entity['model']}) in every section. Include, when applicable:
- Overview and technical specifications (with concrete numbers)
- Installation and setup (numbered steps)
- How to restart / reset the device or service, step by step
- Error code reference: at least 6 error codes (for example Error 01, Error 1001...), each with description, symptoms, cause, numbered resolution steps and escalation policy
- Troubleshooting scenarios
- Firmware / maintenance procedures
- Warranty, return and refund policy
- FAQ with at least 8 questions and answers
- Support and escalation contacts"""


def call_llm_api(prompt: str) -> str:
    """Llama al LLM configurado (LLM_ENDPOINT_NAME) usando las credenciales del workspace actual."""
    from mlflow.deployments import get_deploy_client
    client = get_deploy_client("databricks")
    response = client.predict(
        endpoint=LLM_ENDPOINT_NAME,
        inputs={
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 4000,
        },
    )
    content = response["choices"][0]["message"]["content"]
    # Algunos modelos devuelven el contenido como lista de bloques (p. ej. razonamiento + texto)
    if isinstance(content, list):
        content = "\n".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
    return content

# COMMAND ----------

def to_latin1(text: str) -> str:
    """Las fuentes base de fpdf2 solo soportan latin-1: reemplazamos los caracteres no soportados."""
    replacements = {"–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"',
                    "•": "-", "…": "...", " ": " ", "→": "->", "≤": "<=", "≥": ">="}
    for k, v in replacements.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("latin-1", "ignore").decode("latin-1")


def break_long_words(line: str, max_len: int = 60) -> str:
    """Corta palabras muy largas (URLs, comandos) para que fpdf2 pueda ajustarlas al ancho de página."""
    return " ".join(w if len(w) <= max_len else " ".join(w[i:i + max_len] for i in range(0, len(w), max_len))
                    for w in line.split(" "))


def convert_text_to_pdf(text: str, output_path: str):
    """Convierte el texto (markdown ligero) a PDF con fpdf2 y lo guarda en output_path."""
    from fpdf import FPDF
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    for raw in to_latin1(text).splitlines():
        line = break_long_words(raw.strip().replace("**", "").replace("`", ""))
        if not line:
            pdf.ln(2)  # línea en blanco = pequeño espacio vertical
            continue
        # Estilo según el tipo de línea: título, sección, sub-sección o texto normal
        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 18); line = line[2:]
        elif line.startswith("## "):
            pdf.ln(3); pdf.set_font("Helvetica", "B", 14); line = line[3:]
        elif line.startswith("### "):
            pdf.set_font("Helvetica", "B", 12); line = line[4:]
        else:
            pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
    # Escribimos primero en disco local y luego copiamos al volumen de UC
    tmp = os.path.join(tempfile.mkdtemp(), os.path.basename(output_path))
    pdf.output(tmp)
    shutil.copy(tmp, output_path)
    print(f"  ✓ PDF guardado: {output_path}")

# COMMAND ----------

def generate_one(entity, folder):
    """Genera un manual: llama al LLM y convierte la respuesta a PDF."""
    try:
        text = call_llm_api(get_documentation_prompt(entity))
        filename = f"{entity['model'].replace('-', '_').lower()}_manual.pdf"  # p. ej. adsl_r500_manual.pdf
        convert_text_to_pdf(text, f"{folder}/{filename}")
    except Exception as e:
        print(f"  ✗ Error generando {entity['title']}: {str(e)[:300]}")


def generate_documentation_manuals(path: str, n: int):
    """Genera n manuales en paralelo (4 a la vez) dentro de {path}/pdf_documentation."""
    folder = f"{path}/pdf_documentation"
    dbutils.fs.mkdirs(folder)
    entities = get_prioritized_entities(n)
    print(f"Generando {len(entities)} manuales con el modelo {LLM_ENDPOINT_NAME}...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda e: generate_one(e, folder), entities))
    print(f"\n¡Documentación generada! Archivos guardados en '{folder}'.")

# COMMAND ----------

path = f'/Volumes/{catalog}/{schema}/{volume_name}'
generate_documentation_manuals(path, NUM_DOCS)

# Mostramos los archivos generados
display(dbutils.fs.ls(f"{path}/pdf_documentation"))
