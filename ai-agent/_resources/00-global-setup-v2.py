# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Notebook de configuración técnica (puedes ocultar los resultados de esta celda)
# MAGIC Inicializa el catálogo/esquema/volumen del usuario y define funciones auxiliares.
# MAGIC
# MAGIC No es necesario editar este notebook.
# MAGIC
# MAGIC *Nota: respecto a la versión original de dbdemos se eliminaron utilidades que esta demo no usa
# MAGIC (streams, mock de AutoML, permisos de grupos, etc.) para que el código sea más fácil de leer.*

# COMMAND ----------

# Widget opcional: si se pone en "true", se borra el esquema y se regeneran todos los datos
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reiniciar todos los datos")

# COMMAND ----------

import requests
import collections
import os

# COMMAND ----------

# MAGIC %md
# MAGIC #### Parche TEMPORAL de MLflow para serverless (ES-2047313)
# MAGIC `mlflow.pyfunc.spark_udf` / `score_batch` fallan actualmente en serverless con
# MAGIC `InvalidVersion: '18.x-...-photon-scala2'` porque MLflow interpreta la versión del runtime
# MAGIC esperando el formato `<major>.<minor>`. Este parche normaliza la versión en tiempo de ejecución.
# MAGIC
# MAGIC **Es una solución provisional. Se puede eliminar cuando salga la versión corregida de MLflow.**

# COMMAND ----------

import re, sys
_UNCUT_MINOR = 999

def _normalize_runtime_version(raw):
    # Convierte versiones como "18.x-photon-scala2" en "18.999" para que MLflow pueda interpretarlas
    if raw is None:
        return raw
    parts = str(raw).split(".")
    if not parts or not parts[0].isdigit():
        return raw
    major = int(parts[0]); minor = _UNCUT_MINOR
    if len(parts) > 1:
        m = re.match(r"(\d+)", parts[1]); minor = int(m.group(1)) if m else _UNCUT_MINOR
    return f"{major}.{minor}"

def _apply_mlflow_serverless_shim():
    # Parche para ES-2047313: no hace nada si las funciones internas de MLflow no existen.
    try:
        import mlflow.utils.databricks_utils as du
    except Exception as e:
        print(f"Parche serverless de mlflow omitido (mlflow no disponible): {e}")
        return
    # Guardamos la función original una sola vez (evita parchear dos veces si se re-ejecuta la celda)
    orig = getattr(du, "_es2047313_orig", None) or du.get_dbconnect_udf_sandbox_info
    du._es2047313_orig = orig
    def patched(spark):
        info = orig(spark)
        try:
            fixed = _normalize_runtime_version(info.runtime_version)
            if fixed != info.runtime_version:
                info = info.__class__(**{**info.__dict__, "runtime_version": fixed})
        except Exception:
            pass
        return info
    du.get_dbconnect_udf_sandbox_info = patched
    # Reemplazamos también las referencias ya importadas en otros módulos
    for mod in list(sys.modules.values()):
        if getattr(mod, "get_dbconnect_udf_sandbox_info", None) is orig:
            mod.get_dbconnect_udf_sandbox_info = patched
    try:
        DRV = du.DatabricksRuntimeVersion
    except Exception:
        return
    orig_parse = getattr(DRV, "_es2047313_orig_parse", None) or DRV.parse.__func__
    DRV._es2047313_orig_parse = orig_parse
    def patched_parse(cls, databricks_runtime=None):
        # Intenta el parser original y, si falla, interpreta la versión de forma tolerante
        try:
            return orig_parse(cls, databricks_runtime)
        except Exception:
            dbr = databricks_runtime or du.get_databricks_runtime_version()
            is_gpu = dbr.endswith("-gpu")
            if is_gpu: dbr = dbr[:-4]
            sp = dbr.split(".", maxsplit=2)
            if sp[0] == "client":
                major = int(re.match(r"(\d+)", sp[1]).group(1))
                minor = int(re.match(r"(\d+)", sp[2]).group(1)) if len(sp) > 2 and re.match(r"(\d+)", sp[2]) else 0
            else:
                major = int(re.match(r"(\d+)", sp[0]).group(1))
                minor = int(re.match(r"(\d+)", sp[1]).group(1)) if len(sp) > 1 and re.match(r"(\d+)", sp[1]) else _UNCUT_MINOR
            return cls(sp[0] == "client", major, minor, is_gpu)
    DRV.parse = classmethod(patched_parse)


def _apply_mlflow_log_model_sdk_shim():
    # En serverless, importlib.metadata.version("databricks-sdk") puede devolver None (el SDK viene
    # integrado en el runtime). MLflow 3.x hace Version(None) -> TypeError y rompe mlflow.*.log_model
    # al subir artefactos. Este parche tolera la versión ausente.
    import mlflow.store.artifact.databricks_sdk_artifact_repo as dsar
    import importlib.metadata as _im
    from packaging.version import Version
    def _supports():
        try:
            v = _im.version("databricks-sdk")
            return v is not None and Version(v) >= Version("0.45.0")
        except Exception:
            return False
    dsar._sdk_supports_large_file_uploads = _supports


# Aplicamos ambos parches; si fallan, solo mostramos un aviso (la demo puede seguir)
try:
    _apply_mlflow_serverless_shim()
    print("Parche serverless de mlflow aplicado (ES-2047313)")
except Exception as e:
    print(f"Parche serverless de mlflow no aplicado: {e}")

try:
    _apply_mlflow_log_model_sdk_shim()
    print("Parche de versión del databricks-sdk para log_model aplicado")
except Exception as e:
    print(f"Parche de log_model no aplicado: {e}")

# COMMAND ----------


class DBDemos():
  @staticmethod
  def setup_schema(catalog, db, reset_all_data, volume_name = None):
    """Crea (si no existen) el catálogo, el esquema y el volumen, y los deja como contexto actual."""
    if reset_all_data:
      # Borra el volumen y el esquema completo para empezar desde cero
      print(f'Borrando el volumen `{catalog}`.`{db}`.`{volume_name}`')
      try:
        spark.sql(f"DROP VOLUME IF EXISTS `{catalog}`.`{db}`.`{volume_name}`")
        spark.sql(f"DROP SCHEMA IF EXISTS `{catalog}`.`{db}` CASCADE")
      except Exception as e:
        print(f'El catálogo `{catalog}` o el esquema `{db}` no existen. Se omite el reinicio de datos')

    # La demo requiere Unity Catalog (no funciona con hive_metastore)
    assert catalog not in ['hive_metastore', 'spark_catalog'], "Esta demo solo soporta Unity Catalog. Cambia el nombre del catálogo."

    # Si el catálogo no existe, intentamos crearlo (en Free Edition usa "workspace", que ya existe)
    current_catalog = spark.sql("select current_catalog()").collect()[0]['current_catalog()']
    if current_catalog != catalog:
      catalogs = [r['catalog'] for r in spark.sql("SHOW CATALOGS").collect()]
      if catalog not in catalogs:
        spark.sql(f"CREATE CATALOG IF NOT EXISTS `{catalog}`")

    print(f"USE CATALOG `{catalog}`")
    spark.sql(f"USE CATALOG `{catalog}`")
    spark.sql(f"CREATE DATABASE IF NOT EXISTS `{db}`")

    print(f"Usando catalog.database `{catalog}`.`{db}`")
    spark.sql(f"USE `{catalog}`.`{db}`")

    # Volumen de UC para los archivos crudos
    if volume_name:
      spark.sql(f'CREATE VOLUME IF NOT EXISTS {volume_name};')


  @staticmethod
  def is_folder_empty(folder):
    """Devuelve True si la carpeta está vacía o no existe."""
    try:
      return len(dbutils.fs.ls(folder)) == 0
    except:
      return True

  @staticmethod
  def download_file_from_git(dest, owner, repo, path):
    """Descarga todos los archivos de una carpeta de un repositorio de GitHub hacia `dest`.

    En Free Edition el acceso a internet está limitado a dominios de confianza, por lo que
    esta descarga puede fallar; en ese caso 01-setup genera los datos localmente.
    """
    def download_file(url, destination):
      local_filename = url.split('/')[-1]
      # stream=True para no cargar el archivo completo en memoria
      with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        print('Guardando ' + destination + '/' + local_filename)
        with open(destination + '/' + local_filename, 'wb') as f:
          for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
      return local_filename

    if not os.path.exists(dest):
      os.makedirs(dest)
    from concurrent.futures import ThreadPoolExecutor
    # Listamos el contenido de la carpeta con la API de GitHub
    files = requests.get(f'https://api.github.com/repos/{owner}/{repo}/contents{path}', timeout=30).json()
    if not isinstance(files, list):
      # Por ejemplo, límite de peticiones de la API de GitHub o dominio bloqueado
      raise Exception(f"No se pudo listar {path} en GitHub: {files}")
    files = [f['download_url'] for f in files if 'NOTICE' not in f['name']]
    def download_to_dest(url):
      try:
        # Intentamos primero el bucket S3 de dbdemos (evita los límites de GitHub)
        s3url = url.replace("https://raw.githubusercontent.com/databricks-demos/dbdemos-dataset/main/", "https://dbdemos-dataset.s3.amazonaws.com/")
        download_file(s3url, dest)
      except:
        download_file(url, dest)
    # Descarga en paralelo con 10 hilos
    with ThreadPoolExecutor(max_workers=10) as executor:
      collections.deque(executor.map(download_to_dest, files))

# COMMAND ----------

# Ocultamos algunas advertencias para que la salida sea más limpia
import warnings
warnings.filterwarnings("ignore")
