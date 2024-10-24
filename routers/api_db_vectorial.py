from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from database_vectorial.model_db import VectorialDB
import os
import shutil

router = APIRouter(
    prefix="/database",
    tags=["Database Vectorial"],
    responses={404: {"description": "No encontrado"}},
)

vectorial_db = VectorialDB()


class QueryModel(BaseModel):
    query: str
    k: int = 3
    source: str = None


def clear_docs_folder():
    """Elimina todos los archivos de la carpeta 'docs' después de su uso."""
    folder = vectorial_db.docs_path
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Eliminar archivos o enlaces simbólicos
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Eliminar directorios
        except Exception as e:
            print(f"No se pudo borrar {file_path}. Error: {e}")


# Ruta raíz
@router.get("/")
def read_root():
    return {"message": "API de base de datos vectorial"}


# Listar bases de datos vectoriales
@router.get("/list")
def list_databases():
    list_databases = vectorial_db.list_vector_stores()
    dict_list_databases = {i: database for i, database in enumerate(list_databases)}
    return dict_list_databases


@router.post("/create/{database_name}")
def create_database(database_name: str, files: List[UploadFile] = File(...)):
    # Crear carpeta docs si no existe
    if not os.path.exists(vectorial_db.docs_path):
        os.makedirs(vectorial_db.docs_path)

    # Guardar los archivos subidos en la carpeta docs
    for file in files:
        file_path = os.path.join(vectorial_db.docs_path, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    # Crear la base de datos vectorial
    try:
        vectorial_db.create_vector_store(database_name)  # Usar await aquí
        return {"message": f"Base de datos '{database_name}' creada con éxito."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Limpiar la carpeta docs después de crear la base de datos
        clear_docs_folder()


# Buscar similitud en una base de datos vectorial
@router.post("/search/{database_name}")
def search_database(database_name: str, query: QueryModel):
    try:
        results = vectorial_db.search_similarity(
            name=database_name, query=query.query, k=query.k, source=query.source
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Actualizar base de datos vectorial (agregando nuevos archivos)
@router.put("/update/{database_name}")
def update_database(database_name: str, files: List[UploadFile] = File(...)):
    # Guardar los nuevos archivos en la carpeta docs
    for file in files:
        file_path = os.path.join(vectorial_db.docs_path, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    # Actualizar la base de datos vectorial
    try:
        vectorial_db.update_vector_store(database_name)
        return {"message": f"Base de datos '{database_name}' actualizada con éxito."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Limpiar la carpeta docs después de actualizar la base de datos
        clear_docs_folder()


# Eliminar base de datos vectorial
@router.delete("/delete/{database_name}")
def delete_database(database_name: str):
    try:
        if vectorial_db.delete_vector_store(database_name):
            return {"message": f"Base de datos '{database_name}' eliminada con éxito."}
        else:
            raise HTTPException(status_code=404, detail="Base de datos no encontrada")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Buscar si existe una base de datos con un nombre específico
@router.get("/exists/{database_name}")
def database_exists(database_name: str):
    if vectorial_db.database_exists(database_name):
        return True
    else:
        return False
