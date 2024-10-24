import asyncio
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    Docx2txtLoader,
    DirectoryLoader,
)
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import shutil

# Cargar las variables de entorno
load_dotenv()

# Crear las carpetas docs y database en caso de que no existan
if not os.path.exists("docs"):
    os.makedirs("docs")

if not os.path.exists("database"):
    os.makedirs("database")


class VectorialDB:
    """
    Clase para el manejo de la base de datos vectorial.

    Attributes:
        embeddings: Modelo de embeddings de HuggingFace.

    Methods:
        get_embeddings: Devuelve el modelo de embeddings.
        files_to_texts: Carga los documentos de la carpeta docs.
        create_vector_store: Crea un vector store y lo guarda en la carpeta database.
        load_vector_store: Carga un vector store de la carpeta database.
        update_vector_store: Actualiza un vector store con los documentos de la carpeta docs.
        search_similarity: Realiza una búsqueda de similitud en un vector store.
        delete_vector_store: Elimina un vector store de la carpeta database.

    """

    def __init__(self):
        # Definir el modelo de embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={
                "trust_remote_code": True,
                "device": os.getenv("DEVICE", "cpu"),
            },
        )
        self.docs_path = "docs"
        self.database_path = "database"

    def get_embeddings(self):
        return self.embeddings

    def files_to_texts(self):
        loaders = []

        # Verificar si hay archivos PDF en la carpeta docs
        if any(fname.endswith(".pdf") for fname in os.listdir(self.docs_path)):
            pdf_loader = DirectoryLoader(
                path=self.docs_path, glob="*.pdf", loader_cls=PyMuPDFLoader
            )
            loaders.append(pdf_loader)

        # Verificar si hay archivos TXT en la carpeta docs
        if any(fname.endswith(".txt") for fname in os.listdir(self.docs_path)):
            text_loader = DirectoryLoader(
                path=self.docs_path,
                glob="*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
            )
            loaders.append(text_loader)

        # Verificar si hay archivos DOCX en la carpeta docs
        if any(fname.endswith(".docx") for fname in os.listdir(self.docs_path)):
            docx_loader = DirectoryLoader(
                path=self.docs_path, glob="*.docx", loader_cls=Docx2txtLoader
            )
            loaders.append(docx_loader)

        # Verificar si hay archivos DOC en la carpeta docs
        if any(fname.endswith(".doc") for fname in os.listdir(self.docs_path)):
            doc_loader = DirectoryLoader(
                path=self.docs_path, glob="*.doc", loader_cls=Docx2txtLoader
            )
            loaders.append(doc_loader)

        documents = []
        for loader in loaders:
            documents.extend(loader.load())

        return documents

    def create_vector_store(self, name: str) -> FAISS:
        documents = self.files_to_texts()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        texts = text_splitter.split_documents(
            documents
        )  # Asegúrate de usar el método correcto
        vector_store: FAISS = FAISS.from_documents(
            documents=texts, embedding=self.embeddings
        )
        # Guardar el vector store en la carpeta database
        dir_path = os.path.join(self.database_path, name)
        vector_store.save_local(folder_path=dir_path)
        return vector_store

    def load_vector_store(self, name: str) -> FAISS:
        dir_path = os.path.join(self.database_path, name)
        vector_store = FAISS.load_local(
            folder_path=dir_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return vector_store

    def update_vector_store(self, name: str) -> FAISS:
        vector_store = self.load_vector_store(name)  # Cargar el vector store existente
        documents = self.files_to_texts()  # Cargar nuevos documentos
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        texts = text_splitter.split_documents(documents)  # Cambiar a split_documents
        vector_store.add_documents(texts)  # Agregar nuevos documentos al vector store
        return vector_store

    def search_similarity(
        self, name: str, query: str, k: int = 3, source: Optional[str] = None
    ) -> list:
        vector_store = self.load_vector_store(name)
        if source:
            filter = {"source": source}
            retriever = vector_store.similarity_search(query=query, k=k, filter=filter)
        else:
            retriever = vector_store.similarity_search(query=query, k=k)
        return retriever

    def delete_vector_store(self, name: str):
        try:
            shutil.rmtree(os.path.join(self.database_path, name))
        except FileNotFoundError:
            return False
        return True

    def list_vector_stores(self):
        return os.listdir(self.database_path)

    # buscar si existe una base de datos con un nombre específico
    def database_exists(self, database_name: str):
        list_databases = self.list_vector_stores()
        if database_name in list_databases:
            return {
                "exists": True,
                "message": f"La base de datos '{database_name}' existe.",
            }
        else:
            return {
                "exists": False,
                "message": f"La base de datos '{database_name}' no existe.",
            }

    def upload_files_and_create_db(self, name: str, files: list):
        # Cargar archivos
        for file in files:
            with open(os.path.join(self.docs_path, file.filename), "wb") as f:
                f.write(file.file.read())  # Cambia según tu método de carga

        # Crear la base de datos
        return self.create_vector_store(name)
