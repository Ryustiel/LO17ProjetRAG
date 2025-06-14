# rag_core.py

from typing import List
import pydantic
import os
import dotenv
import chromadb.utils.embedding_functions
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

dotenv.load_dotenv()

# --- Initialisation unique des modèles et clients ---


def get_embedding_model():
    """Crée et retourne une instance du modèle d'embedding LangChain."""
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


def get_chroma_client():
    """Crée et retourne une instance du client ChromaDB."""
    return chromadb.PersistentClient(
        path=os.path.join(os.getcwd(), "database", "chroma_db")
    )


def get_llm():
    """Crée et retourne une instance du LLM."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-05-20",
        temperature=0.7,
    )


client = get_chroma_client()
chroma_embedding_function = (
    chromadb.utils.embedding_functions.ChromaLangchainEmbeddingFunction(
        embedding_function=get_embedding_model()
    )
)

documents_collection = client.get_or_create_collection(
    name="documents", embedding_function=chroma_embedding_function
)
titles_collection = client.get_or_create_collection(
    name="titles", embedding_function=chroma_embedding_function
)


# Modèle de document
class Document(pydantic.BaseModel):
    id: str
    rating: float
    title: str
    content: str

    def __hash__(self):
        return hash(self.id)


# Fonction de retrieval
def query(q: str, n_results: int) -> List[Document]:
    document_results = documents_collection.query(
        query_texts=[q],
        n_results=n_results,
    )
    if not document_results or not document_results["ids"][0]:
        return []

    ids = document_results["ids"][0]
    titles_results = titles_collection.get(ids=ids)

    return [
        Document(
            id=ids[i],
            rating=round(document_results["distances"][0][i], 2),
            title=titles_results["documents"][i],
            content=document_results["documents"][0][i],
        )
        for i in range(len(ids))
    ]
