# rag_core.py

from typing import List
import pydantic
import os
import dotenv
import chromadb.utils.embedding_functions
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

dotenv.load_dotenv()

# --- Initialisation unique des modèles et clients ---


def get_embedding_model():
    """Crée et retourne une instance du modèle d'embedding LangChain."""
    dotenv.load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", google_api_key=api_key
    )


def get_embedding_model_openai():
    """Crée et retourne une instance du modèle d'embedding OpenAI."""
    return OpenAIEmbeddings(model="text-embedding-3-small")


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


def get_llm_openai():
    """Crée et retourne une instance du LLM OpenAI."""
    return ChatOpenAI(
        model="gpt-4.1",
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
    if (
        not document_results
        or not document_results["ids"]
        or not document_results["ids"][0]
    ):
        return []

    doc_ids_ordered = document_results["ids"][0]
    doc_distances_ordered = document_results["distances"][0]
    doc_contents_ordered = document_results["documents"][0]

    titles_data = titles_collection.get(ids=doc_ids_ordered)

    title_map = {}
    if titles_data and titles_data["ids"] and titles_data["documents"]:
        for i in range(len(titles_data["ids"])):
            title_map[titles_data["ids"][i]] = titles_data["documents"][i]

    final_documents = []
    for i in range(len(doc_ids_ordered)):
        current_id = doc_ids_ordered[i]
        current_title = title_map.get(current_id, "Titre non disponible")
        final_documents.append(
            Document(
                id=current_id,
                rating=round(doc_distances_ordered[i], 2),
                title=current_title,
                content=doc_contents_ordered[i],
            )
        )

    return final_documents
