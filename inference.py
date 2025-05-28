"""
Définit les différentes fonctions d'inférence pour le frontend streamlit.
"""

from typing import List, Iterable
import pydantic, os, dotenv, chromadb, chromadb.utils.embedding_functions
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

dotenv.load_dotenv()

# Initialisation de la base de données vectorielle

chroma_compatible_google_ef = chromadb.utils.embedding_functions.ChromaLangchainEmbeddingFunction(
    embedding_function = GoogleGenerativeAIEmbeddings(
        model = "models/text-embedding-004"
    )
)

client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "database", "chroma_db"))

documents = client.get_or_create_collection(
    name="documents",
    embedding_function=chroma_compatible_google_ef
)
titles = client.get_or_create_collection(
    name="titles",
    embedding_function=chroma_compatible_google_ef
)

# Modèle de document

class Document(pydantic.BaseModel):
    id: str
    rating: float
    title: str
    content: str

# Définition de la fonction d'inférence (retrieval)

def query(q: str, n_results: int) -> List[Document]:
    document_results = documents.query(
        query_texts=[q],
        n_results=n_results,
    )
    if not document_results:
        return []
    
    ids = document_results["ids"][0]
    titles_results = titles.get(ids=ids)
    return [
        Document(
            id = ids[i],
            rating = round(document_results["distances"][0][i], 2),
            title = titles_results["documents"][i],
            content = document_results["documents"][0][i]
        ) for i in range(len(ids))
    ]

# Définition de la fonction d'assistance LLM

def llm_summary(q: str, documents: List[Document]) -> Iterable[str]:
    it = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", 
        temperature=0.7
    ).stream(
        "Summarize the following documents in a single markdown setup. Use markdown freely. "
        + "Use the language of the user query (usually french), not the language of the documents. "
        + "Only output the summary and nothing else. "
        + f"\nOriginal user query: {q}"
        + "\nDocuments:\n"
        + "\n".join([f"{doc.title}\n{doc.content}" for doc in documents])
    )
    for tok in it:
        yield tok.content
