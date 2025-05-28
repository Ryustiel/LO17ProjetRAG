"""
Construit la base de données vectorielle à partir des fichiers divers contenus dans database/documents.
"""

from typing import List
import os, dotenv, PyPDF2, chromadb, chromadb.utils.embedding_functions
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

print("Base de données vectorielle initialisée.")

# Remise à zéro de la base de données vectorielle

if documents.count() > 0:
    documents.delete([f"doc_{i+1}" for i in range(documents.count())])
    
if titles.count() > 0:
    titles.delete([f"doc_{i+1}" for i in range(titles.count())])
    
print("Base de données vectorielle remise à zéro.")

# Import des fichiers à indexer

def load_documents() -> List[str]:
    documents = []
    documents_dir = os.path.join(os.getcwd(), "database", "documents")
    for filename in os.listdir(documents_dir):
        if filename.endswith(".txt") or filename.endswith(".md") or filename.endswith(".html"):
            with open(os.path.join(documents_dir, filename), "r", encoding="utf-8") as file:
                documents.append(file.read())
        elif filename.endswith(".pdf"):
            with open(os.path.join(documents_dir, filename), "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                documents.append(text)
    return documents

# Indexation des fichiers dans la base de données vectorielle

print("Chargement des documents à indexer...")

docs = load_documents()

print(f"{len(docs)} documents chargés.")
print("Documents indexés dans la base de données vectorielle.")
    
generate_title = lambda doc: ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", 
        temperature=0.7
    ).invoke(
        "Generate a title for the following document in the language of said document. "
        + f"Only output the title and nothing else. Document:\n\n{doc}"
    ).content

print("Génération des titres pour les documents...")

documents.add(documents=docs, ids=[f"doc_{i+1}" for i in range(len(docs))])
titles.add(documents=[generate_title(doc) for doc in docs], ids=[f"doc_{i+1}" for i in range(len(docs))])

print("Titres générés et indexés dans la base de données vectorielle.")
