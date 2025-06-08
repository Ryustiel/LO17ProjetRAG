"""
Définit les différentes fonctions d'inférence pour le frontend streamlit.
"""

from typing import List, Literal, Iterator
import pydantic, os, dotenv, chromadb.utils.embedding_functions
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    AIMessageChunk,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import streamlit as st

dotenv.load_dotenv()

# Initialisation de la base de données vectorielle


@st.cache_resource
def get_embedding_function():
    """Crée et retourne une instance unique de la fonction d'embedding."""
    return chromadb.utils.embedding_functions.ChromaLangchainEmbeddingFunction(
        embedding_function=GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )
    )


@st.cache_resource
def get_chroma_client():
    """Crée et retourne une instance unique du client ChromaDB."""
    return chromadb.PersistentClient(
        path=os.path.join(os.getcwd(), "database", "chroma_db")
    )


@st.cache_resource
def get_llm():
    """Crée et retourne une instance unique du LLM."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-05-20", temperature=0.7
    )


client = get_chroma_client()
chroma_compatible_google_ef = get_embedding_function()
llm = get_llm()  # Vous utiliserez cette instance dans vos fonctions de chat/summary

documents = client.get_or_create_collection(
    name="documents", embedding_function=chroma_compatible_google_ef
)
titles = client.get_or_create_collection(
    name="titles", embedding_function=chroma_compatible_google_ef
)

# Modèle de document


class Document(pydantic.BaseModel):
    id: str
    rating: float
    title: str
    content: str

    def __hash__(self):
        return hash(self.id)


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
            id=ids[i],
            rating=round(document_results["distances"][0][i], 2),
            title=titles_results["documents"][i],
            content=document_results["documents"][0][i],
        )
        for i in range(len(ids))
    ]


# Définition de la fonction d'assistance LLM


def llm_summary(q: str, documents: List[Document]) -> Iterator[str]:
    it = llm.stream(
        "Summarize the following documents in a single markdown setup. Use markdown freely. "
        + "Use the language of the user query (usually french), not the language of the documents. "
        + "Only output the summary and nothing else. "
        + f"\nOriginal user query: {q}"
        + "\nDocuments:\n"
        + "\n".join([f"{doc.title}\n{doc.content}" for doc in documents])
    )
    for tok in it:
        yield tok.content


# Chain of thoughts : production d'une query adaptée à la recherche au RAG


class SearchQuery(pydantic.BaseModel):
    query: str = pydantic.Field("", description="La requête envoyée au système de RAG.")
    result_expectation: Literal["one match", "few matches", "all relevant"] = (
        pydantic.Field(
            ..., description="Indication du nombre de résultats attendus de la requête."
        )
    )

    def n_results(self, max_results: int) -> int:
        match self.result_expectation:
            case "one match":
                return 2
            case "few matches":
                return 3
            case "all relevant":
                return max_results


class SearchQueryResponse(pydantic.BaseModel):
    queries: List[SearchQuery]


def query_from_conversation(
    conversation: List[BaseMessage], max_results: int
) -> Iterator[str | List[Document]]:
    queries_response: SearchQueryResponse = llm.with_structured_output(
        SearchQueryResponse
    ).invoke(
        [
            SystemMessage(
                content="Create one or many search queries for the RAG system based on the conversation history. "
                + "If the user did not say what they are looking for yet, output an empty list. "
                + "#1. Create one query for each different information type that the user is looking for."
                + "\nFor example, if the user is asking for a document about a certain topic, "
                + "create a query that has many keywords related to that topic and expect all relevant, so that the RAG matches. "
                + "\n#2. The queries are not meant to be read by users or ai, they will only be embedded and compared via cosine similarity. "
                + "This means that they should be short, keyword-based like a weave of words hinting at what's interesting the user. "
                + '(ex. "french revolution napoleon france. french protests french history" if the user seems interested in the french revolution). '
                + "You can add in custom keywords the user did not mention to improve the query."
                + "\n#3. Being a RAG query, translate constraints by omission or twisting of some keywords in the query, "
                + "and/or add words that are related to perpendicular topics, in order to influence the RAG results. "
                + "(ex. If the user does not want to hear about the political aspects of the french revolution, "
                + 'you can twist the query to not include "politics" or "government" in the query, '
                + 'and add in words like "culture", "art", "philosophy" to influence the results.)'
            )
        ]
        + conversation
    )
    docs = set()
    for q in queries_response.queries:
        yield q.query
        docs.update(query(q=q.query, n_results=q.n_results(max_results)))
        print(len(docs))
    yield list(docs)


# Définition de la fonction de chat via LLM


def chat(
    conversation: List[BaseMessage],
) -> Iterator[str | List[Document] | AIMessageChunk]:

    documents = []
    for item in query_from_conversation(conversation, max_results=5):
        if isinstance(item, str):
            yield item
            continue

        elif isinstance(item, list):
            documents = item
            yield documents
            break

    it: Iterator[AIMessageChunk] = llm.stream(
        [
            SystemMessage(
                content="Tu es un assistant de recherche qui présente aux utilisateurs les documents de la base de données fournis par le système de RAG. "
                + "L'utilisateur formule des requêtes qui dictent les documents trouvés."
                + "Tes réponses peuvent être de deux natures : 1. présenter les résultats de recherche, "
                + "2. répondre à des questions de l'utilisateur en utilisant les documents trouvés."
                + '\nSi l\'utilisateur formule une requête vague comme "je cherche des infos sur X", '
                + "présente lui simplement les résultats de recherche et comment ils pourraient être en rapport avec sa requête."
                + "Tu dois répondre en français, et utiliser les documents pour répondre aux questions de l'utilisateur. "
                + "Tu peux utiliser des citations des documents pour appuyer tes réponses."
                + "\nQuery Results:\n"
                + "\n".join([f"{doc.title}\n{doc.content}" for doc in documents])
            ),
        ]
        + conversation
    )
    for chunk in it:
        yield chunk
