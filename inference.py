"""
Définit les différentes fonctions d'inférence pour le frontend streamlit.
"""

from typing import List, Literal, Iterator
import pydantic
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    AIMessageChunk,
)

import rag_core


llm = rag_core.llm
Document = rag_core.Document


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

    def n_results(self, max_results: int) -> int | None:
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
        docs.update(rag_core.query(q=q.query, n_results=q.n_results(max_results)))
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
                content="Salutations, voyageur ! Je suis un chroniqueur de Runeterra, gardien des récits de champions et des légendes des régions. Mon savoir provient des écrits que le système m'a fournis. "
                + "Pose-moi tes questions sur les héros, les terres lointaines ou les rivalités qui façonnent ce monde. "
                + "Ma mission est de te répondre en me basant fidèlement sur ces chroniques. "
                + "Si ta question est précise, je te donnerai une réponse directe, puisant dans les textes. "
                + "Si elle est plus vague, je te présenterai les parchemins qui me semblent les plus pertinents pour ta quête de connaissance. "
                + "Dans tous les cas, je te répondrais uniquement à l'aide des documents fournis par le système, pas de mes connaissances personnelles. "
                + "Je réponds toujours en français. N'hésite pas à citer des passages des documents pour appuyer tes réponses, si cela éclaire ton propos."
                + "\nQuery Results:\n"
                + "\n".join([f"{doc.title}\n{doc.content}" for doc in documents])
            ),
        ]
        + conversation
    )
    for chunk in it:
        yield chunk
