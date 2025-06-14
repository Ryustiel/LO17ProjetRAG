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


llm = rag_core.get_llm()
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
                content="Créez une ou plusieurs requêtes de recherche pour le système RAG en fonction de l'historique de la conversation. "
                + "Si l'utilisateur n'a pas encore dit ce qu'il recherche, retournez une liste vide. "
                + "#1. Créez une requête pour chaque type d'information différent que l'utilisateur recherche."
                + "\nPar exemple, si l'utilisateur demande un document sur un certain sujet, "
                + "créez une requête qui contient de nombreux mots-clés liés à ce sujet afin que le RAG trouve toutes les correspondances pertinentes. "
                + '\n#2. Les requêtes ne sont pas destinées à être lues par les utilisateurs ou l\'IA ; elles seront uniquement vectorisées (ou "embeddées") et comparées via la similarité cosinus. '
                + "Cela signifie qu'elles doivent être courtes et basées sur des mots-clés, comme un tissage de mots suggérant ce qui intéresse l'utilisateur. "
                + "(ex. \"révolution française napoléon france. manifestations france histoire française\" si l'utilisateur semble s'intéresser à la révolution française). "
                + "Vous pouvez ajouter des mots-clés personnalisés que l'utilisateur n'a pas mentionnés pour améliorer la requête."
                + "\n#3. En tant que requête RAG, traduisez les contraintes par l'omission ou la modification de certains mots-clés dans la requête, "
                + "et/ou ajoutez des mots liés à des sujets perpendiculaires, afin d'influencer les résultats du RAG. "
                + "(ex. Si l'utilisateur ne veut pas entendre parler des aspects politiques de la révolution française, "
                + 'vous pouvez modifier la requête pour ne pas inclure "politique" ou "gouvernement", '
                + 'et y ajouter des mots comme "culture", "art", "philosophie" pour influencer les résultats.)'
            )
        ]
        + conversation
    )
    docs = set()
    for q in queries_response.queries:
        yield q.query
        docs.update(rag_core.query(q=q.query, n_results=q.n_results(max_results)))
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
