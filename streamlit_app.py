import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, AIMessageChunk

# Configuration de la page
st.set_page_config(page_title="Chroniqueur de Runeterra", page_icon="📜", layout="wide")

import inference
import rag_core as core

# Initialisation des états de session
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        AIMessage(
            content="Bonjour ! Comment puis-je vous aider à explorer les documents ?"
        )
    ]
if "generating" not in st.session_state:
    st.session_state.generating = False

# Titre de l'application
st.title("📜 Chroniqueur de Runeterra")
st.caption(
    "Interrogez les archives de League of Legends. Posez vos questions sur les champions et les régions."
)

# Création des onglets
tab_chat, tab_search = st.tabs(["💬 Chat avec l'assistant", "🔍 Recherche Directe"])

# --- Onglet Chatbot ---
with tab_chat:
    st.markdown("## ⚔️ Discutez avec vos documents")
    st.markdown("---")
    st.write(
        "Posez vos questions en langage naturel. L'assistant utilisera les documents de la base pour vous répondre."
    )

    # Affichage de l'historique des messages
    for msg in st.session_state.chat_messages:
        role = "assistant" if isinstance(msg, AIMessage) else "user"
        with st.chat_message(role):
            st.markdown(msg.content)
            if isinstance(msg, AIMessage) and "sources" in msg.additional_kwargs:
                if msg.additional_kwargs["sources"]:
                    with st.expander("Sources consultées"):
                        for doc in msg.additional_kwargs["sources"]:
                            st.markdown(f"**- {doc.title}** (Score: {doc.rating})")
                            st.markdown("---")

    # Si une génération est en cours, on exécute la logique de streaming
    if st.session_state.generating:
        with st.chat_message("assistant"):
            source_expander_placeholder = st.empty()
            with st.status("Réflexion en cours...", expanded=False) as status:
                conversation_history = [
                    msg
                    for msg in st.session_state.chat_messages
                    if isinstance(msg, (HumanMessage, AIMessage))
                ]
                response_generator = inference.chat(conversation_history)
                sources_for_storage = []

                def stream_handler(generator):
                    is_generating_answer = False
                    for chunk in generator:
                        if isinstance(chunk, str):
                            status.update(label=f"Recherche avec la query : `{chunk}`")
                        elif isinstance(chunk, list):
                            sorted_chunk = sorted(
                                chunk, key=lambda doc: doc.rating, reverse=True
                            )
                            sources_for_storage[:] = sorted_chunk
                            with source_expander_placeholder.expander(
                                "Sources utilisées pour la réponse"
                            ):
                                for documents in sorted_chunk:
                                    st.markdown(
                                        f"**- {documents.title}** (Score: {documents.rating})"
                                    )
                                    st.markdown("---")
                        elif isinstance(chunk, AIMessageChunk):
                            if not is_generating_answer:
                                status.update(
                                    label="Documents trouvés. Génération de la réponse...",
                                    state="running",
                                    expanded=True,
                                )
                                is_generating_answer = True
                            yield chunk.content

                full_response = st.write_stream(stream_handler(response_generator))

                status.update(label="Terminé !", state="complete", expanded=False)

            if full_response:
                new_message = AIMessage(
                    content=full_response,
                    additional_kwargs={"sources": sources_for_storage},
                )
                st.session_state.chat_messages.append(new_message)

            st.session_state.generating = False
            st.rerun()

    # Champ de saisie du chat, désactivé pendant la génération
    if prompt := st.chat_input(
        "Votre question...", disabled=st.session_state.generating
    ):
        st.session_state.chat_messages.append(HumanMessage(content=prompt))  # type: ignore
        st.session_state.generating = True
        st.rerun()

# --- Onglet Recherche Directe ---
with tab_search:
    st.header("Recherche directe dans la base de documents")
    st.write(
        "Entrez des mots-clés pour trouver les documents les plus pertinents. Appuyez sur 'Entrée' pour valider."
    )

    with st.form(key="search_form"):
        search_query = st.text_input(
            "Votre requête de recherche :",
            placeholder="ex: linguistique, histoire de france, napoléon...",
        )
        n_results = st.slider(
            "Nombre de résultats à retourner :", min_value=1, max_value=10, value=3
        )
        submitted = st.form_submit_button("Rechercher")

    if submitted:
        if not search_query:
            st.warning("Veuillez entrer une requête de recherche.")
        else:
            with st.spinner("Recherche en cours..."):
                results = core.query(search_query, n_results)

            if not results:
                st.info("Aucun document trouvé pour cette requête.")
            else:
                st.success(f"{len(results)} document(s) trouvé(s) :")
                for doc in results:
                    with st.expander(f"**{doc.title}**"):
                        st.markdown(
                            f"**Score de similarité (distance) :** `{doc.rating}`"
                        )
                        st.markdown("---")
                        st.markdown(doc.content)
