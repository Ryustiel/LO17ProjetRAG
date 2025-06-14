"""
Construit la base de données vectorielle à partir des fichiers de lore
générés par le scrapper et contenus dans dataset_rag_lol_definitive/knowledge_base.
"""

from typing import List, Tuple
import os

import chromadb

import rag_core as core

# --- CONFIGURATION ---
DOCUMENTS_SOURCE_DIR = os.path.join(
    os.getcwd(), "dataset_rag_lol_definitive", "knowledge_base"
)

documents_collection = core.documents_collection
titles_collection = core.titles_collection

print("Base de données vectorielle initialisée.")


# --- REMISE À ZÉRO DE LA BASE DE DONNÉES ---
def reset_collection(collection: chromadb.Collection):
    """Vide une collection de tous ses documents."""
    count = collection.count()
    if count > 0:
        # Récupère tous les IDs existants pour les supprimer
        existing_ids = collection.get(include=[])["ids"]
        collection.delete(ids=existing_ids)
        print(
            f"Collection '{collection.name}' remise à zéro ({count} éléments supprimés)."
        )


reset_collection(documents_collection)
reset_collection(titles_collection)

# --- IMPORT ET INDEXATION DES FICHIERS ---


def load_documents_from_source() -> Tuple[List[str], List[str]]:
    """
    Charge les contenus des documents et génère des IDs à partir des noms de fichiers.
    Retourne:
        - Une liste de contenus textuels.
        - Une liste d'IDs (noms de fichiers sans extension).
    """
    contents = []
    ids = []

    if not os.path.exists(DOCUMENTS_SOURCE_DIR):
        print(f"[ERREUR] Le dossier source '{DOCUMENTS_SOURCE_DIR}' n'existe pas.")
        print("Veuillez d'abord exécuter le script du data scrapper.")
        return [], []

    # Le scrapper ne crée que des .txt
    for filename in os.listdir(DOCUMENTS_SOURCE_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(DOCUMENTS_SOURCE_DIR, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                contents.append(file.read())
                # L'ID est le nom du fichier sans ".txt" (ex: "garen", "jinx")
                ids.append(os.path.splitext(filename)[0])

    return contents, ids


print("\nChargement des documents à indexer...")
docs, doc_ids = load_documents_from_source()

if docs:
    print(f"{len(docs)} documents chargés.")

    print("Indexation des contenus dans la collection 'documents'...")
    documents_collection.add(documents=docs, ids=doc_ids)

    print("Indexation des titres (slugs) dans la collection 'titles'...")
    titles_collection.add(documents=doc_ids, ids=doc_ids)

    print("\nOpération terminée.")
    print(
        f"La base de données a été construite avec succès dans : 'database/chroma_db'"
    )
else:
    print("Aucun document n'a été trouvé à indexer.")
