import os
import asyncio
import rag_core as core

from langchain_community.document_loaders import DirectoryLoader

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import default_query_distribution
from ragas.testset.persona import (
    Persona,
)

# --- CONFIGURATION ---
DOCUMENTS_SOURCE_DIR = os.path.join(
    os.getcwd(), "dataset_rag_lol_definitive", "knowledge_base"
)
OUTPUT_DIR = os.path.join(os.getcwd(), "dataset_rag_lol_definitive")
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "synthetic_evaluation.csv")
TESTSET_SIZE = 10


async def main():
    """
    Script principal pour générer le jeu de données d'évaluation synthétique en français.
    """
    print("=" * 60)
    print("--- Générateur de Jeu de Test Synthétique Ragas (Français) ---")
    print("=" * 60)

    # --- Étape 1: Chargement des documents source ---
    print(f"\n1. Chargement des documents depuis '{DOCUMENTS_SOURCE_DIR}'...")
    if not os.path.exists(DOCUMENTS_SOURCE_DIR):
        print(f"[ERREUR] Le dossier source '{DOCUMENTS_SOURCE_DIR}' est introuvable.")
        print("Veuillez d'abord exécuter le script du data scrapper.")
        return

    loader = DirectoryLoader(DOCUMENTS_SOURCE_DIR, glob="**/*.txt")
    docs = loader.load()
    print(f" -> {len(docs)} documents chargés.")

    # --- Étape 2: Initialisation des modèles LLM et Embedding ---
    print("\n2. Initialisation des modèles OpenAI via rag_core...")
    llm = core.get_llm_openai()
    embedding_model = core.get_embedding_model_openai()

    generator_llm = LangchainLLMWrapper(llm)
    generator_embeddings = LangchainEmbeddingsWrapper(embedding_model)
    print(" -> Modèles prêts pour Ragas.")

    print("\n3. Définition des personas personnalisés...")
    personas = [
        Persona(
            name="historien_du_lore",
            role_description="Un historien méticuleux qui cherche à comprendre les liens profonds et les chronologies entre les différents champions et régions de Runeterra.",
        ),
        Persona(
            name="nouveau_joueur",
            role_description="Un nouveau joueur curieux qui pose des questions simples et directes pour comprendre les bases d'un personnage ou d'une région.",
        ),
        Persona(
            name="fan_de_faction",
            role_description="Un fan passionné d'une faction spécifique (comme Demacia ou Noxus) qui pose des questions comparatives et cherche des détails sur les motivations des personnages de cette faction.",
        ),
    ]
    print(f" -> {len(personas)} personas définis.")

    # --- Étape 4: Adaptation des générateurs de questions en Français ---
    print("\n4. Adaptation des prompts des synthétiseurs en français...")

    distribution = default_query_distribution(generator_llm)

    for synthesizer, _ in distribution:
        try:
            prompts = await synthesizer.adapt_prompts("french", llm=generator_llm)
            synthesizer.set_prompts(**prompts)
            print(f" -> {synthesizer.__class__.__name__} adapté avec succès.")
        except Exception as e:
            print(
                f" [AVERTISSEMENT] Échec de l'adaptation pour {synthesizer.__class__.__name__}: {e}"
            )
            print(" -> Utilisation des prompts par défaut (anglais).")

    # --- Étape 5: Génération du jeu de données de test ---
    print(
        f"\n5. Lancement de la génération du jeu de test ({TESTSET_SIZE} questions)..."
    )

    generator = TestsetGenerator(
        llm=generator_llm, embedding_model=generator_embeddings, persona_list=personas
    )

    dataset = generator.generate_with_langchain_docs(
        documents=docs,
        testset_size=TESTSET_SIZE,
        query_distribution=distribution,
    )
    print(" -> Génération terminée.")

    # --- Étape 6: Formatage et sauvegarde du résultat ---
    print("\n6. Formatage et sauvegarde du jeu de données...")
    df = dataset.to_pandas()

    question_col = "user_input" if "user_input" in df.columns else "question"
    ground_truth_col = "reference" if "reference" in df.columns else "ground_truth"

    if question_col in df.columns and ground_truth_col in df.columns:
        output_df = df[[question_col, ground_truth_col]]

        output_df = output_df.rename(
            columns={question_col: "question", ground_truth_col: "ground_truth"}
        )

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        output_df.to_csv(OUTPUT_FILENAME, index=False, encoding="utf-8")
        print(
            f"\nJeu de données de test sauvegardé avec succès dans : '{os.path.abspath(OUTPUT_FILENAME)}'"
        )
    else:
        print(
            "[ERREUR] Impossible de trouver les colonnes de question et de réponse de référence."
        )
        print(
            f"Colonnes attendues (nouveau format): '{question_col}', '{ground_truth_col}'"
        )

    print("\n" + "=" * 60)
    print("--- OPÉRATION TERMINÉE ---")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
