"""
Script d'évaluation du RAG en utilisant les métriques de Faithfulness et Correctness.
Ce script se base sur le fichier 'evaluation.csv' généré par 'data_scrapper.py'.
"""

import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, AnswerCorrectness
from langchain_core.messages import SystemMessage, HumanMessage
from tqdm import tqdm

import rag_core as core

print("Initialisation des modèles et de l'évaluateur...")

llm = core.llm

# Wrapper Ragas pour le LLM
ragas_llm = LangchainLLMWrapper(llm)
embedding_model_for_ragas = core.embedding_model

# Initialisation des métriques comme dans les slides
# Faithfulness: vérifie que la réponse est basée sur le contexte
faithfulness_evaluator = Faithfulness(llm=ragas_llm)
print("Métrique 'Faithfulness' initialisée.")

# AnswerCorrectness: vérifie la justesse de la réponse par rapport à une référence
correctness_evaluator = AnswerCorrectness(llm=ragas_llm, weights=[1, 0])
print("Métrique 'AnswerCorrectness' initialisée.")


def generate_rag_answers(eval_df: pd.DataFrame) -> list:
    """
    Génère les réponses et récupère les contextes pour tout le dataframe.
    """
    results_list = []
    print("\n--- Étape 1: Génération des réponses pour le jeu de données ---")
    for index, row in tqdm(
        eval_df.iterrows(), total=eval_df.shape[0], desc="Génération des réponses"
    ):
        question = row["question"]
        ground_truth = row["reponse_attendue"]

        # 1. Retrieval
        retrieved_contexts_docs = core.query(question, n_results=5)
        retrieved_contexts_str = [doc.content for doc in retrieved_contexts_docs]

        # 2. Augmentation & Generation
        generation_prompt = [
            SystemMessage(
                content="Réponds à la question de l'utilisateur en te basant UNIQUEMENT sur les documents fournis. Sois concis et factuel."
                + "\n\nDocuments:\n"
                + "\n\n---\n\n".join(retrieved_contexts_str)
            ),
            HumanMessage(content=question),
        ]
        generated_response = llm.invoke(generation_prompt).content

        results_list.append(
            {
                "question": question,
                "answer": generated_response,
                "contexts": retrieved_contexts_str,
                "ground_truth": ground_truth,
            }
        )
    return results_list


def main():
    """
    Script principal pour lancer l'évaluation par lot.
    """
    evaluation_file_path = os.path.join("dataset_rag_lol_definitive", "evaluation.csv")
    if not os.path.exists(evaluation_file_path):
        print(f"\n[ERREUR] Le fichier '{evaluation_file_path}' n'a pas été trouvé.")
        return

    print(f"Chargement du jeu de données depuis '{evaluation_file_path}'...")
    eval_df = pd.read_csv(evaluation_file_path)

    # Étape 1: Générer toutes les réponses et contextes en premier
    evaluation_data = generate_rag_answers(eval_df)

    # Étape 2: Convertir la liste en format Dataset pour Ragas
    evaluation_dataset = Dataset.from_list(evaluation_data)

    print("\n--- Étape 2: Évaluation par lot avec Ragas ---")
    run_config = RunConfig(max_workers=1)

    # 3. Lancer l'évaluation en une seule fois
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[
            faithfulness_evaluator,
            correctness_evaluator,
        ],
        embeddings=embedding_model_for_ragas,
        run_config=run_config,
    )

    print("\n\n--- Fin de l'évaluation ---")
    results_df = result.to_pandas()
    print(results_df)

    if not results_df.empty:
        faithfulness_score = results_df["faithfulness"].mean()
        correctness_score = results_df["answer_correctness"].mean()

        print(f"\n**Score moyen de Faithfulness : {faithfulness_score:.4f}**")
        print(f"**Score moyen de Correctness  : {correctness_score:.4f}**")

        # Sauvegarder les résultats détaillés dans un fichier CSV
        results_df.to_csv("evaluation_results.csv", index=False)
        print(
            "\nLes résultats détaillés ont été sauvegardés dans 'evaluation_results.csv'"
        )


if __name__ == "__main__":
    main()
