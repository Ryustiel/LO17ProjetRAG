"""
Script d'évaluation du RAG en utilisant les métriques de Faithfulness et Correctness.
Ce script se base sur le fichier 'synthetic_evaluation.csv' généré par 'generate_testset.py'.
"""

import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, AnswerCorrectness
from langchain_core.messages import SystemMessage, HumanMessage
from tqdm import tqdm

from rag_core import get_llm, get_embedding_model, query


def generate_rag_answers(eval_df: pd.DataFrame, llm) -> list:
    """
    Génère les réponses et récupère les contextes pour tout le dataframe.
    """
    results_list = []
    print("\n--- Étape 1: Génération des réponses pour le jeu de données ---")
    for index, row in tqdm(
        eval_df.iterrows(), total=eval_df.shape[0], desc="Génération des réponses"
    ):
        question = row["question"]
        ground_truth = row["ground_truth"]

        # 1. Retrieval
        retrieved_contexts_docs = query(question, n_results=5)
        retrieved_contexts_str = [doc.content for doc in retrieved_contexts_docs]

        # 2. Augmentation & Generation
        generation_prompt = [
            SystemMessage(
                content=(
                    "Tu es un expert du lore de Runeterra. Réponds directement à la question de l'utilisateur."
                    "Ta réponse doit être une **synthèse concise** des informations les plus pertinentes trouvées dans les documents fournis. "
                    "Ne mentionne pas les documents sources. Va droit au but tout en étant précis."
                    "Ta réponse doit être entièrement basée sur les faits présents dans les documents suivants.\n\n"
                    "Documents:\n" + "\n\n---\n\n".join(retrieved_contexts_str)
                )
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
    # --- Initialisation des modèles dans la fonction main ---
    print("Initialisation des modèles et de l'évaluateur...")
    llm = get_llm()
    embedding_model = get_embedding_model()

    ragas_llm = LangchainLLMWrapper(llm)

    # Initialisation des métriques
    faithfulness_evaluator = Faithfulness(llm=ragas_llm)
    print("Métrique 'Faithfulness' initialisée.")
    correctness_evaluator = AnswerCorrectness(llm=ragas_llm, weights=[1, 0])
    print("Métrique 'AnswerCorrectness' initialisée.")

    evaluation_file_path = os.path.join(
        "dataset_rag_lol_definitive", "synthetic_evaluation.csv"
    )
    if not os.path.exists(evaluation_file_path):
        print(f"\n[ERREUR] Le fichier '{evaluation_file_path}' n'a pas été trouvé.")
        print("Veuillez d'abord exécuter 'python generate_testset.py' pour le créer.")
        return

    print(f"Chargement du jeu de données depuis '{evaluation_file_path}'...")
    eval_df = pd.read_csv(evaluation_file_path)

    # Étape 1: Générer toutes les réponses et contextes en premier
    evaluation_data = generate_rag_answers(eval_df, llm)

    # Étape 2: Convertir la liste en format Dataset pour Ragas
    # Ragas s'attend à trouver les clés 'question', 'answer', 'contexts', 'ground_truth'
    evaluation_dataset = Dataset.from_list(evaluation_data)

    print("\n--- Étape 2: Évaluation par lot avec Ragas ---")

    # 3. Lancer l'évaluation en une seule fois
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[
            faithfulness_evaluator,
            correctness_evaluator,
        ],
        embeddings=embedding_model,
        run_config=RunConfig(max_workers=1),
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
