# Chroniqueur de Runeterra : Un Agent RAG sur l'univers de League of Legends

Ce projet implémente un agent conversationnel avancé basé sur une architecture RAG (Retrieval-Augmented Generation). Le "Chroniqueur de Runeterra" est un chatbot spécialisé dans le lore de l'univers du jeu vidéo *League of Legends*. Il est capable de répondre à des questions précises sur les champions et les régions en se basant sur une base de connaissances construite à partir de sources officielles.

## Demo

Une version live de l'application est déployée ici : [**lo17.raphcvr.me**](https://lo17.raphcvr.me/)


## ✨ Fonctionnalités

  - **Interface de Chat intuitive** : Une application Streamlit permet aux utilisateurs de dialoguer en langage naturel avec l'assistant.
  - **Base de Connaissances Automatisée** : Un script de scraping (`data_scrapper.py`) collecte automatiquement le lore depuis des sources de confiance (wiki Fandom, League of Legends Universe).
  - **Base de Données Vectorielle** : Utilisation de **ChromaDB** pour stocker et rechercher efficacement les documents de lore grâce à des embeddings sémantiques.
  - **Transformation de Requête Avancée** : Le système utilise un LLM pour analyser la conversation et transformer la question de l'utilisateur en requêtes optimisées pour la recherche vectorielle, améliorant ainsi la pertinence des résultats.
  - **Transparence des Sources** : Pour chaque réponse, l'assistant cite les documents qu'il a utilisés, permettant à l'utilisateur de vérifier l'information à la source.
  - **Évaluation Rigoureuse** : Le projet inclut un framework d'évaluation (`evaluation.py`) utilisant la bibliothèque `ragas` pour mesurer la fidélité (`Faithfulness`) et la correction (`Correctness`) des réponses générées.
  - **Génération de Données de Test** : Un script (`generate_testset.py`) permet de créer un jeu de données d'évaluation synthétique de haute qualité pour valider la performance du RAG.
  - **Prêt pour le Déploiement** : L'application est entièrement conteneurisable et fournie avec une configuration Kubernetes (`k8s-lo17-rag-app.yaml`) pour un déploiement cloud-native.

## 🏛️ Architecture

Le workflow de l'application suit les étapes classiques d'un pipeline RAG moderne :

1.  **Interface Utilisateur (Streamlit)** : L'utilisateur saisit sa question dans l'interface de chat.
2.  **Transformation de la Requête (`inference.py`)** : Un premier appel au LLM (Google Gemini) analyse la question dans le contexte de la conversation et génère des requêtes de recherche sémantique optimisées.
3.  **Récupération d'Information (`rag_core.py`)** : Les requêtes optimisées sont utilisées pour interroger la base de données vectorielle ChromaDB. Les documents les plus pertinents sont récupérés.
4.  **Augmentation du Contexte** : Les documents récupérés sont injectés dans le contexte d'un nouveau prompt.
5.  **Génération de la Réponse (`inference.py`)** : Le prompt augmenté est envoyé au LLM, qui a pour instruction d'agir comme le "Chroniqueur de Runeterra" et de synthétiser une réponse en se basant *uniquement* sur les documents fournis.
6.  **Affichage en Streaming (Streamlit)** : La réponse est affichée en temps réel dans l'interface, et les sources sont listées dans un menu déroulant pour la transparence.

## 🗂️ Structure du Projet

```
.
├── .streamlit/
│   └── config.toml           # Thème et configuration de l'interface Streamlit
├── dataset_rag_lol_definitive/
│   ├── knowledge_base/       # (Généré) Contient les .txt du lore scrapé
│   └── synthetic_evaluation.csv # (Généré) Jeu de données pour l'évaluation
├── app.py                      # Script CLI simple pour tester le RAG
├── create_database.py          # Script pour construire la base de données ChromaDB
├── data_scrapper.py            # Script pour scraper le lore et créer la base de connaissance
├── evaluation.py               # Script pour évaluer le RAG avec Ragas
├── generate_testset.py         # Script pour générer le jeu de données d'évaluation
├── inference.py                # Logique d'inférence du chatbot
├── k8s-lo17-rag-app.yaml       # Fichier de déploiement Kubernetes
├── rag_core.py                 # Cœur du système RAG (connexion DB, query, modèles)
├── streamlit_app.py            # Application principale Streamlit
├── pyproject.toml              # Dépendances et configuration du projet
└── ...
```

## 🚀 Installation et Lancement Local

Suivez ces étapes pour lancer l'application sur votre machine.

### Prérequis

  - Python 3.12+
  - Un gestionnaire de paquets comme `pip` ou `uv`.

### 1\. Cloner le Dépôt

```bash
git clone https://github.com/Ryustel/LO17ProjetRAG.git
cd LO17ProjetRAG
```

### 2\. Installer les Dépendances

Il est recommandé d'utiliser un environnement virtuel à l'aide d'uv ([**Installation d'uv**](https://docs.astral.sh/uv/getting-started/installation/)

Installez les dépendances listées dans `pyproject.toml` :

```bash
uv sync
```

### 3\. Configurer les Clés d'API

Créez un fichier `.env` à la racine du projet et ajoutez vos clés d'API. Le projet utilise les modèles de Google pour le RAG et potentiellement OpenAI pour la génération du jeu de test.

```env
# .env
GOOGLE_API_KEY="VOTRE_CLE_API_GOOGLE"
OPENAI_API_KEY="VOTRE_CLE_API_OPENAI"
```

### 4\. Construire la Base de Connaissances

Ces scripts doivent être exécutés dans l'ordre.

```bash
# 1. Scraper les données du web
uv run data_scrapper.py

# 2. Construire la base de données vectorielle
uv run create_database.py
```

À la fin de cette étape, vous devriez avoir un dossier `database/chroma_db` peuplé.

### 5\. Lancer l'Application Streamlit

```bash
uv run streamlit run streamlit_app.py
```

L'application devrait être accessible à l'adresse `http://localhost:8501`.

## 📊 Évaluation du Système

Le projet est doté d'un système d'évaluation pour mesurer la qualité des réponses.

1.  **Générer le jeu de test (Optionnel)** :
    Le script `generate_testset.py` utilise un LLM (OpenAI par défaut) pour créer des questions/réponses pertinentes à partir de la base de connaissances.

    ```bash
    python generate_testset.py
    ```

    Cela créera le fichier `dataset_rag_lol_definitive/synthetic_evaluation.csv`.

2.  **Lancer l'évaluation** :
    Ce script utilise le fichier CSV généré pour évaluer le pipeline RAG sur les métriques de `faithfulness` et `answer_correctness`.

    ```bash
    python evaluation.py
    ```

    Les résultats détaillés seront sauvegardés dans `evaluation_results.csv`.

## 📦 Déploiement

Le fichier `k8s-lo17-rag-app.yaml` contient la configuration complète pour un déploiement sur un cluster Kubernetes.

Points clés :

  - **Init Container** : Un conteneur d'initialisation se charge d'exécuter `data_scrapper.py` et `create_database.py` au premier démarrage du pod.
  - **Persistance** : Un `PersistentVolumeClaim` est utilisé pour que la base de données ChromaDB ne soit pas reconstruite à chaque redémarrage du pod.
  - **Secrets** : Les clés d'API doivent être fournies au cluster via des secrets Kubernetes (`google-api-secret`, `openai-api-secret`).
  - **Ingress** : Une règle Ingress est définie pour exposer le service Streamlit sur le web, par exemple via le domaine `lo17.raphcvr.me`.

## 🛠️ Technologies Utilisées

  - **Frontend** : Streamlit
  - **Backend & Orchestration** : Python, LangChain
  - **LLMs** : Google Gemini, OpenAI (pour la création du jeu de test)
  - **Embeddings** : Google Generative AI Embeddings, OpenAI Embeddings (pour la création du jeu de test)
  - **Base de Données Vectorielle** : ChromaDB
  - **Scraping** : BeautifulSoup, Requests
  - **Évaluation RAG** : Ragas
  - **Déploiement** : Docker, Kubernetes