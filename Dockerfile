FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=true

# --- Dépendances Système ---
# Installation de 'uv' et des dépendances système nécessaires.
RUN apt-get update && apt-get install -y --no-install-recommends curl libmagic1 && \
    pip install uv && \
    rm -rf /var/lib/apt/lists/*

# --- Configuration de l'Application ---
WORKDIR /app

COPY pyproject.toml ./

# Installation les dépendances Python avec uv.
RUN uv sync

# --- Utilisateur et Permissions ---
RUN useradd -ms /bin/bash appuser

RUN chown -R appuser:appuser /app

USER appuser

COPY . .

# --- Exécution ---
EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]