# ===================================================================
# ==         Lanceur pour le projet Chroniqueur de Runeterra       ==
# ===================================================================
# Auteur: Votre expert en analyse de code
# Version: 1.2 - Gestion propre de l'arrêt de Streamlit
# Description:
# Ce script automatise l'installation, la configuration et le
# lancement du projet RAG sur League of Legends.
# ===================================================================

# --- Configuration de l'encodage UTF-8 ---
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# --- Configuration et Couleurs ---
Clear-Host
$Host.UI.RawUI.WindowTitle = "Chroniqueur de Runeterra - Lanceur"

# Couleurs
$ColorInfo = "Cyan"
$ColorPrompt = "Yellow"
$ColorSuccess = "Green"
$ColorWarning = "Yellow"
$ColorError = "Red"
$ColorTitle = "Magenta"
$ColorDefault = $Host.UI.RawUI.ForegroundColor

# --- Fonctions utilitaires ---

# Fonction pour écrire du texte en couleur
function Write-Host-Colored {
    param(
        [string]$Message,
        [string]$Color = $ColorDefault
    )
    Write-Host $Message -ForegroundColor $Color
}

# Fonction pour vérifier si 'uv' est installé et l'installer si besoin
function Ensure-Uv-Is-Installed {
    Write-Host-Colored "Vérification de l'installation de 'uv'..." $ColorInfo
    $uv_path = Get-Command uv -ErrorAction SilentlyContinue
    if ($null -eq $uv_path) {
        Write-Host-Colored "'uv' n'est pas détecté. Lancement de l'installation..." $ColorWarning
        try {
            powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
            Write-Host-Colored "Installation de 'uv' terminée." $ColorSuccess
            $uvInstallPath = Join-Path $env:USERPROFILE ".local\bin"
            $env:Path += ";$uvInstallPath"
            Write-Host-Colored "Le chemin de 'uv' a été ajouté à la session actuelle." $ColorInfo
        } catch {
            Write-Host-Colored "Erreur lors de l'installation de 'uv'. Veuillez vérifier votre connexion internet ou les permissions." $ColorError
            exit 1
        }
    } else {
        Write-Host-Colored "'uv' est déjà installé." $ColorSuccess
    }
}

# Fonction pour télécharger et décompresser le projet depuis GitHub
function Download-And-Unzip-Repo {
    $repoUrl = "https://github.com/Ryustiel/LO17ProjetRAG/archive/refs/heads/main.zip"
    $zipPath = Join-Path $PSScriptRoot "LO17ProjetRAG.zip"
    $extractPath = $PSScriptRoot
    $finalRepoDir = Join-Path $PSScriptRoot "LO17ProjetRAG"
    $unzippedDirName = "LO17ProjetRAG-main"

    if (-not (Test-Path $finalRepoDir -PathType Container)) {
        Write-Host-Colored "Téléchargement du projet depuis GitHub..." $ColorInfo
        try {
            Invoke-WebRequest -Uri $repoUrl -OutFile $zipPath
            Write-Host-Colored "Téléchargement terminé. Décompression de l'archive..." $ColorInfo
            Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force
            Rename-Item -Path (Join-Path $extractPath $unzippedDirName) -NewName "LO17ProjetRAG"
            Remove-Item $zipPath
            Write-Host-Colored "Projet 'LO17ProjetRAG' prêt." $ColorSuccess
        } catch {
            Write-Host-Colored "Une erreur est survenue lors du téléchargement ou de la décompression." $ColorError
            if (Test-Path $zipPath) { Remove-Item $zipPath }
            exit 1
        }
    } else {
        Write-Host-Colored "Le dossier du projet existe déjà." $ColorInfo
    }
    Set-Location $finalRepoDir
}

# Fonction pour l'installation complète
function Setup-Project {
    Write-Host-Colored "--- Lancement de l'installation complète ---" $ColorTitle
    Write-Host-Colored "`n1. Configuration des clés d'API" $ColorInfo
    Write-Host "Le projet a besoin d'une clé API Google (obligatoire) et OpenAI (optionnelle, pour créer le jeu de test)."
    $googleApiKey = Read-Host -Prompt "Entrez votre GOOGLE_API_KEY"
    $openaiApiKey = Read-Host -Prompt "Entrez votre OPENAI_API_KEY (laissez vide si non disponible/nécessaire)"

    $envContent = "GOOGLE_API_KEY=$googleApiKey"
    if (-not [string]::IsNullOrEmpty($openaiApiKey)) {
        $envContent += "`nOPENAI_API_KEY=$openaiApiKey"
    }
    Set-Content -Path ".env" -Value $envContent -Encoding UTF8
    Write-Host-Colored "Fichier .env créé avec succès." $ColorSuccess

    Write-Host-Colored "`n2. Installation des dépendances Python avec 'uv sync'..." $ColorInfo
    uv sync; if ($LASTEXITCODE -ne 0) { Write-Host-Colored "Erreur." $ColorError; return }
    Write-Host-Colored "Dépendances installées." $ColorSuccess

    Write-Host-Colored "`n3. Construction de la base de connaissances (cela peut prendre quelques minutes)..." $ColorInfo
    $envFilePath = Join-Path $PWD ".env"
    Write-Host-Colored "   - Étape 3a: Scraping des données..." $ColorInfo
    uv run --env-file $envFilePath data_scrapper.py; if ($LASTEXITCODE -ne 0) { Write-Host-Colored "Erreur." $ColorError; return }
    Write-Host-Colored "   - Étape 3b: Création de la base de données vectorielle..." $ColorInfo
    uv run --env-file $envFilePath create_database.py; if ($LASTEXITCODE -ne 0) { Write-Host-Colored "Erreur." $ColorError; return }

    Write-Host-Colored "`n--- Installation complète terminée avec succès ! ---`n" $ColorSuccess
}

# --- Script Principal ---
Ensure-Uv-Is-Installed
Download-And-Unzip-Repo

do {
    Write-Host-Colored "=================================================" $ColorTitle
    Write-Host-Colored "      MENU - Chroniqueur de Runeterra" $ColorTitle
    Write-Host-Colored "=================================================" $ColorTitle
    Write-Host-Colored "1. Installation et Configuration Complète" $ColorPrompt
    Write-Host "   (Configure les clés API, installe les dépendances et crée la base de données)"
    Write-Host-Colored "2. Lancer l'application Streamlit (le site)" $ColorPrompt
    Write-Host-Colored "3. Lancer l'évaluation du système RAG" $ColorPrompt
    Write-Host-Colored "4. Générer le jeu de données d'évaluation" $ColorPrompt
    Write-Host-Colored "5. Quitter" $ColorPrompt

    $choice = Read-Host "`nFaites votre choix"

    switch ($choice) {
        "1" {
            Setup-Project
        }
        "2" {
            $envFile = ".env"
            if (-not (Test-Path "database/chroma_db")) {
                Write-Host-Colored "`n[ERREUR] La base de données n'existe pas. Veuillez d'abord lancer l'option 1." $ColorError
                Read-Host "`nAppuyez sur Entrée pour retourner au menu..."
            } else {
                # On lance Streamlit en tant que tâche d'arrière-plan (Job)
                $streamlitJob = $null
                try {
                    Write-Host-Colored "`nLancement de l'application Streamlit en arrière-plan..." $ColorInfo
                    $streamlitJob = Start-Job -ScriptBlock {
                        param($projectDir, $environmentFileName)

                        Set-Location $projectDir
                        Write-Host "Job: Répertoire de travail défini sur : $projectDir"
                        Write-Host "Job: Utilisation du fichier d'environnement : $environmentFileName"
                        Write-Host "Job: Lancement de Streamlit..."

                        uv run --env-file $environmentFileName streamlit run streamlit_app.py

                    } -ArgumentList $PWD, $envFile
                    # On attend un peu pour que le serveur démarre
                    Start-Sleep -Seconds 5

                    Write-Host-Colored "`n==========================================================" $ColorSuccess
                    Write-Host-Colored "  Le serveur Streamlit est DÉMARRÉ !" $ColorSuccess
                    Write-Host-Colored "  Ouvrez votre navigateur et allez à l'adresse suivante :" $ColorSuccess
                    Write-Host-Colored "  http://localhost:8501" -ForegroundColor "White"
                    Write-Host-Colored "==========================================================" $ColorSuccess
                    Write-Host ""
                    Read-Host -Prompt "Appuyez sur ENTRÉE dans cette fenêtre pour ARRÊTER le serveur et retourner au menu"
                
                } finally {
                    if ($null -ne $streamlitJob) {
                        Write-Host-Colored "`nArrêt du serveur Streamlit..." $ColorWarning
                        Stop-Job -Job $streamlitJob
                        Remove-Job -Job $streamlitJob
                        Write-Host-Colored "Serveur arrêté." $ColorSuccess
                    }
                }
            }
        }
        "3" {
            $envFile = ".env"
            if ((-not (Test-Path "database/chroma_db")) -or (-not (Test-Path "dataset_rag_lol_definitive/synthetic_evaluation.csv"))) {
                 Write-Host-Colored "`n[ERREUR] La base de données ou le fichier d'évaluation 'synthetic_evaluation.csv' est manquant." $ColorError
                 Write-Host-Colored "Veuillez d'abord lancer l'option 1 (pour la base de données) et/ou l'option 4 (pour le fichier d'évaluation)." $ColorError
            } else {
                Write-Host-Colored "`nLancement de l'évaluation... Les résultats s'afficheront ici." $ColorInfo
                uv run --env-file $envFile evaluation.py
                Write-Host-Colored "`nÉvaluation terminée. Les résultats détaillés sont dans 'evaluation_results.csv'." $ColorSuccess
            }
        }
        "4" {
            Write-Host-Colored "`nGénération du jeu de données d'évaluation..." $ColorInfo
            $envFile = ".env"
            $knowledgeBasePath = "dataset_rag_lol_definitive/knowledge_base"
            $openaiKeyPresent = $false

            if (Test-Path $envFile) {
                $envVars = Get-Content $envFile | ConvertFrom-StringData -Delimiter '='
                if ($envVars.OPENAI_API_KEY -and $envVars.OPENAI_API_KEY.Trim() -ne "") {
                    $openaiKeyPresent = $true
                }
            }

            if (-not $openaiKeyPresent) {
                Write-Host-Colored "`n[ERREUR] La clé OPENAI_API_KEY est manquante ou vide dans le fichier .env." $ColorError
                Write-Host-Colored "Veuillez configurer la clé OpenAI via l'option 1 pour générer le jeu de données." $ColorError
            } elseif (-not (Test-Path $knowledgeBasePath -PathType Container)) {
                Write-Host-Colored "`n[ERREUR] Le dossier de la base de connaissances '$knowledgeBasePath' est manquant." $ColorError
                Write-Host-Colored "Veuillez d'abord lancer l'option 1 pour scraper les données et créer la base de connaissances." $ColorError
            } else {
                Write-Host-Colored "Lancement de 'generate_testset.py' (cela peut prendre quelques minutes et consommer des crédits API)..." $ColorInfo
                uv run --env-file $envFile generate_testset.py
                if ($LASTEXITCODE -ne 0) {
                    Write-Host-Colored "Erreur lors de la génération du jeu de données." $ColorError
                } else {
                    Write-Host-Colored "Jeu de données d'évaluation 'synthetic_evaluation.csv' généré avec succès dans 'dataset_rag_lol_definitive/'." $ColorSuccess
                }
            }
        }
        "5" {
            Write-Host-Colored "À bientôt, Invocateur !" $ColorSuccess
        }
        default {
            Write-Host-Colored "`nChoix invalide. Veuillez sélectionner une option de 1 à 5." $ColorWarning
        }
    }
    
    if ($choice -ne "5") {
        if ($choice -ne "2") {
            if (($choice -eq "2" -and (Test-Path "database/chroma_db")) -eq $false) {
                 Read-Host "`nAppuyez sur Entrée pour retourner au menu..."
            }
        }
        Clear-Host
    }

} while ($choice -ne "5")

