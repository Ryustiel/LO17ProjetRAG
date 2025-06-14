import os
import csv
import requests
import re
import concurrent.futures
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
from typing import List, Tuple, Literal, cast

# ==============================================================================
# --- CONFIGURATION & DONNÉES ---
# ==============================================================================

OUTPUT_DIR = "dataset_rag_lol_definitive"
KNOWLEDGE_BASE_DIR = os.path.join(OUTPUT_DIR, "knowledge_base")
EVALUATION_FILENAME = os.path.join(OUTPUT_DIR, "evaluation.csv")

CHAMPION_LIST_URL = "https://leagueoflegends.fandom.com/wiki/List_of_champions"
UNIVERSE_BASE_URL = "https://universe.leagueoflegends.com/fr_FR"

REGIONS_RAW = """
Bandle
Bilgewater
Demacia
Freljord
Ionia
Ixtal
Le Néant
Noxus
Piltover
Shurima
Targon
Zaun
Iles Obscures
"""
REGIONS = [line.strip() for line in REGIONS_RAW.strip().split("\n")]

# Dictionnaire de forçage pour les slugs URL qui ne suivent pas les règles standard.
SLUG_MAPPING_OVERRIDES = {
    "Aurelion Sol": "aurelionsol",
    "Dr. Mundo": "drmundo",
    "Jarvan IV": "jarvaniv",
    "K'Sante": "ksante",
    "Lee Sin": "leesin",
    "Master Yi": "masteryi",
    "Miss Fortune": "missfortune",
    "Nunu & Willump": "nunu",
    "Renata Glasc": "renataglasc",
    "Tahm Kench": "tahmkench",
    "Twisted Fate": "twistedfate",
    "Xin Zhao": "xinzhao",
    "Wukong": "monkeyking",
    "Bandle": "bandle-city",
    "Le Néant": "void",
    "Targon": "mount-targon",
    "Iles Obscures": "shadow-isles",
}

# Injection manuelle du lore pour les personnages sans page de biographie standard.
MANUAL_LORE_DATA = {
    "ambessa": """Née dans l'une des plus puissantes familles de l'empire de Noxus moderne, Ambessa Medarda a peut-être toujours été destinée à la grandeur. Bien que sa famille ne fasse pas partie des vieilles familles nobles, elle est parvenue à amasser du respect et de l'influence à travers tout l'empire depuis sa fondation. La jeune Ambessa fut très tôt confrontée à la vision du sang. Elle se rendait à l'arène de l'Ordalie pour y observer les gladiateurs qui risquaient leur vie dans l'espoir de se couvrir de gloire. Même si elle était trop jeune pour connaître elle-même la joie du combat, elle étudiait chaque affrontement et intégrait chaque mouvement des combattants. Plus tard, après la bataille d'Hildenard, son père l'envoya récupérer les lames des soldats tombés au combat. Même si elle n'était encore qu'une enfant, Ambessa ne détourna pas les yeux du carnage qui l'entourait. À la fin de la journée, elle avait compris qu'il ne fallait pas craindre la mort, mais la respecter. Le sacrifice est noble. Et la grandeur ne s'atteint pas sans. Le code de la famille Medarda, transmis de génération en génération depuis leurs premiers jours en tant que commerçants des côtes de Shurima, liait les vertus du renard du désert et celles du terrifiant loup des légendes. Ambessa choisit donc, sans surprise, la vie de soldat. Forte des leçons qu'elle avait tirées de ses aventures d'enfance, elle forçait les autres à respecter ses idéaux d'honneur familial, toujours avec des actions décisives. Elle était fière d'être une fille des Medarda. Elle était une guerrière née et, bientôt, elle devint un général commandant de nombreux régiments, à la grande fierté du patriarche de sa famille, son grand-père Menelik. Et pourtant, elle était bien plus. Elle était également une femme, une amante et une mère. Son appétit pour la vie conduisit Ambessa à faire de nombreuses expériences. Mais quand elle tint son fils Kino dans ses bras pour la première fois, elle comprit enfin comment on pouvait dédier sa propre existence à autrui, de façon totalement inconditionnelle. Mais cela lui ouvrit également les portes vers une profonde déception. Même si elle l'aimait profondément, il était clair que Kino n'avait pas le cœur d'un guerrier. Peu après, Ambessa faillit mourir au combat en défendant le foyer ancestral de sa famille, Rokrund, alors qu'elle était enceinte de sa fille, Mel. Étendue auprès des corps de ses alliés et de ses ennemis, prise entre la vie et la mort, elle eut des visions qu'elle ne partagea qu'avec peu de personnes durant sa vie. Ce qu'Ambessa vit ne fit que renforcer sa détermination et son ambition. Le monde ploierait sous sa volonté, pour que ses ennemis ne puissent jamais exploiter la faiblesse de ses enfants. À partir de cet instant, l'ascension d'Ambessa devint fulgurante. Elle dirigeait depuis le front à chaque bataille, jetant un regard noir à la mort. Et après chaque victoire, elle devenait plus ingénieuse, plus téméraire et plus impitoyable. Quand le vieux Menelik Medarda finit par mourir, il ne nomma aucun héritier sur son lit de mort, déclenchant ainsi une guerre de succession au sein de sa propre famille. Pour Ambessa, ce n'était que du vent. Ses adversaires n'avaient aucune chance. C'était sa destinée. Ses rivaux furent vaincus et elle se promit de forger un héritage digne du nom Medarda. Un héritage digne de ses enfants. En tant que matriarche, Ambessa put commencer à parler plus librement de sa propre devise. « Soyez le loup en toutes choses. » Elle ne pardonnait aucune faiblesse et aucune dissension dans son entourage, afin que cette faiblesse ne puisse jamais l'affecter. Elle envoya même sa fille Mel dans la lointaine cité de Piltover. Bien des années après, Ambessa entendit des rumeurs concernant une invention puissante appelée l'« Hextech », fabriquée par les idéalistes sans échine de Piltover. Intrigué par le potentiel d'une telle découverte, Ambessa se rendit à la cité dorée pour rendre visite à sa fille, afin de déterminer si cette technologie pourrait servir la famille Medarda...""",
    "mel": """Mère, Un soldat m'a offert ton masque aujourd'hui. Sans réfléchir, j'en ai parcouru les fissures, capté chaque bosse, chaque cicatrice des innombrables combats dont tu es sortie victorieuse... et celle du combat dont tu ne t'es pas relevée. Ce n'est que maintenant, alors que notre navire vogue en direction de Noxus, que la réalité s'impose à moi. Tu n'es plus là. Une fois encore. Mais cette fois, je ne peux plus espérer ton retour. Je sais que tu ne voudrais pas que je m'attarde sur ta mort. Tu me dirais d'être fière. Je suis enfin devenue « le loup » que tu désirais si ardemment. Mais je ne peux m'empêcher de me demander si je suis devenue ce que tu espérais... ou simplement ce que tu avais besoin que je sois. Tant de choses ont changé, et pourtant une part de moi est aussi perdue que je l'étais une décennie plus tôt. Quand j'y repense, le visage de cousin Jago à mon arrivée à Piltover était empreint de pitié. Tu m'avais bien fait comprendre que j'étais livrée en offrande et que tu ne reviendrais pas me chercher. Malgré tout, des années durant je m'endormais en désirant te revoir, comme n'importe quelle fille désirerait voir sa mère. Malgré tout ce que tu m'avais fait. Chaque matin, je m'éveillais aux côtés du vide que tu m'avais imposé. J'ai passé ma vie à tenter de remplir ce vide, à devenir digne de l'amour de ma propre mère. J'ai vécu de la seule façon que je connais : en suivant la voie du Renard, aussi rapide que rusée. J'ai gagné la confiance du Conseil et ai failli tenir Piltover dans le creux de ma main. Si seulement tu avais pu me faire confiance, Mère. Je ne me serais pas retrouvée dans cette situation. Je n'aurais pas eu à. Non. Ce n'est pas si simple. Je sais à présent que tu tentais de me protéger, à ta façon. Je n'aurais jamais pu imaginer qu'une telle magie sommeillait en moi. Mais il y a tant de choses de mon passé auxquelles je n'ai jamais pensé avant ton décès. Honnêtement, c'est insoutenable, et c'est une raison de plus pour laquelle j'aurais aimé pouvoir combattre à tes côtés. Je n'ose imaginer ce que tu as pu ressentir, à être pourchassée par la Rose noire pendant toutes ces années. Mais je sais que si tu as ressenti de la peur, ce ne fut jamais pour toi-même. Ce fut pour Kino et pour moi. Et en fin de compte, je t'ai menée à ta perte. Peut-être savais-tu tout du long que j'étais celle qu'il fallait craindre. Je suis désolée, Mère, mais je ne regrette pas d'avoir protégé ma ville. Il faut faire des sacrifices pour devenir plus fort. N'est-ce pas là ce que tu répétais sans cesse ? Ton crédo, ton excuse face à n'importe quelle situation. T'es-tu jamais souciée de ce que ça m'a fait ? De ce que ça nous a fait ? Est-ce que ça en a valu la peine ? C'est désormais à moi de supporter le coût de tout cela. Tu m'as caché tant de choses. La vérité sur mon père. Sur le meurtrier de Kino. Et plus important encore, sur cette vendetta que la Rose noire avait contre toi, et dans laquelle je suis désormais impliquée. Je suppose que cela ne fait qu'effleurer la surface de tes mensonges, et de ceux de cette « Manipulatrice ». Je compte bien découvrir tout ce que tu m'as caché. Je regrette que ces mots ne te parviennent jamais, mais j'espère que tu m'observes depuis Volrachnun. Je vais jeter cette lettre par-dessus bord, afin qu'elle puisse être entraînée jusqu'aux profondeurs et ne faire plus qu'un avec les eaux des côtes de Rokrund, où tu as autrefois vaincu la mort. Je vais bientôt arriver en étrangère dans le pays où je suis née. Nos propres gardes ne me voient pas comme une véritable Medarda, bien qu'ils n'aient encore jamais osé exprimer à haute voix leur méfiance à mon égard. Une nation qui prône la force, mais qui prospère grâce aux effusions de sang n'est pas une nation que je peux fièrement appeler mienne. Et je ne resterai pas les bras croisés alors que ce chaos se poursuit. Tu m'as appris à survivre, Mère. J'ai appris par moi-même à vivre. Et bien que tu m'aies poussée à suivre la voie du Loup, je n'abandonnerai jamais celle du Renard. Ce n'est peut-être pas ainsi que tu l'avais imaginé... mais je rentre à la maison, Mère. Et je vais faire la différence. Jusqu'à ce que mon cœur cesse de battre. Ta fille, Mel""",
}

# --- PARAMETRES DE PERFORMANCE ---
MAX_WORKERS = 16

# ==============================================================================
# --- FONCTIONS ---
# ==============================================================================


def generate_slug(name: str, subject_type: Literal["champion", "region"]) -> str:
    """Génère un slug URL pour un sujet, en priorisant le dictionnaire de forçage."""
    if name in SLUG_MAPPING_OVERRIDES:
        return SLUG_MAPPING_OVERRIDES[name]

    slug = name.lower()
    slug = re.sub(r"[.'’]", "", slug)
    return re.sub(r"\s+", "" if subject_type == "champion" else "-", slug)


def get_champion_names() -> List[str]:
    """Récupère la liste complète des noms de champions depuis le wiki Fandom."""
    print("1. Récupération de la liste des champions...")
    try:
        response = requests.get(CHAMPION_LIST_URL, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        header = soup.find("span", id="List_of_Available_Champions")
        table = header.find_next("table", class_="article-table")

        champions = [
            cell["data-sort-value"].strip()
            for cell in table.find_all("td", {"data-sort-value": True})
            if cell.find("a", href=re.compile(r"/wiki/.*/LoL"))
        ]

        unique_champions = sorted(list(set(champions)))
        print(f"   -> {len(unique_champions)} champions uniques trouvés.")
        return unique_champions
    except Exception as e:
        print(f"[ERREUR] Échec de la récupération de la liste des champions: {e}")
        return []


def fetch_and_save_lore(
    subject_info: Tuple[str, Literal["champion", "region"]],
) -> Tuple[str, bool, str]:
    """Worker exécuté en parallèle pour scraper et sauvegarder le lore d'un sujet."""
    subject_name, subject_type = subject_info
    slug = generate_slug(subject_name, subject_type)
    url_part = "story/champion" if subject_type == "champion" else "region"
    url = f"{UNIVERSE_BASE_URL}/{url_part}/{slug}/"

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 404:
            return subject_name, False, f"404 (URL: {url})"

        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        meta_tag = soup.find("meta", attrs={"name": "description"})

        if not (meta_tag and meta_tag.get("content")):
            return subject_name, False, "Meta description vide/manquante"

        file_path = os.path.join(KNOWLEDGE_BASE_DIR, f"{slug}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(meta_tag["content"].strip())
        return subject_name, True, file_path

    except requests.exceptions.RequestException as e:
        return subject_name, False, str(e)


def create_knowledge_base(champions_to_scrape: List[str], regions: List[str]):
    """Orchestre la création de la base de connaissances."""
    print("\n2. Création de la base de connaissances...")
    os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)

    print("   - Injection du lore manuel...")
    for slug, content in MANUAL_LORE_DATA.items():
        with open(
            os.path.join(KNOWLEDGE_BASE_DIR, f"{slug}.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(content)

    print("   - Lancement du scraping parallèle pour le reste des sujets...")
    tasks: list[tuple[str, Literal["champion", "region"]]] = [
        (name, cast(Literal["champion", "region"], "champion"))
        for name in champions_to_scrape
    ] + [(name, cast(Literal["champion", "region"], "region")) for name in regions]

    if not tasks:
        print("   -> Aucun sujet à scraper.")
        return

    success_count, fail_count = 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_subject = {
            executor.submit(fetch_and_save_lore, task): task for task in tasks
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_subject),
            total=len(tasks),
            desc="   Progression",
        ):
            subject_name, success, message = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                tqdm.write(f"     [ECHEC] {subject_name}: {message}")

    print(
        f"\n   -> Opération terminée. {success_count} fichiers scrapés, {fail_count} échecs."
    )


def create_evaluation_file():
    """Génère le fichier CSV d'évaluation."""
    print("\n3. Création du fichier d'évaluation...")

    existing_files = {f for f in os.listdir(KNOWLEDGE_BASE_DIR) if f.endswith(".txt")}
    questions = []

    potential_questions = [
        (
            ("vi.txt", "jinx.txt"),
            "Quel est le lien entre Vi et Jinx ?",
            "Elles sont sœurs devenues ennemies.",
            "vi.txt;jinx.txt",
        ),
        (
            ("garen.txt", "darius.txt"),
            "Comparer Garen et Darius.",
            "Garen est un soldat d'élite de Demacia, loyal à son roi. Darius est un général impitoyable de Noxus qui ne croit qu'en la force.",
            "garen.txt;darius.txt",
        ),
        (
            ("jayce.txt", "viktor.txt"),
            "Quel est le conflit idéologique entre Jayce et Viktor ?",
            "Jayce veut un progrès contrôlé de l'Hextech, Viktor prône une fusion homme-machine ('Glorieuse Évolution').",
            "jayce.txt;viktor.txt",
        ),
        (
            ("caitlyn.txt", "vi.txt"),
            "Qui est la partenaire de Caitlyn ?",
            "Vi.",
            "caitlyn.txt;vi.txt",
        ),
        (
            ("nasus.txt", "renekton.txt"),
            "Pourquoi Renekton en veut-il à son frère Nasus ?",
            "Renekton croit à tort que Nasus l'a trahi en l'enfermant dans un tombeau pendant des millénaires.",
            "nasus.txt;renekton.txt",
        ),
        (
            ("ambessa.txt", "mel.txt"),
            "Quelle est la relation entre Ambessa et Mel ?",
            "Ambessa est la mère de Mel. C'est une relation complexe où Ambessa, une seigneur de guerre noxienne, a exilé sa fille Mel à Piltover.",
            "ambessa.txt;mel.txt",
        ),
    ]

    for files_needed, q, a, s in potential_questions:
        if all(f in existing_files for f in files_needed):
            questions.append([q, a, s])

    questions.append(
        [
            "Garen est-il originaire de Zaun ?",
            "Non, Garen est originaire de Demacia.",
            "garen.txt;demacia.txt",
        ]
    )
    questions.append(
        [
            "Qui sont les parents de Jinx ?",
            "L'information n'est pas clairement précisée dans les textes, seulement qu'ils sont morts.",
            "jinx.txt;vi.txt",
        ]
    )

    with open(EVALUATION_FILENAME, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "reponse_attendue", "source_ideale"])
        writer.writerows(questions)
    print(
        f"   -> Fichier '{EVALUATION_FILENAME}' créé avec {len(questions)} questions."
    )


# ==============================================================================
# --- SCRIPT PRINCIPAL ---
# ==============================================================================


def main():
    """Orchestre la création complète du dataset."""
    start_time = time.time()
    print("=" * 60)
    print("--- Générateur de Dataset RAG pour League of Legends (DÉFINITIF) ---")
    print("=" * 60)

    all_champions = get_champion_names()
    if not all_champions:
        print("Arrêt : la liste de champions est vide.")
        return

    # Exclure les champions gérés manuellement de la liste de scraping
    champions_to_exclude = ["Ambessa Medarda", "Mel Medarda"]
    champions_to_scrape = [c for c in all_champions if c not in champions_to_exclude]

    create_knowledge_base(champions_to_scrape, REGIONS)
    create_evaluation_file()

    end_time = time.time()
    print("\n" + "=" * 60)
    print("--- OPÉRATION TERMINÉE ---")
    print(f"Temps d'exécution : {end_time - start_time:.2f} secondes.")
    print(f"Dataset prêt dans le dossier : '{os.path.abspath(OUTPUT_DIR)}'")
    print("=" * 60)


if __name__ == "__main__":
    main()
