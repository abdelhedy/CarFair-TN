"""
Nettoyage ligne par ligne des annonces Tayara.

Toutes les fonctions de ce module n'utilisent QUE la ligne traitée (aucune
statistique calculée sur d'autres lignes), elles peuvent donc être appliquées
avant le split train/test sans fuite de données.

Seule exception documentée : `price_plausibility_reference()` qui sert à
filtrer les labels aberrants (contrôle qualité de la cible, pas une feature).
"""
import re
import unicodedata

import numpy as np
import pandas as pd

CURRENT_YEAR = 2026
YEAR_MIN, YEAR_MAX = 1980, CURRENT_YEAR
PF_MIN, PF_MAX = 1, 30
CYL_MIN, CYL_MAX = 0.6, 7.0
KM_MAX = 1_000_000
PRICE_MIN, PRICE_MAX = 1_000, 1_000_000


# ---------------------------------------------------------------------------
# Texte
# ---------------------------------------------------------------------------
def normalize_text(s):
    """Minuscules, sans accents, espaces compactés. NaN -> ''."""
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s.lower()).strip()


def clean_description(s):
    """Supprime le 'Lire plus' ajouté par le scraping à chaque description."""
    if pd.isna(s):
        return ""
    return re.sub(r"\s*Lire plus\s*", " ", str(s)).strip()


def listing_text(row):
    """Titre + description normalisés (source pour l'extraction)."""
    return normalize_text(f"{row.get('titre', '')} . {clean_description(row.get('description'))}")


# ---------------------------------------------------------------------------
# Catégories : orthographe unifiée + synonymes
# ---------------------------------------------------------------------------
CARBURANT_MAP = {
    "essence": "essence",
    "diesel": "diesel",
    "hybride essence": "hybride essence",
    "hybride": "hybride essence",
    "hybride diesel": "hybride diesel",
    "electrique": "electrique",
}
BOITE_MAP = {"manuelle": "manuelle", "automatique": "automatique"}
CARROSSERIE_MAP = {
    "berline": "berline",
    "compacte": "compacte",
    "compact": "compacte",
    "4 x 4": "4x4",
    "4x4": "4x4",
    "utilitaire": "utilitaire",
    "pick up": "pick up",
    "monospace": "monospace",
    "cabriolet": "cabriolet",
    "autres": "autres",
}
COULEUR_MAP = {"or": "dore", "doree": "dore"}
ETAT_MAP = {
    "avec kilometrage": "avec kilometrage",
    "nouveau": "nouveau",
    "rs": "rs",
    "non dedouanne": "non dedouanne",
    "pieces manquantes": "pieces manquantes",
}


def normalize_category(value, mapping=None):
    """Normalise une valeur catégorielle. Valeur hors mapping -> NaN."""
    v = normalize_text(value)
    if not v or v == "nan":
        return np.nan
    if mapping is None:
        return v
    return mapping.get(v, np.nan)


def normalize_couleur(value):
    v = normalize_text(value)
    if not v or v == "nan":
        return np.nan
    return COULEUR_MAP.get(v, v)


# ---------------------------------------------------------------------------
# Marques : liste de référence fermée (indépendante des données)
# ---------------------------------------------------------------------------
# Marques présentes sur Tayara, complétées par les marques vendues ou importées
# en Tunisie. 'AC' et 'Acrea' sont volontairement absentes : premières marques
# de la liste du site, elles servent de valeur par défaut et ne désignent
# presque jamais la vraie voiture.
KNOWN_BRANDS = {
    # Europe
    "abarth", "alfa romeo", "aston martin", "audi", "bentley", "bmw", "citroen", "cupra",
    "dacia", "ds", "ferrari", "fiat", "iveco", "jaguar", "lada", "lamborghini", "lancia",
    "land rover", "maserati", "mercedes-benz", "mini", "opel", "peugeot", "porsche",
    "renault", "rolls-royce", "rover", "saab", "seat", "skoda", "smart", "volkswagen", "volvo",
    # Asie
    "acura", "daewoo", "daihatsu", "honda", "hyundai", "infiniti", "isuzu", "kia", "lexus",
    "mahindra", "mazda", "mitsubishi", "nissan", "proton", "ssangyong", "subaru", "suzuki",
    "tata", "toyota", "vinfast",
    # Chine
    "baic", "brilliance", "byd", "changan", "changhe", "chery", "dfsk", "dongfeng", "exeed",
    "faw", "foton", "gac", "geely", "great wall", "haval", "jac", "jaecoo", "jetour", "jmc",
    "leapmotor", "mg", "omoda", "soueast",
    # Amérique
    "cadillac", "chevrolet", "chrysler", "dodge", "ford", "gmc", "hummer", "jeep", "lincoln",
    "pontiac", "ram", "tesla",
    # Tunisie
    "wallyscar",
    # Catégorie du site pour une marque absente de sa liste
    "autres",
}
BRAND_ALIASES = {
    "vw": "volkswagen",
    "mercedes": "mercedes-benz",
    "mercedes benz": "mercedes-benz",
    "benz": "mercedes-benz",
    "range rover": "land rover",
    "landrover": "land rover",
    "alfa-romeo": "alfa romeo",
    "rolls royce": "rolls-royce",
    "great wall motors": "great wall",
    "gwm": "great wall",
    "ssang yong": "ssangyong",
    "kgm": "ssangyong",
    "ds automobiles": "ds",
    "mini cooper": "mini",
}
# Noms trop ambigus pour être cherchés seuls. Dans un titre, 'mini' et 'jeep' désignent
# presque toujours la marque ; dans une description, beaucoup moins (« prix mini »,
# « jeep » pour un 4x4). 'smart' est aussi une finition (« polo 8 smart »).
BRANDS_NOT_SEARCHED = {
    "titre": {"autres", "smart"},
    "description": {"autres", "smart", "mini", "jeep", "ram"},
}
TEXT_ONLY_BRAND_ALIASES = {
    "jeep wrangler": "jeep", "jeep compass": "jeep", "jeep renegade": "jeep",
    "jeep cherokee": "jeep", "grand cherokee": "jeep",
    "smart fortwo": "smart", "smart forfour": "smart",
}


def normalize_marque(value):
    """Marque de la liste de référence ; toute autre valeur -> NaN."""
    v = normalize_category(value)
    if pd.isna(v):
        return np.nan
    v = BRAND_ALIASES.get(v, v)
    return v if v in KNOWN_BRANDS else np.nan


def fix_marque_from_modele(marque, modele):
    """Un Range Rover est un Land Rover (le site range parfois ces annonces sous 'Rover')."""
    if isinstance(modele, str) and modele.startswith("range rover"):
        return "land rover"
    return marque


# ---------------------------------------------------------------------------
# Modèles : filtre de forme
# ---------------------------------------------------------------------------
_MODEL_BRAND_PREFIX = re.compile(r"^[a-z][a-z -]*\.")      # 'peugeot.208', 'over.range rover'
_MODEL_FORBIDDEN_CHARS = re.compile(r"[^a-z0-9 +\-/'.]")   # puces, emojis, ':', ','
_MODEL_FORBIDDEN_WORDS = re.compile(r"\b(afficher|numero|telephone|contact|lire|prix)\b")


def normalize_modele(value):
    """
    Modèle normalisé. Retire le préfixe de marque ('Peugeot.208' -> '208') et
    rejette le texte de description ('Afficher le numéro', '• SUV robuste...').
    """
    v = normalize_text(value)
    if not v or v == "nan":
        return np.nan
    v = _MODEL_BRAND_PREFIX.sub("", v).strip()
    if (not v or _MODEL_FORBIDDEN_CHARS.search(v) or _MODEL_FORBIDDEN_WORDS.search(v)
            or len(v.split()) > 4):
        return np.nan
    return v


# ---------------------------------------------------------------------------
# Valeurs numériques
# ---------------------------------------------------------------------------
def clean_annee(x):
    if pd.isna(x):
        return np.nan
    x = int(x)
    return float(x) if YEAR_MIN <= x <= YEAR_MAX else np.nan


def clean_puissance_fiscale(x):
    if pd.isna(x):
        return np.nan
    return float(x) if PF_MIN <= x <= PF_MAX else np.nan


def parse_cylindree(val):
    """'1.6L' -> 1.6 ; '1461 cm3' -> 1.461 ; '<1.0L' -> 0.8 ; '>4.0L' -> 5.0."""
    if pd.isna(val):
        return np.nan
    s = normalize_text(val).replace(",", ".")
    if "km" in s or "mk" in s:
        return np.nan
    if s.startswith("<"):
        return 0.8
    if s.startswith(">"):
        return 5.0
    s = re.sub(r"\(.*?\)", "", s)
    m = re.search(r"\d+(?:\.\d+)?", s)
    if not m:
        return np.nan
    x = float(m.group(0))
    liters = x / 1000 if ("cm" in s or "cc" in s or x > 20) else x
    return round(liters, 1) if CYL_MIN <= liters <= CYL_MAX else np.nan


def _digits(x):
    """Chiffres d'une valeur brute ('73 500' -> '73500', 145000.0 -> '145000')."""
    if pd.isna(x):
        return ""
    if isinstance(x, (int, float, np.integer, np.floating)):
        return str(int(x))
    return re.sub(r"\D", "", str(x))


def _is_repeated_digits(s):
    """
    Placeholder 'chiffre répété' : 111, 5555, 99999, 1111111, 111111110, 1111111200.
    Volontairement strict : 55000, 99000 ou 110000 sont des valeurs réelles.
    """
    return bool(re.fullmatch(r"(\d)\1{2,}", s) or re.fullmatch(r"(\d)\1{3,}\d?0*", s))


def fix_kilometrage(raw, annee):
    """
    Corrige le kilométrage d'une annonce.
    Retourne (valeur, tag) où tag décrit la correction appliquée.
    """
    s = _digits(raw)
    if not s:
        return np.nan, "missing"

    # 1. Kilométrage collé à l'année par le scraping : '1440002021' -> 144000
    if pd.notna(annee) and len(s) >= 7 and s.endswith(str(int(annee))):
        s = s[: -len(str(int(annee)))]
        if not s:
            return np.nan, "glued_year"
        tag = "glued_year"
    else:
        tag = "original"

    if len(s) >= 4 and _is_repeated_digits(s):
        return np.nan, "placeholder"
    km = float(s)
    if km > KM_MAX:
        return np.nan, "placeholder"

    # 2. Valeurs < 1000 : écrites en milliers ('181' = 181 000 km),
    #    sauf pour une voiture réellement récente.
    if km < 1000:
        if pd.isna(annee):
            return np.nan, "ambiguous"
        age = CURRENT_YEAR - annee
        if age <= 1:
            return km, tag                      # voiture neuve : 0-999 km plausible
        if km == 0:
            return np.nan, "placeholder"        # 0 km sur une voiture de 5 ans
        candidate = km * 1000
        if candidate <= 60_000 * age:           # max ~60 000 km / an
            return candidate, "x1000"
        return np.nan, "ambiguous"
    return km, tag


KNOWN_PRICE_PLACEHOLDERS = {
    123, 1233, 1234, 1235, 4123, 12345, 123456, 1234567, 12345678, 123456789,
    999, 9999, 99999, 999999, 55111110,
}


def normalize_price(p):
    """
    Interprète la valeur du champ prix.
    Retourne (prix_TND_candidat, tag). La plausibilité est vérifiée plus tard.
    """
    if pd.isna(p):
        return np.nan, "missing"
    p = float(p)
    s = str(int(p))
    if p < 10:
        return np.nan, "placeholder"            # 1-9 : surtout '5'
    if int(p) in KNOWN_PRICE_PLACEHOLDERS or _is_repeated_digits(s):
        return np.nan, "placeholder"
    if p > 1e9:
        return np.nan, "placeholder"
    if p > PRICE_MAX:
        # 8 chiffres commençant par 2/4/5/9 et non ronds : numéro de téléphone
        if len(s) == 8 and s[0] in "2459" and p % 1000 != 0:
            return np.nan, "phone"
        if p % 1000 == 0:
            return p / 1000, "millimes"          # 36 000 000 millimes = 36 000 TND
        return np.nan, "garbage"
    if p < 1000:
        return p * 1000, "x1000"                # '61' = 61 000 TND
    return p, "original"


# ---------------------------------------------------------------------------
# Extraction depuis le texte (mots entiers uniquement)
# ---------------------------------------------------------------------------
_NUM = r"(\d{1,3}(?:[ .,]\d{3})+|\d+)"
_EXCLUDE_PRICE_CONTEXT = re.compile(r"neu(f|ve)|concession|achete|vendu chez|a partir de|location|par jour")


def _to_number(s):
    return float(re.sub(r"[ .,]", "", s))


def extract_prix(text):
    """Prix explicitement annoncé ('prix 38 000', '46md', '95000 dinars')."""
    patterns = [
        re.compile(rf"prix\s*[:\-]?\s*{_NUM}\s*(md|mdt|mille|m dt|dt|tnd|dinars?)?\b"),
        re.compile(rf"{_NUM}\s*(md|mdt|mille dinars?|m dt|dt|tnd|dinars?)\b"),
    ]
    for rx in patterns:
        for m in rx.finditer(text):
            before = text[max(0, m.start() - 40): m.start()]
            after = text[m.end(): m.end() + 15]
            if _EXCLUDE_PRICE_CONTEXT.search(before) or re.search(r"km|compteur|kilom", after):
                continue
            value = _to_number(m.group(1))
            unit = m.group(2) or ""
            if unit.startswith(("md", "mille", "m dt")):
                value *= 1000
            return value
    return np.nan


def extract_annee(text):
    m = re.search(r"\b(?:annee|modele|model|mod)\s*[:\-]?\s*((?:19|20)\d{2})\b", text)
    if not m:
        m = re.search(r"\b((?:19[89]|20[0-2])\d)\b(?!\s*(?:km|dt|tnd|cc|cm|mille|\d))", text)
    return clean_annee(int(m.group(1))) if m else np.nan


def extract_kilometrage(text):
    m = re.search(rf"{_NUM}\s*(?:k\s?m|kms|kilometres?)\b", text)
    if m:
        return _to_number(m.group(1))
    m = re.search(r"\b(\d{1,3})\s*mille\s*(?:km|au compteur)", text)
    return float(m.group(1)) * 1000 if m else np.nan


_ENGINE_CODES = r"tsi|tdi|tfsi|hdi|dci|crdi|vti|thp|puretech|gti|tce|mpi|ecoboost|bluehdi|cdi|fsi"


def extract_cylindree(text):
    m = re.search(rf"\b(\d[.,]\d)\s*(?:l\b|litres?\b|(?:{_ENGINE_CODES})\b)", text)
    if m:
        v = float(m.group(1).replace(",", "."))
        return v if CYL_MIN <= v <= CYL_MAX else np.nan
    m = re.search(r"\b(\d{3,4})\s*(?:cc|cm3)\b", text)
    if m:
        v = round(int(m.group(1)) / 1000, 1)
        return v if CYL_MIN <= v <= CYL_MAX else np.nan
    return np.nan


def extract_puissance_fiscale(text):
    # 'cv' = chevaux fiscaux en Tunisie ; 'ch'/'chevaux' = puissance réelle -> ignorés
    m = re.search(r"\b(\d{1,2})\s*cv\b", text)
    return clean_puissance_fiscale(int(m.group(1))) if m else np.nan


def _first_match(text, rules):
    for label, pattern in rules:
        if re.search(pattern, text):
            return label
    return np.nan


CARBURANT_RULES = [
    ("hybride diesel", r"\bhybride? diesel\b"),
    ("hybride essence", r"\bhybride?\b"),
    ("electrique", r"\b(electrique|electric)\b"),
    ("diesel", rf"\b(diesel|gasoil|gazoil|tdi|hdi|dci|crdi|cdi|bluehdi)\b"),
    ("essence", rf"\b(essence|benzine|tsi|tfsi|vti|thp|tce|puretech|mpi|ecoboost|fsi)\b"),
]
BOITE_RULES = [
    ("automatique", r"\b(automatique|auto|bva|dsg|cvt|tiptronic|edc|eat\d)\b"),
    ("manuelle", r"\b(manuelle|manuel|bvm)\b"),
]
COULEUR_RULES = [
    ("blanc", r"\bblanc(he)?s?\b"), ("noir", r"\bnoire?s?\b"), ("gris", r"\bgrise?s?\b"),
    ("rouge", r"\brouges?\b"), ("bleu", r"\bbleue?s?\b"), ("vert", r"\bverte?s?\b"),
    ("jaune", r"\bjaunes?\b"), ("orange", r"\borange\b"), ("marron", r"\bmarron\b"),
    ("beige", r"\bbeige\b"), ("argent", r"\bargente?e?\b"), ("dore", r"\bdoree?\b"),
]
ETAT_RULES = [
    ("pieces manquantes", r"\bpieces? manquantes?\b"),
    ("non dedouanne", r"\bnon d[e]?douane?\b"),
    ("nouveau", r"\b(0|zero) ?km\b"),
    ("avec kilometrage", r"\b(1ere|premiere|2eme|deuxieme|1er|premier) main\b|\boccasion\b"),
]
CARROSSERIE_RULES = [
    ("pick up", r"\bpick[ -]?up\b"),
    ("cabriolet", r"\b(cabriolet|cabrio|decapotable|convertible)\b"),
    ("monospace", r"\b(monospace|minivan|mpv)\b"),
    ("utilitaire", r"\b(utilitaire|fourgon|camionnette)\b"),
    ("4x4", r"\b(4 ?x ?4|suv|tout[ -]terrain|crossover)\b"),
    ("berline", r"\b(berline|sedan)\b"),
    ("compacte", r"\b(compacte?|citadine|hatchback)\b"),
    ("autres", r"\b(coupe|break|roadster)\b"),
]


def extract_carburant(text):
    return _first_match(text, CARBURANT_RULES)


def extract_boite(text):
    return _first_match(text, BOITE_RULES)


def extract_couleur(text):
    return _first_match(text, COULEUR_RULES)


def extract_etat(text):
    return _first_match(text, ETAT_RULES)


def extract_carrosserie(text):
    return _first_match(text, CARROSSERIE_RULES)


def build_brand_patterns(source="titre"):
    """
    Liste (marque, regex mot-entier) construite depuis la liste de référence,
    noms les plus longs d'abord ('land rover' avant 'rover').
    source : 'titre' ou 'description' (exclusions plus strictes pour la description).
    """
    names = {b: b for b in KNOWN_BRANDS - BRANDS_NOT_SEARCHED[source]}
    names.update(BRAND_ALIASES)
    names.update(TEXT_ONLY_BRAND_ALIASES)
    return [(brand, re.compile(rf"\b{re.escape(name)}\b"))
            for name, brand in sorted(names.items(), key=lambda kv: -len(kv[0]))]


def extract_marque(text, brand_patterns):
    for brand, rx in brand_patterns:
        if rx.search(text):
            return brand
    return np.nan


def build_model_patterns(brand_models):
    """{marque: [(modele, regex)]}, modèles les plus longs d'abord."""
    return {
        b: [(m, re.compile(rf"\b{re.escape(m)}\b")) for m in sorted(ms, key=len, reverse=True) if len(m) >= 2]
        for b, ms in brand_models.items()
    }


def extract_modele(text, marque, model_patterns):
    for modele, rx in model_patterns.get(marque, []):
        if rx.search(text):
            return modele
    return np.nan


# ---------------------------------------------------------------------------
# Annonces qui ne sont pas des voitures
# ---------------------------------------------------------------------------
# Titre qui COMMENCE par une pièce/un accessoire ('pneus de voiture', 'retroviseur pajero')
NON_CAR_TITLE_START = re.compile(
    r"^\W*(?:\d+\s+|lot de\s+|a vendre\s+|vente\s+|divers\s+)?"
    r"(?:pneus?|jantes?|roues|retroviseurs?|pieces?|accessoires?|compteur|moteur)\b"
)
# Expressions sans ambiguïté n'importe où dans le titre
NON_CAR_TITLE_ANY = re.compile(
    r"\b(?:pour pieces?|pieces? (?:et|de|d)|location|louer|cherche|recherche|obd|"
    r"avec chauffeur|mariage)\b"
)


def is_non_car(titre):
    t = normalize_text(titre)
    return bool(NON_CAR_TITLE_START.search(t) or NON_CAR_TITLE_ANY.search(t))


# ---------------------------------------------------------------------------
# Contrôle de plausibilité du prix (filtre qualité sur la cible)
# ---------------------------------------------------------------------------
REFERENCE_LEVELS = [
    ["marque", "modele", "annee"],
    ["marque", "annee"],
    ["marque", "modele"],
    ["marque"],
    ["annee"],
]


def price_plausibility_reference(df, trusted_mask, price_col="prix_candidat", min_count=3):
    """
    Médiane du log-prix des annonces fiables, par niveau hiérarchique.
    Retourne une Series alignée sur df : prix de référence (log) par ligne.
    """
    log_p = np.log(df.loc[trusted_mask, price_col])
    ref = pd.Series(np.nan, index=df.index)
    for keys in REFERENCE_LEVELS:
        trusted = df.loc[trusted_mask, keys].assign(_lp=log_p)
        stats = trusted.groupby(keys)["_lp"].agg(["median", "count"])
        stats = stats[stats["count"] >= min_count]["median"].rename("_ref")
        level_ref = df[keys].join(stats, on=keys)["_ref"]
        ref = ref.fillna(level_ref)
    return ref.fillna(log_p.median())
