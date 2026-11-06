"""
CarFair-TN : estimation du juste prix d'une voiture d'occasion en Tunisie (interface en français et en anglais).

Lancement, depuis la racine du dépôt :
    streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # le modèle sauvegardé utilise les transformateurs de src/

MODEL_PATH = PROJECT_ROOT / "models" / "carfair_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "carfair_model_metadata.json"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "tayara_cars_cleaned.csv"

UNKNOWN = "__unknown__"  # valeur des listes déroulantes pour « Je ne sais pas » (affichée dans la langue choisie)
# Palette : encre et gris teintés vers le bleu d'accent ; couleurs de verdict séparées de l'accent
INK, MUTED, BORDER, SURFACE = "#1d2a3a", "#5b6b80", "#dde4ee", "#f1f4f9"
PRIMARY, MEDIAN, ASKED = "#1f6fd1", "#1d2a3a", "#b45309"
VERDICT_COLORS = {"good": ("#1a7f4b", "#e2f3e9"), "fair": ("#1f5fae", "#e5eefa"), "high": ("#b45309", "#fdf0e1")}

# Libellés lisibles pour les catégories normalisées (minuscules, sans accents) du dataset, par langue
LABELS = {
    "fr": {
        "electrique": "Électrique", "hybride essence": "Hybride essence", "hybride diesel": "Hybride diesel",
        "avec kilometrage": "Avec kilométrage", "non dedouanne": "Non dédouané", "pieces manquantes": "Pièces manquantes",
        "rs": "RS", "pick up": "Pick-up",
    },
    "en": {
        "manuelle": "Manual", "automatique": "Automatic",
        "essence": "Petrol", "diesel": "Diesel", "hybride essence": "Hybrid (petrol)", "hybride diesel": "Hybrid (diesel)",
        "electrique": "Electric",
        "compacte": "Hatchback", "berline": "Sedan", "4x4": "SUV / 4x4", "autres": "Other", "utilitaire": "Van",
        "monospace": "Minivan", "pick up": "Pickup", "cabriolet": "Convertible",
        "avec kilometrage": "Used", "nouveau": "New", "rs": "Duty-suspended (RS)",
        "non dedouanne": "Not customs-cleared", "pieces manquantes": "Missing parts",
    },
}
# Noms de marques en capitales, identiques dans les deux langues
BRAND_LABELS = {"bmw": "BMW", "mg": "MG", "byd": "BYD", "gmc": "GMC", "ds": "DS", "jac": "JAC",
                "faw": "FAW", "gac": "GAC", "dfsk": "DFSK", "jmc": "JMC", "baic": "BAIC"}

# Textes de l'interface. Les variables entre accolades sont remplies au moment de l'affichage.
TEXTS = {
    "fr": {
        "slogan": "Achetez et vendez au juste prix",
        "chip_listings": "<b>{n}</b> annonces tayara.tn",
        "chip_period": "Prix de <b>{period}</b>",
        "period": "novembre 2025",
        "lead": "Acheteur ou vendeur, obtenez le prix du marché d'une voiture d'occasion, calculé à partir des "
                "annonces publiées sur tayara.tn.",
        "missing": "Modèle ou données introuvables. Exécutez d'abord les notebooks `02_preprocessing` puis `03_modeling`.",
        "unknown": "Je ne sais pas",
        "vehicle": "Le véhicule",
        "brand": "Marque",
        "model": "Modèle",
        "model_help": "« Je ne sais pas » si le modèle n'est pas dans la liste : l'estimation se basera sur la marque.",
        "year": "Année",
        "mileage": "Kilométrage (km)",
        "prefilled": "Pré-rempli avec les caractéristiques les plus courantes de ce modèle : ajustez-les pour une "
                     "estimation plus précise.",
        "gearbox": "Boîte de vitesses",
        "fuel": "Carburant",
        "body": "Carrosserie",
        "engine": "Cylindrée (L)",
        "power": "Puissance fiscale (CV)",
        "power_unit": "CV",
        "condition": "État déclaré",
        "estimate_button": "Estimer le prix",
        "placeholder": "Renseignez le véhicule, puis cliquez sur <b>Estimer le prix</b>.",
        "estimation": "Estimation",
        "range": "Fourchette probable",
        "range_note": "Dans 3 cas sur 4, le prix affiché d'une voiture estimée à ce niveau se situe dans cette fourchette.",
        "no_model": "Modèle non précisé : l'estimation repose sur la marque, l'année et les autres caractéristiques.",
        "few_listings": "Seulement {n} annonce(s) de ce modèle dans les données : estimation moins fiable.",
        "listing_title": "Vous avez repéré une annonce ?",
        "listing_caption": "Saisissez son prix pour savoir s'il est bas, dans la norme ou élevé par rapport à notre "
                           "estimation.",
        "listing_price": "Prix de l'annonce (TND)",
        "listing_placeholder": "Ex. 30 000",
        "verdict_good": "Prix bas",
        "verdict_fair": "Prix dans la norme",
        "verdict_high": "Prix élevé",
        "detail_good": "Ce prix est <b>{gap:.0f} % sous notre estimation</b>. Un prix bas peut cacher un défaut : "
                       "vérifiez l'état du véhicule avant de vous décider.",
        "detail_high": "Ce prix est <b>{gap:.0f} % au-dessus de notre estimation</b>. Comparez-le aux annonces "
                       "similaires ci-dessous avant de vous décider.",
        "detail_fair_above": "Ce prix est <b>{gap:.0f} % au-dessus de notre estimation</b>, dans la fourchette "
                             "probable : il correspond au prix du marché.",
        "detail_fair_below": "Ce prix est <b>{gap:.0f} % en dessous de notre estimation</b>, dans la fourchette "
                             "probable : il correspond au prix du marché.",
        "similar_title": "Annonces similaires sur tayara.tn",
        "similar_none": "Aucune annonce {de_group} de {y0} à {y1} parmi les annonces collectées en {period}.",
        "similar_many": "Parmi les annonces collectées en {period} : <b>{n} annonces</b> {de_group} de {y0} à {y1}. "
                        "La moitié étaient affichées entre <b>{low} et {high} TND</b>.",
        "similar_few": "Parmi les annonces collectées en {period} : seulement <b>{n} annonce(s)</b> {de_group} de {y0} "
                       "à {y1}, affichée(s) entre <b>{low} et {high} TND</b>.",
        "zone_half": "La moitié des annonces",
        "zone_all": "Annonces",
        "median": "Prix médian",
        "our_estimate": "Notre estimation",
        "asked": "Prix de l'annonce",
        "spread": "Les prix de ces annonces varient fortement (finition, motorisation, importation…) : renseignez "
                  "précisément les caractéristiques, et prenez l'estimation avec prudence.",
        "about": "À propos des estimations",
        "about_source": "**D'où viennent les prix ?** Des {n} annonces de voitures publiées sur tayara.tn en {period}, "
                        "après vérification et nettoyage. Ce sont les prix affichés par les vendeurs.",
        "about_trust": "**Jusqu'où s'y fier ?** Testée sur des annonces qu'elle n'avait jamais vues, l'estimation était "
                       "à moins de 10 % du prix affiché pour {p10:.0f} % des voitures, et à moins de 20 % pour "
                       "{p20:.0f} % d'entre elles.",
        "about_limits": "**Ce qu'elle ne voit pas.** L'état réel du véhicule et ses options ne sont pas pris en compte : "
                        "l'estimation est un point de départ pour négocier, pas une expertise.",
    },
    "en": {
        "slogan": "Buy and sell at the right price",
        "chip_listings": "<b>{n}</b> tayara.tn listings",
        "chip_period": "<b>{period}</b> prices",
        "period": "November 2025",
        "lead": "Buying or selling? Get the market price of a used car, based on listings published on tayara.tn.",
        "missing": "Model or data not found. Run the `02_preprocessing` then `03_modeling` notebooks first.",
        "unknown": "I don't know",
        "vehicle": "Your car",
        "brand": "Make",
        "model": "Model",
        "model_help": "Choose \"I don't know\" if the model is not listed: the estimate will be based on the make.",
        "year": "Year",
        "mileage": "Mileage (km)",
        "prefilled": "Pre-filled with this model's most common specs. Adjust them for a more accurate estimate.",
        "gearbox": "Transmission",
        "fuel": "Fuel type",
        "body": "Body type",
        "engine": "Engine size (L)",
        "power": "Tax horsepower (CV)",
        "power_unit": "CV",
        "condition": "Condition",
        "estimate_button": "Get the estimate",
        "placeholder": "Enter the car's details, then click <b>Get the estimate</b>.",
        "estimation": "Estimate",
        "range": "Likely range",
        "range_note": "In 3 out of 4 cases, the asking price of a car with this estimate falls within this range.",
        "no_model": "Model not specified: the estimate is based on the make, the year and the other specs.",
        "few_listings": "Only {n} listing(s) of this model in the data, so the estimate is less reliable.",
        "listing_title": "Spotted a listing?",
        "listing_caption": "Enter its asking price to see whether it is below, in line with or above the market.",
        "listing_price": "Asking price (TND)",
        "listing_placeholder": "e.g. 30,000",
        "verdict_good": "Below market",
        "verdict_fair": "Fair price",
        "verdict_high": "Above market",
        "detail_good": "This price is <b>{gap:.0f}% below our estimate</b>. A low price may hide a problem: inspect the "
                       "car carefully before you decide.",
        "detail_high": "This price is <b>{gap:.0f}% above our estimate</b>. Compare it with the similar listings below "
                       "before you decide.",
        "detail_fair_above": "This price is <b>{gap:.0f}% above our estimate</b>, within the likely range: it is in "
                             "line with the market.",
        "detail_fair_below": "This price is <b>{gap:.0f}% below our estimate</b>, within the likely range: it is in "
                             "line with the market.",
        "similar_title": "Similar listings on tayara.tn",
        "similar_none": "Our {period} data has no {group} listings from {y0} to {y1}.",
        "similar_many": "Our {period} data has <b>{n} {group} listings</b> from {y0} to {y1}. Half of them were "
                        "priced between <b>{low} and {high} TND</b>.",
        "similar_few": "Our {period} data has only <b>{n} {group} listing(s)</b> from {y0} to {y1}, priced between "
                       "<b>{low} and {high} TND</b>.",
        "zone_half": "Middle half of listings",
        "zone_all": "Listings",
        "median": "Median price",
        "our_estimate": "Our estimate",
        "asked": "Asking price",
        "spread": "Prices for these listings vary widely (trim, engine, import status…). Fill in the specs carefully "
                  "and treat the estimate with caution.",
        "about": "About the estimates",
        "about_source": "**Where do the prices come from?** From {n} car listings published on tayara.tn in {period}, "
                        "checked and cleaned. These are the sellers' asking prices.",
        "about_trust": "**How accurate is it?** On listings it had never seen, the estimate was within 10% of the "
                       "asking price for {p10:.0f}% of cars, and within 20% for {p20:.0f}%.",
        "about_limits": "**What it doesn't account for.** The car's actual condition and its options are not taken "
                        "into account: the estimate is a starting point for negotiation, not a professional appraisal.",
    },
}

CSS = f"""
<style>
header[data-testid="stHeader"] {{ display: none; }}
.block-container {{ padding-top: 6.2rem; padding-left: 2.5rem; padding-right: 2.5rem; max-width: 1400px; }}
.cf-nav {{ position: fixed; top: 0; left: 0; right: 0; height: 64px; z-index: 1000001; background: #ffffff;
          border-bottom: 1px solid {BORDER}; display: flex; align-items: center; justify-content: space-between;
          padding: 0 11rem 0 2.5rem; box-shadow: 0 1px 3px rgba(29, 42, 58, 0.05); }}
.st-key-lang_switch {{ position: fixed; top: 14px; right: 2.5rem; z-index: 1000002; width: auto; }}
.cf-brand {{ display: flex; align-items: center; gap: 0.65rem; min-width: 0; }}
.cf-logo {{ font-size: 2rem; line-height: 1; display: inline-flex; align-items: center; transform: translateY(-0.2em); }}
.cf-title {{ font-family: "Sora", sans-serif; font-size: 1.35rem; font-weight: 700; color: {INK}; letter-spacing: -0.02em; line-height: 1;
             white-space: nowrap; }}
.cf-slogan {{ color: {MUTED}; font-size: 0.92rem; margin-left: 0.35rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.cf-chips {{ display: flex; flex-wrap: wrap; gap: 0.5rem; }}
.cf-chip {{ border: 1px solid {BORDER}; border-radius: 999px; padding: 0.2rem 0.75rem; font-size: 0.85rem;
            color: {MUTED}; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.cf-chip b {{ color: {INK}; font-weight: 600; }}
.cf-lead {{ color: {MUTED}; font-size: 1.02rem; margin: 0 0 1.1rem 0; }}
.block-container h3 {{ font-size: 1.3rem !important; padding-bottom: 0.35rem; }}
@media (max-width: 760px) {{
  .cf-slogan, .cf-nav .cf-chips {{ display: none; }}
  .cf-nav {{ padding: 0 8rem 0 1rem; }}
  .st-key-lang_switch {{ right: 1rem; }}
  .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
}}
.cf-eyebrow {{ color: {MUTED}; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; }}
.cf-price {{ font-family: "Sora", sans-serif; font-size: 2.7rem; font-weight: 700; color: {PRIMARY};
             line-height: 1.15; margin-top: 0.15rem; font-variant-numeric: tabular-nums; letter-spacing: -0.01em; }}
.cf-range {{ color: {INK}; margin-top: 0.35rem; font-variant-numeric: tabular-nums; }}
.cf-note {{ color: {MUTED}; font-size: 0.85rem; margin-top: 0.15rem; }}
.cf-placeholder {{ color: {MUTED}; text-align: center; padding: 3.5rem 1rem; }}
.cf-pill {{ display: inline-block; border-radius: 999px; padding: 0.3rem 0.9rem; font-weight: 600; font-size: 0.95rem; }}
.cf-verdict-text {{ color: {INK}; margin-top: 0.55rem; font-variant-numeric: tabular-nums; }}
.cf-bar {{ position: relative; height: 10px; background: {SURFACE}; border-radius: 999px; margin: 1.6rem 0.6rem 1.4rem; }}
.cf-span {{ position: absolute; top: 0; height: 10px; background: #c9d3e0; border-radius: 999px; }}
.cf-mark {{ position: absolute; width: 3px; border-radius: 2px; transform: translateX(-1.5px); }}
.cf-mark.median {{ top: -9px; height: 28px; background: {MEDIAN}; }}
.cf-mark.asked {{ top: -13px; height: 36px; background: {ASKED}; }}
.cf-dot {{ position: absolute; top: -4px; width: 18px; height: 18px; border-radius: 50%; background: {PRIMARY};
          border: 3px solid #ffffff; box-shadow: 0 0 0 1px {PRIMARY}; transform: translateX(-9px); }}
.cf-legend {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.45rem 1.2rem; font-size: 0.86rem; color: {MUTED};
              font-variant-numeric: tabular-nums; }}
.cf-legend b {{ color: {INK}; font-weight: 600; }}
.cf-key {{ display: inline-block; vertical-align: -2px; margin-right: 0.45rem; }}
.cf-key.span {{ width: 18px; height: 10px; border-radius: 999px; background: #c9d3e0; }}
.cf-key.median {{ width: 3px; height: 14px; border-radius: 2px; background: {MEDIAN}; }}
.cf-key.estimate {{ width: 12px; height: 12px; border-radius: 50%; background: {PRIMARY}; }}
.cf-key.asked {{ width: 3px; height: 14px; border-radius: 2px; background: {ASKED}; }}
@media (max-width: 640px) {{ .cf-legend {{ grid-template-columns: 1fr; }} .cf-price {{ font-size: 2.2rem; }} }}
</style>
"""


def t(key, **values):
    """Texte de l'interface dans la langue choisie."""
    return TEXTS[lang][key].format(**values)


def label(value):
    if value == UNKNOWN:
        return t("unknown")
    text = str(value)
    if text in BRAND_LABELS:
        return BRAND_LABELS[text]
    return LABELS[lang].get(text, text.title() if " " in text or "-" in text else text.capitalize())


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


@st.cache_data
def load_listings():
    return pd.read_csv(DATA_PATH)


def options(series):
    """Valeurs présentes dans les données, de la plus fréquente à la plus rare."""
    return series.dropna().value_counts().index.tolist()


def tnd(value):
    """30975.4 -> '30 975' en français, '30,975' en anglais."""
    text = f"{value:,.0f}"
    return text.replace(",", " ") if lang == "fr" else text


def de(name):
    """Préposition avec élision : « de Peugeot », « d'Audi »."""
    return f"d'{name}" if name[:1].lower() in "aeiouyhéèêâî" else f"de {name}"


def price_range(estimate, ranges):
    """Fourchette contenant le prix affiché dans 3 cas sur 4, selon la gamme de l'estimation."""
    for r in ranges:
        if r["estimation_min"] <= estimate < r["estimation_max"]:
            return estimate * r["ratio_bas"], estimate * r["ratio_haut"]
    return estimate * ranges[-1]["ratio_bas"], estimate * ranges[-1]["ratio_haut"]


def optional(value):
    return None if value == UNKNOWN else value


def choose(title, choices, state_key, default_index=0, fmt=None, **kwargs):
    """
    Liste déroulante dont les libellés dépendent de la langue. Streamlit retient le choix d'après le libellé
    affiché : on garde donc la valeur choisie à part, et la liste est recréée (clé suffixée par la langue)
    au changement de langue, positionnée sur cette valeur.
    """
    saved = st.session_state.get(state_key)
    index = choices.index(saved) if saved in choices else default_index
    value = st.selectbox(title, choices, index=index, format_func=fmt or label, key=f"{state_key}|{lang}", **kwargs)
    st.session_state[state_key] = value
    return value


def market_bar(low, high, median, estimate, asked=None, zone_label=""):
    """
    Barre à l'échelle : zone où se trouvent la plupart des annonces similaires, prix médian,
    estimation et prix de l'annonce saisie. Aucune étiquette sur la barre (elles se chevaucheraient) :
    les valeurs sont dans la légende.
    """
    sep = " :" if lang == "fr" else ":"  # espace avant les deux-points en français seulement
    values = [low, high, median, estimate] + ([asked] if asked else [])
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.06 or hi * 0.1
    lo, hi = lo - pad, hi + pad
    pos = lambda v: (v - lo) / (hi - lo) * 100
    asked_mark = f'<div class="cf-mark asked" style="left:{pos(asked):.2f}%"></div>' if asked else ""
    asked_key = (f'<div><span class="cf-key asked"></span>{t("asked")}{sep} <b>{tnd(asked)} TND</b></div>'
                 if asked else "")
    return f"""
<div class="cf-bar">
  <div class="cf-span" style="left:{pos(low):.2f}%; width:{pos(high) - pos(low):.2f}%"></div>
  <div class="cf-mark median" style="left:{pos(median):.2f}%"></div>
  {asked_mark}
  <div class="cf-dot" style="left:{pos(estimate):.2f}%"></div>
</div>
<div class="cf-legend">
  <div><span class="cf-key span"></span>{zone_label}{sep} <b>{tnd(low)} – {tnd(high)} TND</b></div>
  <div><span class="cf-key median"></span>{t("median")}{sep} <b>{tnd(median)} TND</b></div>
  <div><span class="cf-key estimate"></span>{t("our_estimate")}{sep} <b>{tnd(estimate)} TND</b></div>
  {asked_key}
</div>
"""


def verdict_pill(kind):
    color, background = VERDICT_COLORS[kind]
    return f'<span class="cf-pill" style="color:{color}; background:{background}">{t("verdict_" + kind)}</span>'


st.set_page_config(page_title="CarFair-TN", page_icon="🚗", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

# Choix de la langue, placé à droite de la barre de navigation ; il est aussi lisible dans l'URL (?lang=en)
with st.container(key="lang_switch"):
    lang = st.segmented_control("Langue / Language", ["fr", "en"], default="fr", required=True,
                                format_func=str.upper, key="lang", bind="query-params",
                                label_visibility="collapsed")

if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error(t("missing"))
    st.stop()

model, metadata = load_model()
listings = load_listings()
price_ranges = metadata["fourchette_3_cas_sur_4"]
period = t("period")

# ---------------------------------------------------------------------------
# En-tête
# ---------------------------------------------------------------------------
st.markdown(
    f"""
<nav class="cf-nav">
  <div class="cf-brand">
    <span class="cf-logo">🚗</span><span class="cf-title">CarFair-TN</span>
    <span class="cf-slogan">{t("slogan")}</span>
  </div>
  <div class="cf-chips">
    <span class="cf-chip">{t("chip_listings", n=tnd(metadata['annonces']))}</span>
    <span class="cf-chip">{t("chip_period", period=period)}</span>
  </div>
</nav>
<p class="cf-lead">{t("lead")}</p>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1.15], gap="large")

# ---------------------------------------------------------------------------
# Colonne de gauche : caractéristiques du véhicule
# ---------------------------------------------------------------------------
# Chaque champ a une clé fixe : changer de langue ne réinitialise pas les valeurs saisies
with left, st.container(border=True, height="stretch"):
    st.subheader(t("vehicle"))

    col1, col2 = st.columns(2)
    with col1:
        brands = sorted(listings["marque"].dropna().unique(), key=label)
        marque = st.selectbox(t("brand"), brands, index=brands.index("peugeot") if "peugeot" in brands else 0,
                              format_func=label, key="marque")
    with col2:
        brand_listings = listings[listings["marque"] == marque]
        brand_models = sorted(brand_listings["modele"].dropna().unique(), key=str)
        most_common = brand_listings["modele"].mode()
        default_model = brand_models.index(most_common.iloc[0]) if len(most_common) else len(brand_models)
        modele = choose(t("model"), brand_models + [UNKNOWN], f"modele|{marque}", default_model,
                        help=t("model_help"))

    col1, col2 = st.columns(2)
    with col1:
        annee = st.number_input(t("year"), min_value=1980, max_value=2026, value=2018, step=1, key="annee")
    with col2:
        kilometrage = st.number_input(t("mileage"), min_value=0, max_value=1_000_000, value=100_000,
                                      step=5_000, key="kilometrage")

    # Les champs facultatifs sont pré-remplis avec les caractéristiques les plus courantes de ce modèle
    same_model = brand_listings if modele == UNKNOWN else brand_listings[brand_listings["modele"] == modele]

    def optional_select(title, column, values, fmt=label):
        choices = [UNKNOWN] + list(values)
        typical = same_model[column].dropna().mode()
        index = choices.index(typical.iloc[0]) if len(typical) and typical.iloc[0] in choices else 0
        # La clé dépend du modèle choisi : le pré-remplissage se met à jour quand on change de modèle
        return choose(title, choices, f"{column}|{marque}|{modele}", index, fmt)

    st.caption(t("prefilled"))
    col1, col2 = st.columns(2)
    with col1:
        boite = optional_select(t("gearbox"), "boite", options(listings["boite"]))
        carburant = optional_select(t("fuel"), "carburant", options(listings["carburant"]))
        carrosserie = optional_select(t("body"), "type_de_carrosserie", options(listings["type_de_carrosserie"]))
    with col2:
        cylindree = optional_select(t("engine"), "cylindree", sorted(listings["cylindree"].dropna().unique()),
                                    fmt=lambda v: label(v) if v == UNKNOWN else f"{v:.1f} L")
        puissance = optional_select(t("power"), "puissance_fiscale",
                                    sorted(listings["puissance_fiscale"].dropna().unique()),
                                    fmt=lambda v: label(v) if v == UNKNOWN else f"{v:.0f} {t('power_unit')}")
        etat = optional_select(t("condition"), "etat_du_vehicule", options(listings["etat_du_vehicule"]))

    if st.button(t("estimate_button"), type="primary", width="stretch", key="estimate"):
        st.session_state["estimated"] = True  # le résultat reste affiché et suit ensuite les modifications

# ---------------------------------------------------------------------------
# Colonne de droite : trois cartes (estimation, prix d'une annonce, annonces similaires)
# ---------------------------------------------------------------------------
with right:
    if not st.session_state.get("estimated"):
        with st.container(border=True, height="stretch", vertical_alignment="center"):
            st.markdown(f'<div class="cf-placeholder"><p>{t("placeholder")}</p></div>', unsafe_allow_html=True)
    else:
        car = pd.DataFrame([{
            "kilometrage": float(kilometrage),
            "annee": int(annee),
            "cylindree": optional(cylindree),
            "puissance_fiscale": optional(puissance),
            "marque": marque,
            "modele": optional(modele),
            "boite": optional(boite),
            "carburant": optional(carburant),
            "type_de_carrosserie": optional(carrosserie),
            "etat_du_vehicule": optional(etat),
        }])[metadata["variables"]]

        price = float(np.expm1(model.predict(car))[0])
        low, high = price_range(price, price_ranges)
        car_name = f"{label(marque)} {label(modele) if modele != UNKNOWN else ''} {annee}".replace("  ", " ")
        sep = " :" if lang == "fr" else ":"

        # Carte 1 : estimation et fourchette probable
        with st.container(border=True):
            st.subheader(t("estimation"))
            st.markdown(
                f'<div class="cf-eyebrow">{car_name} · {tnd(kilometrage)} km</div>'
                f'<div class="cf-price">{tnd(price)} TND</div>'
                f'<div class="cf-range">{t("range")}{sep} <b>{tnd(low)} – {tnd(high)} TND</b></div>'
                f'<div class="cf-note">{t("range_note")}</div>',
                unsafe_allow_html=True,
            )
            if modele == UNKNOWN:
                st.info(t("no_model"))
            elif len(same_model) < 10:
                st.warning(t("few_listings", n=len(same_model)))

        # Carte 2 : comparer le prix d'une annonce
        with st.container(border=True):
            # Comparer le prix d'une annonce à l'estimation : en dehors de la fourchette probable, le prix est bas ou élevé
            st.subheader(t("listing_title"))
            st.caption(t("listing_caption"))
            asked = st.number_input(t("listing_price"), min_value=0, max_value=5_000_000, value=None,
                                    step=500, placeholder=t("listing_placeholder"), key="asked")
            if asked:
                gap = (asked - price) / price * 100
                if asked < low:
                    kind, detail = "good", t("detail_good", gap=abs(gap))
                elif asked > high:
                    kind, detail = "high", t("detail_high", gap=gap)
                else:
                    kind = "fair"
                    detail = t("detail_fair_above" if gap >= 0 else "detail_fair_below", gap=abs(gap))
                st.markdown(f'{verdict_pill(kind)}<div class="cf-verdict-text">{detail}</div>', unsafe_allow_html=True)

        # Carte 3 : annonces similaires
        with st.container(border=True):
            # Annonces similaires : même marque (et même modèle si connu), année à ±1 an
            st.subheader(t("similar_title"))
            similar = listings[(listings["marque"] == marque) & listings["annee"].between(annee - 1, annee + 1)]
            if modele != UNKNOWN:
                similar = similar[similar["modele"] == modele]
            car_group = f"{label(marque)}" + (f" {label(modele)}" if modele != UNKNOWN else "")
            words = dict(group=car_group, de_group=de(car_group), y0=annee - 1, y1=annee + 1, period=period)
            prices = similar["prix"]
            if similar.empty:
                st.write(t("similar_none", **words))
            else:
                # À partir de 4 annonces, on montre la zone où se trouve la moitié d'entre elles (25 % – 75 %),
                # moins sensible aux annonces extrêmes que le minimum et le maximum
                if len(prices) >= 4:
                    # Quartiles pris parmi les prix réellement affichés (pas de valeur interpolée)
                    zone_low = prices.quantile(0.25, interpolation="nearest")
                    zone_high = prices.quantile(0.75, interpolation="nearest")
                    zone_label = t("zone_half")
                    sentence = t("similar_many", n=len(prices), low=tnd(zone_low), high=tnd(zone_high), **words)
                else:
                    zone_low, zone_high, zone_label = prices.min(), prices.max(), t("zone_all")
                    sentence = t("similar_few", n=len(prices), low=tnd(zone_low), high=tnd(zone_high), **words)
                st.markdown(f'<div class="cf-verdict-text">{sentence}</div>', unsafe_allow_html=True)
                st.markdown(market_bar(zone_low, zone_high, prices.median(), price, asked, zone_label),
                            unsafe_allow_html=True)
                if len(similar) >= 3 and prices.max() / prices.min() > 2.5:
                    st.warning(t("spread"))

# ---------------------------------------------------------------------------
# À propos des estimations
# ---------------------------------------------------------------------------
with st.expander(t("about")):
    test = metadata["test"]
    st.markdown(t("about_source", n=tnd(metadata["annonces"]), period=period))
    st.markdown(t("about_trust", p10=test["À ±10 % (%)"], p20=test["À ±20 % (%)"]))
    st.markdown(t("about_limits"))
