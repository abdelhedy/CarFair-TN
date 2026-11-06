"""
CarFair-TN : estimation du juste prix d'une voiture d'occasion en Tunisie.

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

UNKNOWN = "Je ne sais pas"
DATA_PERIOD = "novembre 2025"  # date de publication des annonces collectées
# Palette : encre et gris teintés vers le bleu d'accent ; couleurs de verdict séparées de l'accent
INK, MUTED, BORDER, SURFACE = "#1d2a3a", "#5b6b80", "#dde4ee", "#f1f4f9"
PRIMARY, MEDIAN, ASKED = "#1f6fd1", "#1d2a3a", "#b45309"
VERDICTS = {
    "good": ("Prix bas", "#1a7f4b", "#e2f3e9"),
    "fair": ("Prix dans la norme", "#1f5fae", "#e5eefa"),
    "high": ("Prix élevé", "#b45309", "#fdf0e1"),
}

# Libellés lisibles pour les catégories normalisées (minuscules, sans accents) du dataset
LABELS = {
    "electrique": "Électrique", "hybride essence": "Hybride essence", "hybride diesel": "Hybride diesel",
    "avec kilometrage": "Avec kilométrage", "non dedouanne": "Non dédouané", "pieces manquantes": "Pièces manquantes",
    "rs": "RS", "pick up": "Pick-up", "4x4": "4x4", "bmw": "BMW", "mg": "MG", "byd": "BYD", "gmc": "GMC",
    "ds": "DS", "jac": "JAC", "faw": "FAW", "gac": "GAC", "dfsk": "DFSK", "jmc": "JMC", "baic": "BAIC",
}

CSS = f"""
<style>
header[data-testid="stHeader"] {{ display: none; }}
.block-container {{ padding-top: 6.2rem; padding-left: 2.5rem; padding-right: 2.5rem; max-width: 1400px; }}
.cf-nav {{ position: fixed; top: 0; left: 0; right: 0; height: 64px; z-index: 1000001; background: #ffffff;
          border-bottom: 1px solid {BORDER}; display: flex; align-items: center; justify-content: space-between;
          padding: 0 2.5rem; box-shadow: 0 1px 3px rgba(29, 42, 58, 0.05); }}
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
  .cf-nav {{ padding: 0 1rem; }}
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


def label(value):
    if value == UNKNOWN:
        return value
    return LABELS.get(value, str(value).title() if " " in str(value) or "-" in str(value) else str(value).capitalize())


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
    """30975.4 -> '30 975' (séparateur de milliers à la française)."""
    return f"{value:,.0f}".replace(",", " ")


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


def market_bar(low, high, median, estimate, asked=None, zone_label="La moitié des annonces"):
    """
    Barre à l'échelle : zone où se trouvent la plupart des annonces similaires, prix médian,
    estimation et prix de l'annonce saisie. Aucune étiquette sur la barre (elles se chevaucheraient) :
    les valeurs sont dans la légende.
    """
    values = [low, high, median, estimate] + ([asked] if asked else [])
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.06 or hi * 0.1
    lo, hi = lo - pad, hi + pad
    pos = lambda v: (v - lo) / (hi - lo) * 100
    asked_mark = f'<div class="cf-mark asked" style="left:{pos(asked):.2f}%"></div>' if asked else ""
    asked_key = (f'<div><span class="cf-key asked"></span>Prix de l\'annonce : <b>{tnd(asked)} TND</b></div>'
                 if asked else "")
    return f"""
<div class="cf-bar">
  <div class="cf-span" style="left:{pos(low):.2f}%; width:{pos(high) - pos(low):.2f}%"></div>
  <div class="cf-mark median" style="left:{pos(median):.2f}%"></div>
  {asked_mark}
  <div class="cf-dot" style="left:{pos(estimate):.2f}%"></div>
</div>
<div class="cf-legend">
  <div><span class="cf-key span"></span>{zone_label} : <b>{tnd(low)} – {tnd(high)} TND</b></div>
  <div><span class="cf-key median"></span>Prix médian : <b>{tnd(median)} TND</b></div>
  <div><span class="cf-key estimate"></span>Notre estimation : <b>{tnd(estimate)} TND</b></div>
  {asked_key}
</div>
"""


def verdict_pill(kind):
    text, color, background = VERDICTS[kind]
    return f'<span class="cf-pill" style="color:{color}; background:{background}">{text}</span>'


st.set_page_config(page_title="CarFair-TN", page_icon="🚗", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error("Modèle ou données introuvables. Exécutez d'abord les notebooks `02_preprocessing` puis `03_modeling`.")
    st.stop()

model, metadata = load_model()
listings = load_listings()
price_ranges = metadata["fourchette_3_cas_sur_4"]

# ---------------------------------------------------------------------------
# En-tête
# ---------------------------------------------------------------------------
st.markdown(
    f"""
<nav class="cf-nav">
  <div class="cf-brand">
    <span class="cf-logo">🚗</span><span class="cf-title">CarFair-TN</span>
    <span class="cf-slogan">Achetez et vendez au juste prix</span>
  </div>
  <div class="cf-chips">
    <span class="cf-chip"><b>{tnd(metadata['annonces'])}</b> annonces tayara.tn</span>
    <span class="cf-chip">Prix de <b>{DATA_PERIOD}</b></span>
  </div>
</nav>
<p class="cf-lead">Acheteur ou vendeur, obtenez le prix du marché d'une voiture d'occasion, calculé à partir des
annonces publiées sur tayara.tn.</p>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1.15], gap="large")

# ---------------------------------------------------------------------------
# Colonne de gauche : caractéristiques du véhicule
# ---------------------------------------------------------------------------
with left, st.container(border=True, height="stretch"):
    st.subheader("Le véhicule")

    col1, col2 = st.columns(2)
    with col1:
        brands = sorted(listings["marque"].dropna().unique(), key=label)
        marque = st.selectbox("Marque", brands, index=brands.index("peugeot") if "peugeot" in brands else 0,
                              format_func=label)
    with col2:
        brand_listings = listings[listings["marque"] == marque]
        brand_models = sorted(brand_listings["modele"].dropna().unique(), key=str)
        most_common = brand_listings["modele"].mode()
        default_model = brand_models.index(most_common.iloc[0]) if len(most_common) else len(brand_models)
        modele = st.selectbox("Modèle", brand_models + [UNKNOWN], index=default_model, format_func=label,
                              help="« Je ne sais pas » si le modèle n'est pas dans la liste : "
                                   "l'estimation se basera sur la marque.")

    col1, col2 = st.columns(2)
    with col1:
        annee = st.number_input("Année", min_value=1980, max_value=2026, value=2018, step=1)
    with col2:
        kilometrage = st.number_input("Kilométrage (km)", min_value=0, max_value=1_000_000, value=100_000,
                                      step=5_000)

    # Les champs facultatifs sont pré-remplis avec les caractéristiques les plus courantes de ce modèle
    same_model = brand_listings if modele == UNKNOWN else brand_listings[brand_listings["modele"] == modele]

    def optional_select(title, column, values, fmt=label):
        choices = [UNKNOWN] + list(values)
        typical = same_model[column].dropna().mode()
        index = choices.index(typical.iloc[0]) if len(typical) and typical.iloc[0] in choices else 0
        # La clé dépend du modèle choisi : le pré-remplissage se met à jour quand on change de modèle
        return st.selectbox(title, choices, index=index, format_func=fmt, key=f"{column}|{marque}|{modele}")

    st.caption("Pré-rempli avec les caractéristiques les plus courantes de ce modèle : ajustez-les pour une "
               "estimation plus précise.")
    col1, col2 = st.columns(2)
    with col1:
        boite = optional_select("Boîte de vitesses", "boite", options(listings["boite"]))
        carburant = optional_select("Carburant", "carburant", options(listings["carburant"]))
        carrosserie = optional_select("Carrosserie", "type_de_carrosserie", options(listings["type_de_carrosserie"]))
    with col2:
        cylindree = optional_select("Cylindrée (L)", "cylindree", sorted(listings["cylindree"].dropna().unique()),
                                    fmt=lambda v: v if v == UNKNOWN else f"{v:.1f} L")
        puissance = optional_select("Puissance fiscale (CV)", "puissance_fiscale",
                                    sorted(listings["puissance_fiscale"].dropna().unique()),
                                    fmt=lambda v: v if v == UNKNOWN else f"{v:.0f} CV")
        etat = optional_select("État déclaré", "etat_du_vehicule", options(listings["etat_du_vehicule"]))

    if st.button("Estimer le prix", type="primary", width="stretch"):
        st.session_state["estimated"] = True  # le résultat reste affiché et suit ensuite les modifications

# ---------------------------------------------------------------------------
# Colonne de droite : trois cartes (estimation, prix d'une annonce, annonces similaires)
# ---------------------------------------------------------------------------
with right:
    if not st.session_state.get("estimated"):
        with st.container(border=True, height="stretch", vertical_alignment="center"):
            st.markdown(
                '<div class="cf-placeholder"><p>Renseignez le véhicule, '
                'puis cliquez sur <b>Estimer le prix</b>.</p></div>',
                unsafe_allow_html=True,
            )
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

        # Carte 1 : estimation et fourchette probable
        with st.container(border=True):
            st.subheader("Estimation")
            st.markdown(
                f'<div class="cf-eyebrow">{car_name} · {tnd(kilometrage)} km</div>'
                f'<div class="cf-price">{tnd(price)} TND</div>'
                f'<div class="cf-range">Fourchette probable : <b>{tnd(low)} – {tnd(high)} TND</b></div>'
                f'<div class="cf-note">Dans 3 cas sur 4, le prix affiché d\'une voiture estimée à ce niveau se situe dans '
                f'cette fourchette.</div>',
                unsafe_allow_html=True,
            )
            if modele == UNKNOWN:
                st.info("Modèle non précisé : l'estimation repose sur la marque, l'année et les autres caractéristiques.")
            elif len(same_model) < 10:
                st.warning(f"Seulement {len(same_model)} annonce(s) de ce modèle dans les données : "
                           "estimation moins fiable.")

        # Carte 2 : comparer le prix d'une annonce
        with st.container(border=True):
            # Comparer le prix d'une annonce à l'estimation : en dehors de la fourchette probable, le prix est bas ou élevé
            st.subheader("Vous avez repéré une annonce ?")
            st.caption("Saisissez son prix pour savoir s'il est bas, dans la norme ou élevé par rapport à notre estimation.")
            asked = st.number_input("Prix de l'annonce (TND)", min_value=0, max_value=5_000_000, value=None,
                                    step=500, placeholder="Ex. 30 000")
            if asked:
                gap = (asked - price) / price * 100
                if asked < low:
                    kind = "good"
                    detail = (f"Ce prix est <b>{abs(gap):.0f} % sous notre estimation</b>. Un prix bas peut cacher un "
                              "défaut : vérifiez l'état du véhicule avant de vous décider.")
                elif asked > high:
                    kind = "high"
                    detail = (f"Ce prix est <b>{gap:.0f} % au-dessus de notre estimation</b>. Comparez-le aux annonces "
                              "similaires ci-dessous avant de vous décider.")
                else:
                    kind = "fair"
                    detail = (f"Ce prix est <b>{abs(gap):.0f} % {'au-dessus' if gap >= 0 else 'en dessous'} de notre "
                              "estimation</b>, dans la fourchette probable : il correspond au prix du marché.")
                st.markdown(f'{verdict_pill(kind)}<div class="cf-verdict-text">{detail}</div>', unsafe_allow_html=True)

        # Carte 3 : annonces similaires
        with st.container(border=True):
            # Annonces similaires : même marque (et même modèle si connu), année à ±1 an
            st.subheader("Annonces similaires sur tayara.tn")
            similar = listings[(listings["marque"] == marque) & listings["annee"].between(annee - 1, annee + 1)]
            if modele != UNKNOWN:
                similar = similar[similar["modele"] == modele]
            car_group = f"{label(marque)}" + (f" {label(modele)}" if modele != UNKNOWN else "")
            prices = similar["prix"]
            if similar.empty:
                st.write(f"Aucune annonce {de(car_group)} de {annee - 1} à {annee + 1} parmi les annonces collectées en "
                         f"{DATA_PERIOD}.")
            else:
                # À partir de 4 annonces, on montre la zone où se trouve la moitié d'entre elles (25 % – 75 %),
                # moins sensible aux annonces extrêmes que le minimum et le maximum
                if len(prices) >= 4:
                    # Quartiles pris parmi les prix réellement affichés (pas de valeur interpolée)
                    zone_low = prices.quantile(0.25, interpolation="nearest")
                    zone_high = prices.quantile(0.75, interpolation="nearest")
                    zone_label = "La moitié des annonces"
                    sentence = (f"Parmi les annonces collectées en {DATA_PERIOD} : <b>{len(prices)} annonces</b> "
                                f"{de(car_group)} de {annee - 1} à {annee + 1}. La moitié étaient affichées entre "
                                f"<b>{tnd(zone_low)} et {tnd(zone_high)} TND</b>.")
                else:
                    zone_low, zone_high, zone_label = prices.min(), prices.max(), "Annonces"
                    sentence = (f"Parmi les annonces collectées en {DATA_PERIOD} : seulement <b>{len(prices)} "
                                f"annonce(s)</b> {de(car_group)} de {annee - 1} à {annee + 1}, affichée(s) entre "
                                f"<b>{tnd(zone_low)} et {tnd(zone_high)} TND</b>.")
                st.markdown(f'<div class="cf-verdict-text">{sentence}</div>', unsafe_allow_html=True)
                st.markdown(market_bar(zone_low, zone_high, prices.median(), price, asked, zone_label),
                            unsafe_allow_html=True)
                if len(similar) >= 3 and prices.max() / prices.min() > 2.5:
                    st.warning("Les prix de ces annonces varient fortement (finition, motorisation, importation…) : "
                               "renseignez précisément les caractéristiques, et prenez l'estimation avec prudence.")

# ---------------------------------------------------------------------------
# À propos des estimations
# ---------------------------------------------------------------------------
with st.expander("À propos des estimations"):
    test = metadata["test"]
    st.markdown(
        f"**D'où viennent les prix ?** Des {tnd(metadata['annonces'])} annonces de voitures publiées sur tayara.tn en "
        f"{DATA_PERIOD}, après vérification et nettoyage. Ce sont les prix affichés par les vendeurs."
    )
    st.markdown(
        f"**Jusqu'où s'y fier ?** Testée sur des annonces qu'elle n'avait jamais vues, l'estimation était à moins de "
        f"10 % du prix affiché pour {test['À ±10 % (%)']:.0f} % des voitures, et à moins de 20 % pour "
        f"{test['À ±20 % (%)']:.0f} % d'entre elles."
    )
    st.markdown(
        "**Ce qu'elle ne voit pas.** L'état réel du véhicule et ses options ne sont pas pris en compte : "
        "l'estimation est un point de départ pour négocier, pas une expertise."
    )
