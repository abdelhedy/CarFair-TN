"""
Anonymisation des annonces brutes avant publication.

Entrée  : data/private/tayara_cars_all_pages.csv  (données collectées, non publiées)
Sortie  : data/raw/tayara_cars_all_pages.csv      (données publiées)

- Les numéros de téléphone (titres, descriptions, et champs prix / kilométrage / puissance fiscale où
  certains vendeurs les saisissent) sont remplacés par des numéros fictifs de même format : 8 chiffres, même premier
  chiffre, mêmes séparateurs. Un même numéro donne toujours le même numéro fictif.
- L'url de l'annonce est remplacée par un identifiant anonyme (`id_annonce`).

Les numéros fictifs et les identifiants sont dérivés d'une clé secrète (data/private/anonymization_key.txt,
créée au premier lancement) : sans elle, on ne peut pas retrouver les valeurs d'origine.

Usage, depuis la racine du dépôt :
    python scripts/anonymize_raw_data.py
"""
import hashlib
import re
import secrets
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from src.cleaning import KNOWN_PRICE_PLACEHOLDERS  # noqa: E402

# Valeurs de remplissage connues (ex. 55 111 110) : ce ne sont pas des numéros de personnes, on les garde
PLACEHOLDERS = {str(v) for v in KNOWN_PRICE_PLACEHOLDERS}
PRIVATE = ROOT / "data" / "private"
SOURCE = PRIVATE / "tayara_cars_all_pages.csv"
KEY_FILE = PRIVATE / "anonymization_key.txt"
OUTPUT = ROOT / "data" / "raw" / "tayara_cars_all_pages.csv"

# Numéro tunisien : 8 chiffres commençant par 2 à 9, avec séparateurs éventuels et indicatif +216 facultatif
PHONE_IN_TEXT = re.compile(r"(?<!\d)(?:\+?216[\s.\-]?)?[2-9]\d(?:[\s.\-]?\d){6}(?!\d)")


def load_key():
    if not KEY_FILE.exists():
        KEY_FILE.write_text(secrets.token_hex(32), encoding="utf-8")
    return KEY_FILE.read_text(encoding="utf-8").strip()


def fake_phone(number, key):
    """Numéro fictif de 8 chiffres, même premier chiffre, jamais rond (comme un vrai numéro)."""
    digest = int(hashlib.sha256(f"{key}|tel|{number}".encode()).hexdigest(), 16)
    fake = number[0] + f"{digest % 10**7:07d}"
    if fake.endswith("000") or fake == number:
        fake = fake[:-1] + str((int(fake[-1]) + 1) % 10)
    return fake


def anonymize_text(text, key, seen):
    if not isinstance(text, str):
        return text

    def replace(match):
        chunk = match.group(0)
        positions = [i for i, ch in enumerate(chunk) if ch.isdigit()][-8:]  # numéro local, sans l'indicatif
        local = "".join(chunk[i] for i in positions)
        if local.endswith("000") or local in PLACEHOLDERS:  # nombre rond ou valeur de remplissage : pas un numéro
            return chunk
        seen.add(local)
        chars, fake = list(chunk), fake_phone(local, key)
        for i, digit in zip(positions, fake):
            chars[i] = digit
        return "".join(chars)

    return PHONE_IN_TEXT.sub(replace, text)


def anonymize_number_field(value, key, seen):
    """Champ numérique (prix, kilométrage, puissance fiscale) contenant un numéro de téléphone (8 chiffres, 2/4/5/9, non rond)."""
    if pd.isna(value):
        return value
    is_float = isinstance(value, float)
    digits = str(int(value)) if is_float else re.sub(r"\D", "", str(value))
    if len(digits) == 8 and digits[0] in "2459" and not digits.endswith("000") and digits not in PLACEHOLDERS:
        seen.add(digits)
        fake = fake_phone(digits, key)
        return float(fake) if is_float else fake
    return value


def main():
    key = load_key()
    df = pd.read_csv(SOURCE, dtype={"kilometrage": str})
    seen = set()

    for col in ["titre", "description"]:
        df[col] = df[col].map(lambda t: anonymize_text(t, key, seen))
    for col in ["prix", "kilometrage", "puissance_fiscale"]:
        df[col] = df[col].map(lambda v: anonymize_number_field(v, key, seen))

    ids = df["url"].map(lambda u: hashlib.sha256(f"{key}|url|{u}".encode()).hexdigest()[:12])
    df.insert(df.columns.get_loc("url"), "id_annonce", ids)
    df = df.drop(columns="url")

    # Vérification, cellule par cellule : aucun numéro d'origine ne subsiste, aucune url.
    # L'identifiant est exclu : c'est une empreinte calculée, qui peut contenir des chiffres par hasard.
    published_digits = set()
    for col in df.columns.drop("id_annonce"):
        for cell in df[col].dropna().astype(str):
            compact = re.sub(r"(?<=\d)[ .\-](?=\d)", "", cell)  # « 20 123 456 » -> « 20123456 »
            published_digits.update(re.findall(r"\d+", compact))
    leaked = {n for n in seen if any(n in d for d in published_digits)}
    assert not leaked, f"{len(leaked)} numéros d'origine encore présents"
    assert not df.drop(columns="id_annonce").astype(str).apply(lambda c: c.str.contains("tayara.tn/item")).any().any()
    assert ids.nunique() == pd.read_csv(SOURCE, usecols=["url"])["url"].nunique()

    df.to_csv(OUTPUT, index=False)
    print(f"{len(seen)} numéros de téléphone distincts remplacés ; url remplacées par un identifiant anonyme.")
    print(f"Écrit : {OUTPUT.relative_to(ROOT)} ({len(df):,} annonces)")


if __name__ == "__main__":
    main()
