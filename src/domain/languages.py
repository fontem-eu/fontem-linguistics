"""Canonical EU official languages list.

All 24 ISO-639-1 codes in the order of the EU's Official Journal. Exported so
callers (ETL backfill, consolidator enrichment rule) ask for the same set and
stay consistent over time.
"""
from __future__ import annotations

from typing import Final


EU_OFFICIAL_LANGS: Final[tuple[str, ...]] = (
    "bg",  # Bulgarian
    "cs",  # Czech
    "da",  # Danish
    "de",  # German
    "el",  # Greek
    "en",  # English
    "es",  # Spanish
    "et",  # Estonian
    "fi",  # Finnish
    "fr",  # French
    "ga",  # Irish
    "hr",  # Croatian
    "hu",  # Hungarian
    "it",  # Italian
    "lt",  # Lithuanian
    "lv",  # Latvian
    "mt",  # Maltese
    "nl",  # Dutch
    "pl",  # Polish
    "pt",  # Portuguese
    "ro",  # Romanian
    "sk",  # Slovak
    "sl",  # Slovene
    "sv",  # Swedish
)


LANG_DISPLAY_NAMES: Final[dict[str, str]] = {
    "bg": "Bulgarian", "cs": "Czech", "da": "Danish", "de": "German",
    "el": "Greek",     "en": "English", "es": "Spanish", "et": "Estonian",
    "fi": "Finnish",   "fr": "French",  "ga": "Irish",   "hr": "Croatian",
    "hu": "Hungarian", "it": "Italian", "lt": "Lithuanian", "lv": "Latvian",
    "mt": "Maltese",   "nl": "Dutch",   "pl": "Polish",   "pt": "Portuguese",
    "ro": "Romanian",  "sk": "Slovak",  "sl": "Slovene", "sv": "Swedish",
}
