# coding=utf-8
from typing import Dict


LANGUAGE_ALIASES: Dict[str, str] = {
    "auto": "Auto",
    "chinese": "Chinese",
    "zh": "Chinese",
    "cmn": "Chinese",
    "english": "English",
    "en": "English",
    "japanese": "Japanese",
    "ja": "Japanese",
    "korean": "Korean",
    "ko": "Korean",
    "german": "German",
    "de": "German",
    "french": "French",
    "fr": "French",
    "russian": "Russian",
    "ru": "Russian",
    "portuguese": "Portuguese",
    "pt": "Portuguese",
    "spanish": "Spanish",
    "es": "Spanish",
    "italian": "Italian",
    "it": "Italian",
    "hungarian": "Hungarian",
    "hu": "Hungarian",
    "hun": "Hungarian",
}


def normalize_language_key(language: str) -> str:
    if language is None:
        raise ValueError("Language must not be None.")

    key = str(language).strip().lower()
    if not key:
        raise ValueError("Language must not be empty.")

    return key


def canonicalize_language_name(language: str) -> str:
    key = normalize_language_key(language)
    return LANGUAGE_ALIASES.get(key, str(language).strip())
