"""Arabic/English text helpers shared by ingestion, retrieval and generation."""
import re

DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
ARABIC = re.compile(r"[\u0600-\u06FF]")
WORD = re.compile(r"\w+")

_AR_PREFIXES = sorted(
    ["وال", "بال", "كال", "فال", "لل", "ال", "و", "ب", "ك", "ل", "ف"], key=len, reverse=True
)
_AR_SUFFIXES = sorted(["ات", "ون", "ين", "ان", "ها", "ية", "ة", "ه"], key=len, reverse=True)


def normalize_ar(text: str) -> str:
    """Remove diacritics and tatweel, and unify alef variants."""
    text = DIACRITICS.sub("", text)
    text = text.replace("\u0640", "")
    return re.sub("[إأآ]", "ا", text)


def light_stem(token: str) -> str:
    """Very light Arabic stemmer: strip one common prefix and one common suffix.

    Applied identically to documents and queries, so over-trimming is harmless.
    Non-Arabic tokens are returned unchanged.
    """
    if not ARABIC.match(token):
        return token
    for p in _AR_PREFIXES:
        if token.startswith(p) and len(token) - len(p) >= 3:
            token = token[len(p):]
            break
    for s in _AR_SUFFIXES:
        if token.endswith(s) and len(token) - len(s) >= 3:
            token = token[: -len(s)]
            break
    return token


def tokenize(text: str) -> list[str]:
    """Tokenizer for BM25: normalize, lowercase, split on word characters, stem."""
    return [light_stem(t) for t in WORD.findall(normalize_ar(text).lower())]


def detect_lang(text: str) -> str:
    return "Arabic" if ARABIC.search(text) else "English"
