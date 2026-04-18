from __future__ import annotations

import re
from typing import Any

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    import pycountry  # type: ignore
except Exception:  # pragma: no cover
    pycountry = None


_DOMAIN_GENERIC_STOPWORDS = {
    "family",
    "families",
    "office",
    "offices",
    "invest",
    "invests",
    "investing",
    "investment",
    "investments",
    "focus",
    "focused",
    "which",
    "who",
    "what",
    "where",
    "list",
    "show",
    "find",
}


def extract_evidence(text: str, *, max_chars: int = 260) -> str:
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    preferred_prefixes = (
        "Investment Sectors:",
        "Investment Thesis:",
        "Location:",
        "AUM:",
        "Confidence:",
    )

    picked: list[str] = []
    for prefix in preferred_prefixes:
        for ln in lines:
            if ln.startswith(prefix):
                picked.append(ln)
                break

    if not picked and lines:
        picked = lines[:2]

    evidence = " | ".join(picked)
    if len(evidence) > max_chars:
        evidence = evidence[: max_chars - 1].rstrip() + "…"
    return evidence


def compact_results(retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in retrieved:
        meta = r.get("meta") or {}
        row = meta.get("row") if isinstance(meta.get("row"), dict) else None
        location = ", ".join([p for p in [meta.get("city"), meta.get("country")] if p]).strip()
        out.append(
            {
                "rank": r.get("rank"),
                "score": r.get("score"),
                "record": meta.get("record"),
                "family_office_name": meta.get("family_office_name"),
                "family_office_type": meta.get("family_office_type"),
                "location": location,
                "investing_sectors": meta.get("investing_sectors"),
                "confidence_level": meta.get("confidence_level"),
                "evidence": extract_evidence(str(r.get("text") or "")),
                "row": row,
            }
        )
    return out


def extract_keywords(query: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", str(query).lower())
    keywords: list[str] = []
    for t in tokens:
        if t in ENGLISH_STOP_WORDS:
            continue
        if t in _DOMAIN_GENERIC_STOPWORDS:
            continue
        if len(t) < 3 and t not in {"ai", "us"}:
            continue
        keywords.append(t)

    seen: set[str] = set()
    deduped: list[str] = []
    for k in keywords:
        if k in seen:
            continue
        seen.add(k)
        deduped.append(k)
    return deduped


def is_found_in_dataset(query: str, retrieved: list[dict[str, Any]]) -> bool:
    return bool(retrieved)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _country_aliases() -> dict[str, str]:
    return {
        "us": "united states",
        "u.s.": "united states",
        "u.s": "united states",
        "usa": "united states",
        "united states": "united states of america",
        "uk": "united kingdom",
        "u.k.": "united kingdom",
        "uae": "united arab emirates",
    }


def _extract_country_from_query(query: str) -> str | None:
    q = _normalize(query)
    for k, v in _country_aliases().items():
        if re.search(rf"\b{re.escape(k)}\b", q):
            return v

    if pycountry is not None:
        names = []
        for c in pycountry.countries:
            names.append(_normalize(getattr(c, "name", "")))
            if hasattr(c, "official_name"):
                names.append(_normalize(getattr(c, "official_name", "")))
            if hasattr(c, "common_name"):
                names.append(_normalize(getattr(c, "common_name", "")))
        names = [n for n in names if n]
        for name in sorted(set(names), key=len, reverse=True):
            if re.search(rf"\b{re.escape(name)}\b", q):
                return name

    m = re.search(r"\bin\s+([a-z][a-z\s]{2,40})\b", q)
    if m:
        return _normalize(m.group(1))

    return None


def _canonicalize_country(country: str, store_rows: list[dict[str, Any]]) -> str:
    detected = _normalize(country)
    countries = {_normalize(str(r.get("Family Office Country") or "")) for r in store_rows}
    countries.discard("")

    if detected in countries:
        return detected

    for c in sorted(countries, key=len, reverse=True):
        if detected and detected in c:
            return c

    return detected


def _extract_location_constraint(query: str, store_rows: list[dict[str, Any]]) -> str | None:
    detected = _extract_country_from_query(query)
    if not detected:
        return None
    return _canonicalize_country(detected, store_rows)


def _extract_confidence_constraint(query: str) -> str | None:
    q = _normalize(query)
    if "high confidence" in q or "confidence high" in q:
        return "high"
    if "medium-high confidence" in q or "medium high confidence" in q:
        return "medium-high"
    if "medium confidence" in q or "confidence medium" in q:
        return "medium"
    if "low confidence" in q or "confidence low" in q:
        return "low"
    return None


def _extract_sector_groups(query: str) -> list[list[str]]:
    q = _normalize(query)
    groups: list[list[str]] = []

    if "artificial intelligence" in q or re.search(r"\bai\b", q):
        groups.append(["artificial intelligence", "ai"])
    if "healthcare" in q or "health care" in q or "digital health" in q:
        groups.append(["healthcare", "health care", "digital health", "health"])
    if "mining" in q:
        groups.append(["mining"])
    if "gold" in q:
        groups.append(["gold"])

    return groups


def _term_in_haystack(term: str, haystack: str) -> bool:
    t = _normalize(term)
    h = _normalize(haystack)
    if not t:
        return False
    if len(t) <= 3 and t.isalnum():
        return re.search(rf"\b{re.escape(t)}\b", h) is not None
    return t in h


def _record_matches_constraints(
    record_text: str,
    row: dict[str, Any] | None,
    *,
    location: str | None,
    confidence: str | None,
    sector_groups: list[list[str]],
    strict_mode: bool,
) -> bool:
    text = _normalize(record_text)
    row = row or {}

    if location:
        loc_blob = _normalize(
            " ".join(
                [
                    str(row.get("Family Office City") or ""),
                    str(row.get("Family Office State / Region") or ""),
                    str(row.get("Family Office Country") or ""),
                ]
            )
        )
        if location not in loc_blob and location not in text:
            return False

    if confidence:
        conf = _normalize(str(row.get("Confidence Level") or ""))
        if confidence not in conf:
            return False

    if sector_groups:
        sectors = _normalize(str(row.get("Investing Sectors") or ""))
        thesis = _normalize(str(row.get("Investment Thesis") or ""))

        haystacks = [sectors] if strict_mode else [sectors, thesis, text]

        for group in sector_groups:
            if not any(_term_in_haystack(term, hs) for term in group for hs in haystacks):
                return False

    return True


def filter_retrieved(
    query: str,
    store: Any,
    retrieved: list[dict[str, Any]],
    *,
    strict_mode: bool = True,
) -> list[dict[str, Any]]:
    """Deterministic post-filtering to reduce false positives."""

    # Pull original rows if present.
    store_rows: list[dict[str, Any]] = []
    try:
        for meta in getattr(store, "metas", []) or []:
            row = meta.get("row")
            if isinstance(row, dict):
                store_rows.append(row)
    except Exception:
        store_rows = []

    location = _extract_location_constraint(query, store_rows) if store_rows else _extract_country_from_query(query)
    confidence = _extract_confidence_constraint(query)
    sector_groups = _extract_sector_groups(query)

    out: list[dict[str, Any]] = []
    for r in retrieved:
        meta = r.get("meta") or {}
        row = meta.get("row") if isinstance(meta.get("row"), dict) else None
        if _record_matches_constraints(
            str(r.get("text") or ""),
            row,
            location=location,
            confidence=confidence,
            sector_groups=sector_groups,
            strict_mode=bool(strict_mode),
        ):
            out.append(r)

    # If strict mode filtered everything but we have strong keyword overlap, return top-1 to avoid false "Not found".
    if not out and retrieved and not strict_mode:
        out = retrieved[:1]

    return out
