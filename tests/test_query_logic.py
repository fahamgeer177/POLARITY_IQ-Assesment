from __future__ import annotations

from polarity_iq.query_logic import extract_keywords


def test_extract_keywords_removes_generic_terms() -> None:
    kws = extract_keywords("Family offices investing in AI in the United States")
    # Domain generic words like "family"/"office" should be filtered.
    assert "family" not in kws
    assert "office" not in kws
    # But informative tokens should remain.
    assert "ai" in kws
    assert "united" in kws or "states" in kws
