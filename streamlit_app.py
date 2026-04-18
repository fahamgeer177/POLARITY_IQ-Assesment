from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from rag_service import ask


EXAMPLE_QUERIES = [
    "Family offices investing in AI in the United States",
    "Which family offices focus on healthcare investments?",
    "Family offices with high confidence data",
    "Which family offices focus on mining investments?",
    "Family offices investing in Gold in Japan",
]

BADGES = [
    "No Hallucinations",
    "Evidence-Linked",
    "Confidence Scored",
    "Hybrid Retrieval",
]


def _confidence_color(level: str) -> str:
    lvl = (level or "").strip().upper()
    if "HIGH" in lvl:
        return "🟩 HIGH"
    if "MEDIUM" in lvl:
        return "🟨 MEDIUM"
    if "LOW" in lvl:
        return "🟥 LOW"
    return lvl or "—"


def _chips(items: str) -> list[str]:
    if not items:
        return []
    return [s.strip() for s in items.split(",") if s.strip()]


def _copy_text_button(label: str, text: str, *, key: str) -> None:
    # Uses a tiny HTML component to copy to clipboard.
    # Intentionally copies plain text/markdown (no JSON shown in UI).
    text = str(text)
    html = f"""
    <button style="
        width: 100%;
        padding: 0.4rem 0.6rem;
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 0.5rem;
        background: white;
        cursor: pointer;
    " onclick="navigator.clipboard.writeText({json.dumps(text)});">
      {label}
    </button>
    """
    st.components.v1.html(html, height=44)


def _dataset_reference_md(row: dict) -> str:
    def v(key: str) -> str:
        val = row.get(key)
        if val is None:
            return ""
        s = str(val).strip()
        return "" if s in {"", "nan", "None"} else s

    record = v("#")
    name = v("Family Office Name")
    fo_type = v("Family Office Type")
    city = v("Family Office City")
    state = v("Family Office State / Region")
    country = v("Family Office Country")
    sectors = v("Investing Sectors")
    confidence = v("Confidence Level")
    website = v("Family Office Website URL")
    linkedin = v("Corporate LinkedIn")

    location_parts = [p for p in [city, state, country] if p]
    location = ", ".join(location_parts)

    lines: list[str] = ["**Dataset reference (verbatim fields)**"]
    if record:
        lines.append(f"- Record: {record}")
    if name:
        lines.append(f"- Name: {name}")
    if fo_type:
        lines.append(f"- Type: {fo_type}")
    if location:
        lines.append(f"- Location: {location}")
    if sectors:
        lines.append(f"- Investing Sectors: {sectors}")
    if confidence:
        lines.append(f"- Confidence Level: {confidence}")
    if website:
        lines.append(f"- Website: {website}")
    if linkedin:
        lines.append(f"- LinkedIn: {linkedin}")

    if len(lines) == 1:
        lines.append("- Not available for this record")
    return "\n".join(lines)


def _ensure_state() -> None:
    if "shortlist" not in st.session_state:
        st.session_state.shortlist = []  # list[dict]
    if "last_run" not in st.session_state:
        st.session_state.last_run = None


def main() -> None:
    st.set_page_config(
        page_title="PolarityIQ — Decision‑Grade Family Office RAG",
        layout="wide",
    )

    _ensure_state()

    # Header
    st.title("PolarityIQ — Family Office Intelligence RAG (Decision‑Grade Demo)")
    st.caption("Answers grounded strictly in validated dataset (50 records).")

    cols = st.columns(len(BADGES))
    for c, b in zip(cols, BADGES, strict=True):
        c.markdown(f"**{b}**")

    st.divider()

    # Query box + examples
    left, right = st.columns([2, 1])

    with left:
        query = st.text_area(
            "Ask a question…",
            height=96,
            placeholder="e.g., Family offices investing in AI in the United States",
        )

        ex_cols = st.columns(5)
        for i, q in enumerate(EXAMPLE_QUERIES):
            if ex_cols[i].button(q, key=f"ex_{i}"):
                st.session_state["prefill_query"] = q
                st.rerun()

        if "prefill_query" in st.session_state and not query.strip():
            query = st.session_state.pop("prefill_query")
            st.rerun()

    with right:
        st.markdown("**Mode**")
        strict_mode = st.toggle(
            "Strict mode (only exact sector matches)",
            value=True,
            help="Reduces false positives by requiring sector terms to match Investing Sectors exactly.",
        )
        k = st.slider("Top-k candidates", min_value=3, max_value=10, value=5, step=1)
        run = st.button("Search", type="primary", use_container_width=True)

    # Run
    resp = None
    if run and query.strip():
        resp = ask(query.strip(), k=k, strict_mode=strict_mode)
        st.session_state.last_run = {
            "query": query.strip(),
            "ts": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": resp.elapsed_ms,
            "retrieved_before_filter": resp.retrieved_before_filter,
            "strict_mode": resp.strict_mode,
        }

    # Results + shortlist drawer
    res_col, drawer_col = st.columns([2, 1])

    with drawer_col:
        st.subheader("Shortlist")
        shortlist = st.session_state.shortlist
        ts = None
        if st.session_state.last_run:
            ts = st.session_state.last_run.get("ts")

        st.caption(
            f"Selected: {len(shortlist)}" + (f" • Updated: {ts}" if ts else "")
        )

        if shortlist:
            for i, item in enumerate(shortlist):
                st.markdown(
                    f"**{item.get('family_office_name','—')}**  \
{item.get('location','—')}"
                )

            df = pd.DataFrame(shortlist)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Export shortlist to CSV",
                data=csv,
                file_name="shortlist.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if st.button("Clear shortlist", use_container_width=True):
                st.session_state.shortlist = []
                st.rerun()
        else:
            st.info("No shortlisted family offices yet.")

    with res_col:
        st.subheader("Results Panel")

        if run and not query.strip():
            st.warning("Enter a query to search.")

        if resp is not None:
            if resp.not_found:
                st.error("Not found in dataset")
            else:
                for idx, r in enumerate(resp.results):
                    meta_row = r.get("row") or {}

                    with st.container(border=True):
                        top = st.columns([3, 1])
                        with top[0]:
                            st.markdown(
                                f"### {r.get('family_office_name','—')}"
                                f"  \n{r.get('family_office_type','—')} • {r.get('location','—')}"
                            )
                        with top[1]:
                            st.markdown(f"**{_confidence_color(r.get('confidence_level',''))}**")

                        chips = _chips(r.get("investing_sectors") or "")
                        if chips:
                            st.caption("Investing sectors")
                            st.write(", ".join(chips))

                        st.caption("Evidence")
                        st.write(r.get("evidence") or "—")

                        actions = st.columns(3)
                        with actions[0]:
                            _copy_text_button(
                                "Copy dataset reference",
                                _dataset_reference_md(meta_row),
                                key=f"copy_{idx}",
                            )
                        with actions[1]:
                            if st.button(
                                "Add to shortlist",
                                key=f"add_{idx}",
                                use_container_width=True,
                            ):
                                # Store only the required fields in shortlist.
                                shortlist_item = {
                                    "family_office_name": r.get("family_office_name"),
                                    "family_office_type": r.get("family_office_type"),
                                    "location": r.get("location"),
                                    "investing_sectors": r.get("investing_sectors"),
                                    "confidence_level": r.get("confidence_level"),
                                    "evidence": r.get("evidence"),
                                }
                                if shortlist_item not in st.session_state.shortlist:
                                    st.session_state.shortlist.append(shortlist_item)
                                    st.rerun()
                        with actions[2]:
                            with st.expander("Dataset reference"):
                                st.markdown(_dataset_reference_md(meta_row))

    # Trust & Methodology panel
    st.divider()
    with st.expander("Trust & Methodology", expanded=False):
        st.markdown(
            """
            **How this demo avoids hallucinations**

            - **Retrieval is deterministic:** we retrieve top-k candidates via FAISS similarity search.
            - **Filtering is deterministic:** we apply keyword/constraint checks (location, sector terms, confidence) without using an LLM.
            - **LLM is optional and used for formatting only** (and is disabled by default in this UI).
            - If no record satisfies the query constraints, the system returns **Not found in dataset**.

            **Known limitations**

            - With only 50 records, semantic matching can be noisy; **Strict mode** reduces false positives.
            - Future improvements: better entity/location parsing, typed schema validation, and evaluation harness.
            """
        )

    # Footer diagnostics
    if st.session_state.last_run:
        st.caption(
            f"Latency: {st.session_state.last_run['elapsed_ms']} ms"
            f" • Retrieved before filter: {st.session_state.last_run['retrieved_before_filter']}"
            f" • Mode: {'strict' if st.session_state.last_run['strict_mode'] else 'semantic'}"
        )


if __name__ == "__main__":
    main()
