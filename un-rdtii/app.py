"""
app.py — RDTII Regulatory Analyzer
Streamlit web application for analyzing digital trade regulations.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

# Path bootstrap
# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crawler import SOURCES, download_document, get_file_info
from extractor import extract_text
from mapper import INDICATORS, map_chunks_with_candidates, load_results, _get_classifier

# Page configuration
st.set_page_config(
    page_title="RDTII Regulatory Analyzer",
    page_icon="[RA]",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session state
def _init_state():
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    if "download_status" not in st.session_state:
        st.session_state.download_status = {}
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {}
    if "analysis_candidates" not in st.session_state:
        st.session_state.analysis_candidates = {}
    # Pre-load any existing results from disk
    for country in SOURCES:
        if country not in st.session_state.analysis_results:
            saved = load_results(country)
            if saved:
                st.session_state.analysis_results[country] = saved
    # Sync download status with files on disk
    for country in SOURCES:
        info = get_file_info(country)
        if info["downloaded"]:
            st.session_state.download_status[country] = "downloaded"

_init_state()

# Sidebar rendering
def _render_sidebar():
    with st.sidebar:
        st.title("RDTII Analyzer")
        st.caption("AI-powered digital trade regulation mapping")
        st.divider()

        pages = ["Home", "Document Discovery", "Analysis", "Comparison Table", "World Map"]
        for p in pages:
            if st.button(p, key=f"nav_{p}", use_container_width=True,
                         type="primary" if st.session_state.page == p else "secondary"):
                st.session_state.page = p
                st.rerun()

        st.divider()
        st.markdown("**Document Status**")
        for country in SOURCES:
            info = get_file_info(country)
            status = st.session_state.download_status.get(country, "")
            if status == "downloaded" or info["downloaded"]:
                st.markdown(f"[OK] {country}")
            else:
                st.markdown(f"[--] {country}")

        st.divider()
        st.markdown("**Analysis Status**")
        for country in SOURCES:
            if country in st.session_state.analysis_results:
                n = len(st.session_state.analysis_results[country])
                st.markdown(f"[OK] {country} ({n} indicators)")
            else:
                st.markdown(f"[--] {country}")

        st.divider()
        with st.expander("[i] About RDTII"):
            st.markdown(
                """
**RDTII** (Regulatory & Digital Trade Indicators Index) maps
national regulations to standardized indicators across two pillars:

- **Pillar 6** — Cross-border data flows
- **Pillar 7** — Data protection & cybersecurity

Developed for the **UN ESCAP** Asia-Pacific hackathon on digital trade governance.
                """
            )


# Page: Home
def _page_home():
    st.title("RDTII Regulatory Analyzer")
    st.subheader("AI-powered digital trade regulation mapping for Asia-Pacific economies")
    st.divider()

    country_info = {
        "Thailand": {
            "flag": "🇹🇭",
            "desc": (
                "Thailand's Personal Data Protection Act (PDPA, 2019) is modeled on GDPR. "
                "It establishes data controller obligations, consent requirements, and cross-border "
                "transfer rules with adequacy and safeguard mechanisms."
            ),
        },
        "Vietnam": {
            "flag": "🇻🇳",
            "desc": (
                "Vietnam's Decree 13/2023 on Personal Data Protection introduced strict data "
                "localization provisions and cross-border transfer consent requirements, "
                "with significant compliance obligations for data processors."
            ),
        },
        "Indonesia": {
            "flag": "🇮🇩",
            "desc": (
                "Indonesia's Government Regulation PP 71/2019 sets out electronic systems "
                "operation rules including data localization for strategic sectors and "
                "requirements for government access to data held by private operators."
            ),
        },
    }

    cols = st.columns(3)
    for col, (country, info) in zip(cols, country_info.items()):
        with col:
            with st.container(border=True):
                st.markdown(f"## {country}")
                st.markdown(info["desc"])
                analyzed = country in st.session_state.analysis_results
                downloaded = get_file_info(country)["downloaded"]
                if analyzed:
                    st.success(f"[OK] Analyzed — {len(st.session_state.analysis_results[country])} indicators found")
                elif downloaded:
                    st.info("[OK] Downloaded — ready to analyze")
                else:
                    st.caption("Not yet downloaded")

    st.divider()
    if st.button("[>>] Get Started", type="primary", use_container_width=False):
        st.session_state.page = "Document Discovery"
        st.rerun()

    st.divider()
    with st.expander("How this tool works"):
        st.markdown(
            """
1. **Download** — Automatically fetches PDF regulatory documents from official sources.
2. **Extract** — Pulls text from each PDF page; falls back to OCR for scanned pages.
3. **Analyze** — Runs `facebook/bart-large-mnli` zero-shot classification locally on your machine —
   no API key, no internet required after first download.
4. **Compare** — Displays a side-by-side indicator table for all three countries.

The AI model assigns each text chunk a confidence score against all 10 RDTII indicators.
Matches scoring above 0.5 are kept and classified as *Primary* (>0.85), *Contextual* (0.70–0.85),
or *Implicit* (0.50–0.70).
            """
        )

# Page: Document Discovery
def _page_discovery():
    st.title("Document Discovery")
    st.markdown("Download the source regulatory documents for each country.")
    st.divider()

    def _do_download(country: str):
        prog_bar = st.progress(0.0, text=f"Downloading {country}...")
        try:
            download_document(
                country,
                progress_callback=lambda p: prog_bar.progress(min(p, 1.0), text=f"Downloading {country}... {p*100:.0f}%"),
            )
            prog_bar.progress(1.0, text=f"{country} downloaded [OK]")
            st.session_state.download_status[country] = "downloaded"
            st.success(f"[OK] {country} saved to data/documents/{country.lower()}.pdf")
        except RuntimeError as exc:
            prog_bar.empty()
            st.error(str(exc))

    if st.button("[v] Download All Documents", type="primary"):
        for country in SOURCES:
            _do_download(country)
        st.rerun()

    st.divider()

    for country in SOURCES:
        info = get_file_info(country)
        with st.container(border=True):
            col_info, col_btn = st.columns([4, 1])
            with col_info:
                st.markdown(f"### {country}")
                st.markdown(f"**Document:** {info['document_name']}")
                safe_url = info['url'].replace("[", "%5B").replace("]", "%5D").replace(" ", "%20")
                st.markdown(f"**URL:** [{info['url']}]({safe_url})")
                if info["downloaded"]:
                    st.success(f"[OK] Downloaded — {info['file_size_kb']} KB")
                else:
                    st.caption("Not yet downloaded")
            with col_btn:
                st.write("")
                if st.button(f"Download {country}", key=f"dl_{country}"):
                    _do_download(country)
                    st.rerun()

# Page: Analysis
MATCH_COLORS = {
    "Primary": "[H]",
    "Contextual": "[M]",
    "Implicit": "[L]",
}

DEMO_MAP_DATA = {
    "Thailand": {
        "lat": 15.8700,
        "lon": 100.9925,
        "iso3": "THA",
        "indicator": "6.4 — Conditional flow regimes",
        "summary": (
            "Personal data transfer overseas is allowed if the destination country "
            "or organization has adequate data protection standards."
        ),
        "evidence": "PDPA Section 28",
        "source": "Personal Data Protection Act B.E. 2562",
        "confidence": "Demo data",
    },
    "Vietnam": {
        "lat": 14.0583,
        "lon": 108.2772,
        "iso3": "VNM",
        "indicator": "6.2 — Local storage requirements",
        "summary": (
            "Certain data may be required to be stored locally under cybersecurity "
            "and personal data rules."
        ),
        "evidence": "Decree 13 / Cybersecurity Law reference",
        "source": "Vietnam Decree 13 / Cybersecurity Law",
        "confidence": "Demo data",
    },
    "Indonesia": {
        "lat": -0.7893,
        "lon": 113.9213,
        "iso3": "IDN",
        "indicator": "7.4 — DPIA or DPO requirements",
        "summary": (
            "Personal data protection obligations may require controllers to follow "
            "compliance and accountability requirements."
        ),
        "evidence": "PDP Law / PP 71 reference",
        "source": "Indonesia PDP Law / PP 71",
        "confidence": "Demo data",
    },
}


def _go_to_world_map():
    st.session_state.page = "World Map"


def _map_rows() -> pd.DataFrame:
    rows = []
    for country, demo in DEMO_MAP_DATA.items():
        row = {
            "country": country,
            "iso3": demo["iso3"],
            "lat": demo["lat"],
            "lon": demo["lon"],
            "indicator": demo["indicator"],
            "summary": demo["summary"],
            "evidence": demo["evidence"],
            "source": demo["source"],
            "confidence": demo["confidence"],
            "data_type": "Demo data",
        }

        results = st.session_state.analysis_results.get(country)
        if results:
            ind_id, match = sorted(results.items())[0]
            row.update({
                "indicator": f"{ind_id} — {match['indicator_name']}",
                "summary": match["exact_quote"][:220],
                "evidence": f"Page {match['page_number']}",
                "source": SOURCES[country]["name"],
                "confidence": f"{match['confidence'] * 100:.1f}%",
                "data_type": "AI result",
            })
        rows.append(row)

    return pd.DataFrame(rows)

@st.cache_resource(show_spinner="Loading AI model (first run takes a few minutes)...")
def _load_model():
    return _get_classifier()


def _candidate_count(candidates: dict) -> int:
    return sum(len(matches) for matches in candidates.values())


def _render_candidate_evidence(candidates: dict, country: str):
    if not candidates or _candidate_count(candidates) == 0:
        return

    st.divider()
    st.subheader("Candidate evidence for review")
    st.warning("These are low-confidence candidates and need human verification. They are not confirmed matches.")

    source = "Uploaded PDF" if country == "custom_upload" else SOURCES.get(country, {}).get("name", country)

    for ind_id in sorted(candidates.keys()):
        matches = [match for match in candidates[ind_id] if match.get("confidence", 0) < 0.5]
        if not matches:
            continue

        indicator_name = matches[0].get("indicator_name", INDICATORS.get(ind_id, "").split(" — ")[0])
        with st.container(border=True):
            st.markdown(f"**{ind_id} — {indicator_name}**")
            st.caption(f"Source: {source}")

            for match in matches:
                conf_pct = f"{match['confidence'] * 100:.1f}%"
                page = match.get("page_number", "unknown")
                quote = match.get("exact_quote", "")
                snippet = quote[:400] + ("..." if len(quote) > 400 else "")
                st.markdown(f"**Candidate confidence:** {conf_pct} | **Page:** {page}")
                st.markdown(f"> {snippet}")


def _page_analysis():
    st.title("Analysis")
    st.markdown("Select a country and run AI analysis to map regulations to RDTII indicators.")
    st.divider()

    # Tab: Pre-loaded countries vs. Custom PDF upload
    tab_preset, tab_upload = st.tabs(["Pre-loaded Documents", "Upload Custom PDF"])
    
    with tab_preset:
        country = st.selectbox("Select Country", list(SOURCES.keys()))
        info = get_file_info(country)

        if not info["downloaded"]:
            st.warning(f"[!] {country} PDF not downloaded yet. Go to Document Discovery first.")
            return

        if st.button("[>] Run AI Analysis", type="primary"):
            status_box = st.empty()

            # Step 1: Extract
            status_box.info("[*] Extracting text from PDF...")
            prog1 = st.progress(0.0)
            extraction = extract_text(
                info["local_path"],
                progress_callback=lambda p: prog1.progress(min(p, 1.0)),
            )
            prog1.empty()

            if extraction.error:
                st.error(f"Extraction error: {extraction.error}")
                return

            status_box.info(f"[OK] Extracted {len(extraction.chunks)} chunks from {extraction.page_count} pages. Loading AI model...")

            # Step 2: Load model (cached)
            _load_model()

            # Step 3: Map
            status_box.info(f"[*] Analyzing {len(extraction.chunks)} chunks with AI... (this may take a few minutes)")
            prog2 = st.progress(0.0)

            results, candidates = map_chunks_with_candidates(
                extraction.chunks,
                country,
                progress_callback=lambda p: prog2.progress(min(p, 1.0)),
            )
            prog2.empty()

            st.session_state.analysis_results[country] = results
            st.session_state.analysis_candidates[country] = candidates
            status_box.success(f"[OK] Analysis complete — {len(results)} indicators matched!")
            st.button("View Similar Regulations on Map", on_click=_go_to_world_map)
            st.rerun()

        # Display results
        if country in st.session_state.analysis_results:
            results = st.session_state.analysis_results[country]
            candidates = st.session_state.analysis_candidates.get(country, {})
            st.divider()
            st.subheader(f"Results for {country}")

            if not results:
                st.info("No indicators matched above the confidence threshold (0.5).")
                st.button("View Similar Regulations on Map", on_click=_go_to_world_map)
            else:
                st.button("View Similar Regulations on Map", on_click=_go_to_world_map)

                col_left, col_right = st.columns(2)
                col_left.markdown("**Original Source Text**")
                col_right.markdown("**Mapped Indicator**")
                st.divider()

                for ind_id in sorted(results.keys()):
                    match = results[ind_id]
                    icon = MATCH_COLORS.get(match["match_level"], "[?]")
                    conf_pct = f"{match['confidence'] * 100:.1f}%"

                    col_l, col_r = st.columns(2)
                    with col_l:
                        with st.container(border=True):
                            st.caption(f"Page {match['page_number']}")
                            st.markdown(f"> {match['exact_quote'][:400]}{'...' if len(match['exact_quote']) > 400 else ''}")
                    with col_r:
                        with st.container(border=True):
                            st.markdown(f"{icon} **{ind_id} — {match['indicator_name']}**")
                            st.markdown(f"Match level: **{match['match_level']}**")
                            st.markdown(f"Confidence: **{conf_pct}**")
                            st.caption(f"Page {match['page_number']}")

            _render_candidate_evidence(candidates, country)

        elif info["downloaded"]:
            st.info("Click '[>] Run AI Analysis' to start.")
    
    with tab_upload:
        st.markdown("Upload a custom PDF document to analyze against RDTII indicators.")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file:
            temp_path = Path("/tmp") / uploaded_file.name
            temp_path.write_bytes(uploaded_file.getbuffer())
            
            if st.button("[>] Analyze Uploaded PDF", type="primary", key="analyze_upload"):
                status_box = st.empty()
                
                # Step 1: Extract
                status_box.info("[*] Extracting text from PDF...")
                prog1 = st.progress(0.0)
                extraction = extract_text(
                    str(temp_path),
                    progress_callback=lambda p: prog1.progress(min(p, 1.0)),
                )
                prog1.empty()

                if extraction.error:
                    st.error(f"Extraction error: {extraction.error}")
                    return

                status_box.info(f"[OK] Extracted {len(extraction.chunks)} chunks from {extraction.page_count} pages. Loading AI model...")

                # Step 2: Load model
                _load_model()

                # Step 3: Map
                status_box.info(f"[*] Analyzing {len(extraction.chunks)} chunks with AI... (this may take a few minutes)")
                prog2 = st.progress(0.0)

                results, candidates = map_chunks_with_candidates(
                    extraction.chunks,
                    "custom_upload",
                    progress_callback=lambda p: prog2.progress(min(p, 1.0)),
                )
                prog2.empty()

                st.session_state.analysis_results["custom_upload"] = results
                st.session_state.analysis_candidates["custom_upload"] = candidates
                status_box.success(f"[OK] Analysis complete — {len(results)} indicators matched!")
                st.button("View Similar Regulations on Map", on_click=_go_to_world_map)
                st.rerun()
            
            # Display results
            if "custom_upload" in st.session_state.analysis_results:
                results = st.session_state.analysis_results["custom_upload"]
                candidates = st.session_state.analysis_candidates.get("custom_upload", {})
                st.divider()
                st.subheader(f"Analysis Results")

                if not results:
                    st.info("No indicators matched above the confidence threshold (0.5).")
                    st.button("View Similar Regulations on Map", on_click=_go_to_world_map)
                else:
                    st.button("View Similar Regulations on Map", on_click=_go_to_world_map)

                    col_left, col_right = st.columns(2)
                    col_left.markdown("**Original Source Text**")
                    col_right.markdown("**Mapped Indicator**")
                    st.divider()

                    for ind_id in sorted(results.keys()):
                        match = results[ind_id]
                        icon = MATCH_COLORS.get(match["match_level"], "[?]")
                        conf_pct = f"{match['confidence'] * 100:.1f}%"

                        col_l, col_r = st.columns(2)
                        with col_l:
                            with st.container(border=True):
                                st.caption(f"Page {match['page_number']}")
                                st.markdown(f"> {match['exact_quote'][:400]}{'...' if len(match['exact_quote']) > 400 else ''}")
                        with col_r:
                            with st.container(border=True):
                                st.markdown(f"{icon} **{ind_id} — {match['indicator_name']}**")
                                st.markdown(f"Match level: **{match['match_level']}**")
                                st.markdown(f"Confidence: **{conf_pct}**")
                                st.caption(f"Page {match['page_number']}")

                _render_candidate_evidence(candidates, "custom_upload")

# Page: Comparison Table
def _build_comparison_df() -> pd.DataFrame:
    all_results = st.session_state.analysis_results
    demo_cells = {
        ("Thailand", "6.4"): f"Demo data: {DEMO_MAP_DATA['Thailand']['evidence']}",
        ("Vietnam", "6.2"): f"Demo data: {DEMO_MAP_DATA['Vietnam']['evidence']}",
        ("Indonesia", "7.4"): f"Demo data: {DEMO_MAP_DATA['Indonesia']['evidence']}",
    }
    rows = []
    for ind_id, ind_label in sorted(INDICATORS.items()):
        ind_name = ind_label.split(" — ")[0]
        row = {"Indicator": f"{ind_id}: {ind_name}"}
        for country in SOURCES:
            if country in all_results and ind_id in all_results[country]:
                m = all_results[country][ind_id]
                row[country] = f"{m['match_level']} ({m['confidence']*100:.0f}%) p.{m['page_number']}"
            elif (country, ind_id) in demo_cells:
                row[country] = demo_cells[(country, ind_id)]
            else:
                row[country] = "—"
        rows.append(row)

    # Summary row
    summary = {"Indicator": "TOTAL MATCHED"}
    for country in SOURCES:
        if country in all_results:
            summary[country] = str(len(all_results[country]))
        else:
            summary[country] = "0"
    rows.append(summary)

    return pd.DataFrame(rows)


def _style_cell(val: str) -> str:
    if "Primary" in val:
        return "background-color: #d4edda; color: #155724;"
    if "Contextual" in val:
        return "background-color: #fff3cd; color: #856404;"
    if "Implicit" in val:
        return "background-color: #fde8d8; color: #7d3c00;"
    return ""


def _page_comparison():
    st.title("Comparison Table")
    st.markdown("Side-by-side comparison of all 10 RDTII indicators across Thailand, Vietnam, and Indonesia.")
    st.info("Rows labeled **Demo data** are fixed hackathon demo examples, not fully verified AI results.")
    st.divider()

    if not st.session_state.analysis_results:
        st.info("No countries have been analyzed yet. Run analysis from the Analysis page first.")
        return

    df = _build_comparison_df()

    country_cols = [c for c in df.columns if c != "Indicator"]
    styled = df.style.map(_style_cell, subset=country_cols)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        csv_data = df.to_csv(index=False)
        st.download_button(
            "[v] Download as CSV",
            data=csv_data,
            file_name="rdtii_comparison.csv",
            mime="text/csv",
        )

    with col2:
        json_data = json.dumps(
            {
                country: st.session_state.analysis_results.get(country, {})
                for country in SOURCES
            },
            indent=2,
            ensure_ascii=False,
        )
        st.download_button(
            "[v] Download as JSON",
            data=json_data,
            file_name="rdtii_results.json",
            mime="application/json",
        )

    # Legend
    st.divider()
    st.markdown("**Legend:**")
    lcols = st.columns(4)
    lcols[0].success("[H] Primary (>85%)")
    lcols[1].warning("[M] Contextual (70–85%)")
    lcols[2].error("[L] Implicit (50–70%)")
    lcols[3].info("[--] Not matched")


# Page: World Map
def _page_world_map():
    st.title("World Map")
    st.markdown("Highlighted countries show regulations with similar RDTII themes.")
    st.info("Rows labeled **Demo data** are fixed hackathon demo examples, not fully verified AI results.")
    st.divider()

    df = _map_rows()

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[lon, lat]",
        get_radius=350000,
        get_fill_color="[22, 122, 118, 180]",
        get_line_color="[255, 255, 255]",
        line_width_min_pixels=2,
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=5,
        longitude=105,
        zoom=3,
        pitch=0,
    )

    tooltip = {
        "html": (
            "<b>{country}</b><br/>"
            "<b>{data_type}</b><br/>"
            "Indicator: {indicator}<br/>"
            "Evidence: {evidence}<br/>"
            "Source: {source}<br/>"
            "Confidence: {confidence}"
        ),
        "style": {
            "backgroundColor": "#1f2937",
            "color": "white",
            "fontSize": "12px",
        },
    }

    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=view_state,
            layers=[layer],
            tooltip=tooltip,
        ),
        use_container_width=True,
    )

    st.dataframe(
        df[["country", "data_type", "indicator", "evidence", "source", "confidence"]],
        use_container_width=True,
        hide_index=True,
    )


# Main application

# Render

# Render sidebar
_render_sidebar()

# Route to selected page
page = st.session_state.page
if page == "Home":
    _page_home()
elif page == "Document Discovery":
    _page_discovery()
elif page == "Analysis":
    _page_analysis()
elif page == "Comparison Table":
    _page_comparison()
elif page == "World Map":
    _page_world_map()
