import streamlit as st

from ui.chrome import apply_chrome, init_state

st.set_page_config(
    page_title="PlantVerse — Botanical Intelligence",
    page_icon=":material/eco:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "static" / "logo.svg"

if LOGO_PATH.is_file():
    st.logo(str(LOGO_PATH), size="large")
else:
    st.warning(f"Logo not found: {LOGO_PATH}")
      
apply_chrome()
init_state()

explore = st.Page(
    "app_pages/explore.py",
    title="Explore",
    icon=":material/forest:",
    url_path="explore",
    default=True,
)
analysis = st.Page(
    "app_pages/analysis.py",
    title="Plant Analysis",
    icon=":material/document_scanner:",
    url_path="analysis",
)
library = st.Page(
    "app_pages/library.py",
    title="Plant Library",
    icon=":material/menu_book:",
    url_path="library",
)
ar_explorer = st.Page(
    "app_pages/ar_explorer.py",
    title="AR Explorer",
    icon=":material/explore:",
    url_path="ar-explorer",
)
about = st.Page(
    "app_pages/about.py",
    title="About",
    icon=":material/info:",
    url_path="about",
)

page = st.navigation(
    [explore, analysis, library, ar_explorer, about],
    position="top",
)
page.run()
