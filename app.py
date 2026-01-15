import joblib, re, nltk, time
from nltk.corpus import stopwords
import streamlit as st
import numpy as np

# ------------------------------
# NLTK Setup
# ------------------------------
nltk.download("stopwords")

# ------------------------------
# APP NAME
# ------------------------------
APP_NAME = "DetectiveNews"

# ------------------------------
# CREATOR DETAILS
# ------------------------------
CREATOR_NAME = "SINDHU MUDALIYAR D"
CREATOR_EMAIL = "sindhudeva489@gmail.com"
CREATOR_ROLE = "AI/ML Student • Fake News Detection Project"
MY_LINKEDIN = "https://www.linkedin.com/in/sindhu-mudaliyar-d-a05450290"
MY_GITHUB = "https://github.com/"

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title=APP_NAME,
    page_icon="logo.png",   # ✅ YOUR LOGO
    layout="wide",
)

# ------------------------------
# Load Model + Vectorizer
# ------------------------------
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ------------------------------
# Session defaults
# ------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# =========================================================
# ✅ THEME TOGGLE (TOP RIGHT)
# =========================================================
c1, c2 = st.columns([8, 2])
with c2:
    is_dark = st.toggle("Dark Mode", value=(st.session_state.theme == "dark"))
st.session_state.theme = "dark" if is_dark else "light"

# =========================================================
# ✅ COLORS (Lavender Theme)
# =========================================================
if st.session_state.theme == "dark":
    BG = "radial-gradient(circle at top, #1b0f2e 0%, #0b0615 55%, #07040f 100%)"
    TITLE = "#f5f3ff"
    SUBTXT = "#c4b5fd"
    INPUT_BG = "rgba(255,255,255,0.10)"
    INPUT_TXT = "#ffffff"
    BTN_BG = "linear-gradient(90deg, #7c3aed, #a855f7, #c084fc)"
    BTN_TXT = "#ffffff"
    BAR_BG = "rgba(255,255,255,0.20)"
    CARD_BG = "rgba(255,255,255,0.08)"
    CARD_BORDER = "rgba(255,255,255,0.18)"
else:
    BG = "linear-gradient(135deg, #fdfbff 0%, #f3e8ff 40%, #ede9fe 100%)"
    TITLE = "#2e1065"
    SUBTXT = "#5b21b6"
    INPUT_BG = "#ffffff"
    INPUT_TXT = "#2e1065"
    BTN_BG = "linear-gradient(90deg, #7c3aed, #a855f7, #c084fc)"
    BTN_TXT = "#ffffff"
    BAR_BG = "rgba(0,0,0,0.08)"
    CARD_BG = "rgba(255,255,255,0.92)"
    CARD_BORDER = "rgba(0,0,0,0.08)"

# =========================================================
# ✅ GLOBAL CSS
# =========================================================
st.markdown(
    f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
    background: {BG};
    color: {TITLE};
    font-family: "Segoe UI", sans-serif;
}}
header, footer {{visibility:hidden;}}

.block-container {{
    max-width: 980px;
    padding-top: 25px;
}}

.big-title {{
    text-align:center;
    font-size:48px;
    font-weight:900;
    margin-top:55px;
    margin-bottom:15px;
    color:{TITLE};
}}

.sub-title {{
    text-align:center;
    font-size:18px;
    color:{SUBTXT};
    margin-bottom:30px;
}}

textarea {{
    border-radius:18px !important;
    font-size:20px !important;
    padding:20px !important;
    background:{INPUT_BG} !important;
    color:{INPUT_TXT} !important;
    border:1px solid rgba(255,255,255,0.18) !important;
}}
textarea:focus {{
    border:2px solid #a855f7 !important;
    outline:none !important;
}}

div.stButton > button {{
    border-radius:20px !important;
    font-size:22px !important;
    font-weight:900 !important;
    padding:14px 38px !important;
    background:{BTN_BG} !important;
    color:{BTN_TXT} !important;
    border:none !important;
    box-shadow:0px 12px 25px rgba(168,85,247,0.35) !important;
    transition:0.25s ease-in-out;
}}
div.stButton > button:hover {{
    transform: scale(1.05) !important;
}}

.result-label {{
    text-align:center;
    font-size:42px;
    font-weight:900;
    margin-top:25px;
}}

.progress-box {{
    width:420px;
    height:20px;
    background:{BAR_BG};
    border-radius:16px;
    margin:15px auto 10px auto;
    overflow:hidden;
}}
.progress-fill {{
    height:100%;
    border-radius:16px;
    background:{BTN_BG};
}}

.conf-text {{
    text-align:center;
    font-size:22px;
    margin-top:10px;
    opacity:0.95;
}}

.info-card {{
    background:{CARD_BG};
    border:1px solid {CARD_BORDER};
    padding:18px 20px;
    border-radius:18px;
    margin-top:18px;
    box-shadow:0px 10px 22px rgba(0,0,0,0.25);
}}

.creator-card {{
    position:fixed;
    bottom:18px;
    left:18px;
    width:265px;
    background:rgba(255,255,255,0.06);
    border:1px solid rgba(255,255,255,0.15);
    border-radius:16px;
    padding:12px;
    display:flex;
    gap:10px;
    align-items:center;
    backdrop-filter: blur(10px);
}}
</style>
""",
    unsafe_allow_html=True
)

# =========================================================
# ✅ SIDEBAR (ONE MENU ONLY)
# =========================================================
st.sidebar.image("logo.png", width=70)
st.sidebar.markdown(f"<div style='font-size:22px;font-weight:900;margin-bottom:10px;'>{APP_NAME}</div>",
                    unsafe_allow_html=True)

st.sidebar.text_input("🔍 Search", placeholder="Search")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dashboard",
        "🔗 Integrations",
        "📌 Events",
        "🧾 Reporting",
        "👥 Users",
        "🔑 API Keys",
        "⚙️ Settings"
    ],
    label_visibility="collapsed"
)

# Creator card bottom
st.sidebar.markdown(
    f"""
    <div class="creator-card">
        <img src="data:image/png;base64," style="display:none;"/>
        <div style="
            width:44px;height:44px;border-radius:50%;
            background:linear-gradient(135deg,#7c3aed,#a855f7,#c084fc);
            display:flex;align-items:center;justify-content:center;
            font-weight:900;color:white;font-size:18px;
        ">S</div>
        <div style="line-height:1.2;">
            <div style="font-weight:900;font-size:13px;color:#f5f3ff;">{CREATOR_NAME}</div>
            <div style="font-size:12px;color:#c4b5fd;">{CREATOR_EMAIL}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# ✅ HELPERS
# =========================================================
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower().split()
    stop_words = set(stopwords.words("english"))
    text = [word for word in text if word not in stop_words]
    return " ".join(text)

def show_default_page(title, points):
    st.markdown(f"## {title}")
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.write("✅ Default features included in this section:")
    for p in points:
        st.write("•", p)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# ✅ PAGES
# =========================================================
if page == "🏠 Home":
    st.markdown('<div class="big-title">Check if news is real or fake!</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter an article below, then click Predict.</div>', unsafe_allow_html=True)

    news_text = st.text_area("", placeholder="Enter your article here...", height=160)

    left, mid, right = st.columns([2, 1, 2])
    with mid:
        predict = st.button("Predict")

    if predict:
        if news_text.strip() == "":
            st.warning("Please enter some news content.")
        else:
            with st.spinner("Analyzing..."):
                time.sleep(0.8)

            cleaned = clean_text(news_text)
            vect = vectorizer.transform([cleaned])
            pred = model.predict(vect)[0]
            proba = model.predict_proba(vect)[0]

            real_prob = proba[1] * 100
            label = "REAL" if pred == 1 else "FAKE"

            st.markdown(f'<div class="result-label">{label}</div>', unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="progress-box">
                    <div class="progress-fill" style="width:{real_prob:.2f}%"></div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(f'<div class="conf-text">The news is {real_prob:.2f}% real</div>', unsafe_allow_html=True)

    # ✅ Expand/Collapse sections
    with st.expander("📘 How this system works", expanded=False):
        st.write("• ML model trained on real & fake news datasets")
        st.write("• Converts news text using TF-IDF vectorizer")
        st.write("• Uses ML classifier to predict Fake/Real")
        st.write("• Educational tool (not final fact declaration)")

    with st.expander("⚙️ Technical Highlights", expanded=False):
        st.write("• Kaggle + LIAR style datasets")
        st.write("• TF-IDF vectorizer + trained ML model")
        st.write("• Probability confidence scoring")
        st.write("• Ethical AI with uncertainty awareness")

    st.markdown("---")
    st.markdown(
        f"Created by **{CREATOR_NAME}** • {CREATOR_ROLE} • "
        f"[LinkedIn]({MY_LINKEDIN}) • [GitHub]({MY_GITHUB})"
    )

elif page == "📊 Dashboard":
    show_default_page("📊 Dashboard", [
        "Overview cards: Total Predictions, Fake %, Real %",
        "Recent prediction history table (last 10)",
        "User profile section + Logout",
        "Quick analytics (charts) for credibility score",
    ])

elif page == "🔗 Integrations":
    show_default_page("🔗 Integrations", [
        "Connect NewsAPI to fetch latest articles",
        "Enable URL credibility checker (future)",
        "Connect Fact-check APIs (PolitiFact/Snopes style)",
        "Browser extension integration (future)",
    ])

elif page == "📌 Events":
    show_default_page("📌 Events", [
        "Misinformation trending alerts",
        "Upcoming awareness events",
        "Notification feed for fake news spikes",
        "Auto suggestion: what to verify today",
    ])

elif page == "🧾 Reporting":
    show_default_page("🧾 Reporting", [
        "Download report as PDF",
        "Export results to CSV",
        "Share analysis report link",
        "Admin report: overall prediction summary",
    ])

elif page == "👥 Users":
    show_default_page("👥 Users", [
        "Login / Signup module (future)",
        "User history: saved predictions",
        "User feedback collection (Real/Fake confirmation)",
        "Account settings & reset password",
    ])

elif page == "🔑 API Keys":
    show_default_page("🔑 API Keys", [
        "Store News API keys securely",
        "Enable OpenAI summarizer key (optional)",
        "Enable external fact-check API keys",
        "Toggle integrations ON/OFF",
    ])

elif page == "⚙️ Settings":
    show_default_page("⚙️ Settings", [
        "Theme: Dark/Light mode",
        "Language settings (English/Tamil - future)",
        "Privacy & data retention settings",
        "Notification preferences",
    ])
