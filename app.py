import joblib
import re
import time
import nltk
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
import streamlit.components.v1 as components

# =========================================================
# NLTK setup
# =========================================================
nltk.download("stopwords", quiet=True)

# =========================================================
# APP CONFIG
# =========================================================
APP_NAME = "DetectiveNews"

CREATOR_NAME = "SINDHU MUDALIYAR D"
CREATOR_EMAIL = "sindhudeva489@gmail.com"
CREATOR_ROLE = "AI/ML Student • Fake News Detection Project"
MY_LINKEDIN = "https://www.linkedin.com/in/sindhu-mudaliyar-d-a05450290"
MY_GITHUB = "https://github.com/SindhuBharathi21"

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title=APP_NAME,
    page_icon="logo.png",
    layout="wide",
)

# =========================================================
# LOAD MODEL + VECTORIZER
# =========================================================
# IMPORTANT: your files must exist with same names
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# =========================================================
# SESSION STATE DEFAULTS
# =========================================================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "history" not in st.session_state:
    # store predictions history
    st.session_state.history = []  # list of dicts

if "show_creator" not in st.session_state:
    st.session_state.show_creator = False

# =========================================================
# THEME TOGGLE (Top right)
# =========================================================
c1, c2 = st.columns([8, 2])
with c2:
    is_dark = st.toggle("Dark Mode", value=(st.session_state.theme == "dark"))
st.session_state.theme = "dark" if is_dark else "light"

# =========================================================
# COLORS
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
# GLOBAL CSS (clean & production)
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
    padding-top: 18px;
}}

.big-title {{
    text-align:center;
    font-size:48px;
    font-weight:900;
    margin-top:40px;
    margin-bottom:10px;
    color:{TITLE};
}}
.sub-title {{
    text-align:center;
    font-size:18px;
    color:{SUBTXT};
    margin-bottom:25px;
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
    margin-top:20px;
}}

.progress-box {{
    width:420px;
    height:20px;
    background:{BAR_BG};
    border-radius:16px;
    margin:14px auto 10px auto;
    overflow:hidden;
}}
.progress-fill {{
    height:100%;
    border-radius:16px;
    background:{BTN_BG};
}}

.conf-text {{
    text-align:center;
    font-size:20px;
    margin-top:6px;
    opacity:0.95;
}}

.info-card {{
    background:{CARD_BG};
    border:1px solid {CARD_BORDER};
    padding:18px 20px;
    border-radius:18px;
    margin-top:16px;
    box-shadow:0px 10px 22px rgba(0,0,0,0.25);
}}

.creator-sidebar-card {{
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
    z-index: 999;
}}

.creator-fab-container {{
    position: fixed;
    bottom: 26px;
    right: 26px;
    z-index: 9999;
}}

.creator-fab {{
    width: 58px;
    height: 58px;
    border-radius: 50%;
    background: linear-gradient(135deg,#7c3aed,#a855f7,#c084fc);
    display:flex;
    align-items:center;
    justify-content:center;
    box-shadow:0px 10px 25px rgba(168,85,247,0.45);
    border: none;
    cursor: pointer;
    color: white;
    font-size: 26px;
    font-weight: 900;
}}

.creator-popup {{
    position: fixed;
    bottom: 95px;
    right: 26px;
    width: 340px;
    padding: 18px;
    border-radius: 18px;
    background: rgba(10,8,20,0.92);
    border: 1px solid rgba(255,255,255,0.18);
    backdrop-filter: blur(14px);
    z-index: 9999;
    color: #fff;
    box-shadow:0px 15px 35px rgba(0,0,0,0.40);
}}

.creator-popup a {{
    color: #c4b5fd;
    font-weight: 800;
    text-decoration: none;
}}
.creator-popup a:hover {{
    text-decoration: underline;
}}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# HELPERS
# =========================================================
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower().split()
    sw = set(stopwords.words("english"))
    text = [w for w in text if w not in sw]
    return " ".join(text)

def show_default_page(title, points):
    st.markdown(f"## {title}")
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.write("✅ Default features included in this section:")
    for p in points:
        st.write("•", p)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.image("logo.png", width=72)
st.sidebar.markdown(
    f"<div style='font-size:24px;font-weight:900;margin-bottom:6px;'>{APP_NAME}</div>",
    unsafe_allow_html=True
)

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

# Sidebar creator card (bottom)
st.sidebar.markdown(
    f"""
    <div class="creator-sidebar-card">
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
# FLOATING CREATOR BUTTON + POPUP (NO HTML CODE ISSUE)
# =========================================================
# Use Streamlit button but placed via CSS container
st.markdown('<div class="creator-fab-container">', unsafe_allow_html=True)
fab_clicked = st.button("👤", key="creator_fab")
st.markdown('</div>', unsafe_allow_html=True)

if fab_clicked:
    st.session_state.show_creator = not st.session_state.show_creator

if st.session_state.show_creator:
    components.html(
        f"""
        <div class="creator-popup">
            <div style="display:flex;gap:12px;align-items:center;">
                <div style="
                    width:54px;height:54px;border-radius:50%;
                    background:linear-gradient(135deg,#7c3aed,#a855f7,#c084fc);
                    display:flex;align-items:center;justify-content:center;
                    font-weight:900;font-size:22px;
                ">S</div>
                <div>
                    <div style="font-size:18px;font-weight:900;">{CREATOR_NAME}</div>
                    <div style="font-size:13px;color:#ddd;">{CREATOR_ROLE}</div>
                </div>
            </div>

            <hr style="border:0;border-top:1px solid rgba(255,255,255,0.15);margin:12px 0;">

            <div style="font-size:14px;line-height:1.7;">
                📩 <b>Email:</b> {CREATOR_EMAIL}<br/>
                🔗 <a href="{MY_LINKEDIN}" target="_blank">LinkedIn Profile</a><br/>
                💻 <a href="{MY_GITHUB}" target="_blank">GitHub Profile</a>
            </div>

            <div style="margin-top:12px;font-size:12px;color:#c4b5fd;">
                Tip: Use sidebar menu to explore Dashboard, Reporting & Settings.
            </div>
        </div>
        """,
        height=10,
    )

# =========================================================
# PAGES
# =========================================================
if page == "🏠 Home":
    st.markdown('<div class="big-title">Check if news is real or fake!</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter an article below, then click Predict.</div>', unsafe_allow_html=True)

    news_text = st.text_area("", placeholder="Enter your article here...", height=160)

    left, mid, right = st.columns([2, 1, 2])
    with mid:
        predict = st.button("🔍 Predict", disabled=(news_text.strip() == ""))

    if predict:
        with st.spinner("Analyzing..."):
            time.sleep(0.7)

        cleaned = clean_text(news_text)
        vect = vectorizer.transform([cleaned])

        pred = int(model.predict(vect)[0])
        proba = model.predict_proba(vect)[0]
        real_prob = float(proba[1] * 100)

        label = "REAL ✅" if pred == 1 else "FAKE ❌"

        # Store history
        st.session_state.history.append(
            {
                "text": news_text[:120] + ("..." if len(news_text) > 120 else ""),
                "prediction": "REAL" if pred == 1 else "FAKE",
                "confidence_real_%": round(real_prob, 2),
            }
        )

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

        st.markdown(
            """
            <div class="info-card" style="margin-top:18px;">
            ⚠️ <b>Disclaimer:</b> This is an educational AI tool. Always verify with trusted sources (official news, govt portals).
            </div>
            """,
            unsafe_allow_html=True
        )

    # Expand/Collapse sections
    with st.expander("📘 How this system works", expanded=False):
        st.write("• ML model trained on real & fake news datasets")
        st.write("• Text preprocessing (cleaning + stopwords)")
        st.write("• TF-IDF vectorization converts text to features")
        st.write("• Classifier predicts Fake/Real with confidence")

    with st.expander("⚙️ Technical Highlights", expanded=False):
        st.write("• Kaggle + LIAR style datasets")
        st.write("• TF-IDF Vectorizer + Scikit-learn classifier")
        st.write("• Probability confidence scoring")
        st.write("• Ethical AI with uncertainty awareness")

    st.markdown("---")
    st.markdown(
        f"Created by **{CREATOR_NAME}** • {CREATOR_ROLE} • "
        f"[LinkedIn]({MY_LINKEDIN}) • [GitHub]({MY_GITHUB})"
    )

elif page == "📊 Dashboard":
    st.markdown("## 📊 Dashboard Overview")
    st.markdown(
        "<div class='info-card'>This dashboard summarizes your recent predictions.</div>",
        unsafe_allow_html=True
    )

    history = st.session_state.history
    total = len(history)
    fake_count = sum(1 for h in history if h["prediction"] == "FAKE")
    real_count = sum(1 for h in history if h["prediction"] == "REAL")

    colA, colB, colC = st.columns(3)
    colA.metric("Total Predictions", total)
    colB.metric("FAKE Count", fake_count)
    colC.metric("REAL Count", real_count)

    # Pie chart
    st.markdown("### Prediction Distribution")
    fig, ax = plt.subplots()
    ax.pie(
        [real_count, fake_count],
        labels=["REAL", "FAKE"],
        autopct="%1.1f%%",
        startangle=90
    )
    ax.axis("equal")
    st.pyplot(fig)

    # Recent history table
    st.markdown("### Recent Prediction History")
    if total == 0:
        st.info("No predictions yet. Go to Home page and test an article.")
    else:
        df = pd.DataFrame(history[::-1][:10])
        st.dataframe(df, use_container_width=True)

elif page == "🔗 Integrations":
    show_default_page("🔗 Integrations", [
        "Connect NewsAPI to fetch latest articles",
        "Enable URL credibility checker (future)",
        "Connect fact-check APIs (future)",
        "Browser extension integration (future)",
    ])

elif page == "📌 Events":
    show_default_page("📌 Events", [
        "Fake news trending alerts (future)",
        "Upcoming awareness events",
        "Notification feed for misinformation spikes",
        "Suggestions: topics to verify today",
    ])

elif page == "🧾 Reporting":
    show_default_page("🧾 Reporting", [
        "Download credibility reports as PDF (future)",
        "Export results to CSV",
        "Share analysis report link",
        "Admin summary analytics (future)",
    ])

elif page == "👥 Users":
    show_default_page("👥 Users", [
        "Login/Signup module (future)",
        "User history: saved predictions",
        "Feedback (Real/Fake confirmation)",
        "Account settings & reset password",
    ])

elif page == "🔑 API Keys":
    show_default_page("🔑 API Keys", [
        "Store News API keys securely",
        "Enable external fact-check APIs (future)",
        "Toggle integrations ON/OFF",
        "Secrets management via Streamlit Cloud",
    ])

elif page == "⚙️ Settings":
    show_default_page("⚙️ Settings", [
        "Theme control: Dark/Light mode",
        "Language settings (future)",
        "Privacy & data retention settings",
        "Notification preferences",
    ])
