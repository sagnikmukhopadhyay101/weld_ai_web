import streamlit as st
import cv2
import numpy as np
import os
import uuid
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from streamlit_drawable_canvas import st_canvas

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="WeldAI - Intelligent Weld Inspector",
    page_icon="🔥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ======================================================
# THEME TOGGLE
# ======================================================
mode = st.toggle("🌙 Dark mode", value=True)

if mode:
    bg = "#0a0e27"
    card = "#141b2d"
    card_hover = "#1a2332"
    text = "#e8eaed"
    text_dim = "#9ca3af"
    accent = "#00d4ff"
    accent_secondary = "#7c3aed"
    success = "#10b981"
    warning = "#f59e0b"
    error = "#ef4444"
else:
    bg = "#f8f9fa"
    card = "#ffffff"
    card_hover = "#f1f3f5"
    text = "#1f2937"
    text_dim = "#6b7280"
    accent = "#0ea5e9"
    accent_secondary = "#8b5cf6"
    success = "#059669"
    warning = "#d97706"
    error = "#dc2626"

# ======================================================
# ENHANCED GLOBAL STYLES
# ======================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

body {{
    background: {bg};
    color: {text};
}}

/* Hero Header */
.hero {{
    background: linear-gradient(135deg, {accent} 0%, {accent_secondary} 100%);
    padding: 2.5rem 2rem;
    border-radius: 24px;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    position: relative;
    overflow: hidden;
}}

.hero::before {{
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    animation: pulse 8s ease-in-out infinite;
}}

@keyframes pulse {{
    0%, 100% {{ transform: scale(1); opacity: 0.5; }}
    50% {{ transform: scale(1.1); opacity: 0.8; }}
}}

.hero h1 {{
    color: white;
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    position: relative;
    z-index: 1;
}}

.hero p {{
    color: rgba(255,255,255,0.9);
    font-size: 1.1rem;
    margin: 0;
    position: relative;
    z-index: 1;
}}

/* Chat Messages */
.chat-ai {{
    background: {card};
    padding: 1.5rem;
    border-radius: 20px;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    border: 1px solid {'rgba(255,255,255,0.05)' if mode else 'rgba(0,0,0,0.05)'};
    transition: all 0.3s ease;
    position: relative;
}}

.chat-ai::before {{
    content: '';
    position: absolute;
    left: -4px;
    top: 50%;
    transform: translateY(-50%);
    width: 4px;
    height: 60%;
    background: linear-gradient(180deg, {accent}, {accent_secondary});
    border-radius: 2px;
}}

.chat-ai:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
}}

.chat-human {{
    background: linear-gradient(135deg, {accent} 0%, {accent_secondary} 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 20px;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
}}

.chat-human:hover {{
    transform: translateY(-2px);
    box-shadow: 0 12px 32px rgba(0,0,0,0.25);
}}

/* Status Badges */
.status-badge {{
    display: inline-block;
    padding: 0.5rem 1.2rem;
    border-radius: 100px;
    font-weight: 600;
    font-size: 0.9rem;
    margin: 0.5rem 0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

.status-good {{
    background: linear-gradient(135deg, {success}, #34d399);
    color: white;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
}}

.status-defective {{
    background: linear-gradient(135deg, {error}, #f87171);
    color: white;
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
}}

.status-uncertain {{
    background: linear-gradient(135deg, {warning}, #fbbf24);
    color: white;
    box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
}}

/* Card Containers */
.card {{
    background: {card};
    padding: 1.5rem;
    border-radius: 20px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    border: 1px solid {'rgba(255,255,255,0.05)' if mode else 'rgba(0,0,0,0.05)'};
    transition: all 0.3s ease;
}}

.card:hover {{
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
}}

.image-card {{
    background: {card};
    padding: 1rem;
    border-radius: 20px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    border: 1px solid {'rgba(255,255,255,0.08)' if mode else 'rgba(0,0,0,0.08)'};
    overflow: hidden;
}}

.image-card img {{
    border-radius: 12px;
}}

/* Defect List */
.defect-item {{
    background: {'rgba(255,255,255,0.03)' if mode else 'rgba(0,0,0,0.02)'};
    padding: 0.8rem 1rem;
    border-radius: 12px;
    margin: 0.5rem 0;
    border-left: 3px solid {accent};
    transition: all 0.2s ease;
}}

.defect-item:hover {{
    background: {'rgba(255,255,255,0.05)' if mode else 'rgba(0,0,0,0.04)'};
    transform: translateX(4px);
}}

/* Buttons */
.stButton>button {{
    border-radius: 12px;
    padding: 0.75rem 1.8rem;
    font-weight: 600;
    border: none;
    background: linear-gradient(135deg, {accent}, {accent_secondary});
    color: white;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}}

.stButton>button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.25);
}}

/* File Uploader */
.stFileUploader {{
    background: {card};
    border-radius: 16px;
    padding: 2rem;
    border: 2px dashed {'rgba(255,255,255,0.1)' if mode else 'rgba(0,0,0,0.1)'};
    transition: all 0.3s ease;
}}

.stFileUploader:hover {{
    border-color: {accent};
    background: {card_hover};
}}

/* Radio Buttons */
.stRadio > div {{
    background: {card};
    padding: 1rem;
    border-radius: 12px;
}}

/* Text Input */
.stTextInput>div>div>input {{
    border-radius: 12px;
    border: 2px solid {'rgba(255,255,255,0.1)' if mode else 'rgba(0,0,0,0.1)'};
    padding: 0.75rem 1rem;
    background: {card};
    color: {text};
    transition: all 0.3s ease;
}}

.stTextInput>div>div>input:focus {{
    border-color: {accent};
    box-shadow: 0 0 0 3px {accent}33;
}}

/* Expander */
.streamlit-expanderHeader {{
    background: {card};
    border-radius: 12px;
    font-weight: 600;
    transition: all 0.3s ease;
}}

.streamlit-expanderHeader:hover {{
    background: {card_hover};
}}

/* Toggle */
.stCheckbox {{
    background: {card};
    padding: 0.5rem 1rem;
    border-radius: 12px;
}}

/* Success/Error Messages */
.stSuccess, .stError, .stWarning, .stInfo {{
    border-radius: 12px;
    padding: 1rem 1.5rem;
    border: none;
}}

/* Scrollbar */
::-webkit-scrollbar {{
    width: 10px;
}}

::-webkit-scrollbar-track {{
    background: {bg};
}}

::-webkit-scrollbar-thumb {{
    background: {accent};
    border-radius: 5px;
}}

::-webkit-scrollbar-thumb:hover {{
    background: {accent_secondary};
}}

/* Icon Styling */
.icon {{
    display: inline-block;
    margin-right: 0.5rem;
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));
}}

/* Fade-in Animation */
@keyframes fadeIn {{
    from {{
        opacity: 0;
        transform: translateY(20px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

.fade-in {{
    animation: fadeIn 0.5s ease-out;
}}

/* Stats Grid */
.stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}}

.stat-card {{
    background: {card};
    padding: 1.2rem;
    border-radius: 16px;
    text-align: center;
    border: 1px solid {'rgba(255,255,255,0.05)' if mode else 'rgba(0,0,0,0.05)'};
}}

.stat-value {{
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, {accent}, {accent_secondary});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

.stat-label {{
    color: {text_dim};
    font-size: 0.85rem;
    margin-top: 0.3rem;
}}

</style>
""", unsafe_allow_html=True)

# ======================================================
# HERO TITLE
# ======================================================
st.markdown(f"""
<div class="hero fade-in">
    <h1><span class="icon">🔥</span>WeldAI</h1>
    <p>Intelligent AI-powered weld inspection assistant that detects defects, explains decisions, and learns from your expertise.</p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# SESSION STATE
# ======================================================
for key in ["image", "image_np", "image_path", "status", "defects", "yolo_plot", "edges"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ======================================================
# FILE SYSTEM
# ======================================================
os.makedirs("temp", exist_ok=True)
os.makedirs("feedback_data/images", exist_ok=True)

label_file = "feedback_data/labels.csv"
if not os.path.exists(label_file):
    pd.DataFrame(columns=[
        "image_name", "x", "y", "width", "height", "defect_type"
    ]).to_csv(label_file, index=False)

# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ======================================================
# CRACK DETECTION
# ======================================================
def detect_crack(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5),0), 80, 180)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for c in contours if cv2.arcLength(c, False) > 120) >= 2, edges

# ======================================================
# DECISION LOGIC
# ======================================================
def decide(yolo_boxes, crack):
    defects = []
    strong = False

    if yolo_boxes:
        for b in yolo_boxes:
            conf = float(b.conf[0])
            cls = int(b.cls[0])
            if conf >= 0.6:
                strong = True
                defects.append((cls, conf))
            elif conf >= 0.3:
                defects.append((cls, conf))

    if crack:
        defects.append(("crack", 0.4))

    if strong:
        return "DEFECTIVE", defects
    elif defects:
        return "UNCERTAIN", defects
    return "GOOD", defects

# ======================================================
# IMAGE UPLOAD
# ======================================================
st.markdown('<div class="fade-in">', unsafe_allow_html=True)
uploaded = st.file_uploader("📤 Upload a weld image for inspection", ["jpg","png","jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded and st.session_state.image is None:
    st.session_state.image = Image.open(uploaded).convert("RGB")
    st.session_state.image_np = np.array(st.session_state.image)
    st.session_state.image_path = f"temp/{uuid.uuid4().hex}.jpg"
    cv2.imwrite(
        st.session_state.image_path,
        cv2.cvtColor(st.session_state.image_np, cv2.COLOR_RGB2BGR)
    )

# ======================================================
# SHOW IMAGE
# ======================================================
if st.session_state.image is not None:
    st.markdown('<div class="image-card fade-in">', unsafe_allow_html=True)
    st.image(st.session_state.image, use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🧠 Analyze Weld", use_container_width=True):
            with st.spinner("🔍 Analyzing weld integrity..."):
                yolo = model(st.session_state.image_path, conf=0.2, imgsz=1280)
                crack, edges = detect_crack(st.session_state.image_np)
                st.session_state.status, st.session_state.defects = decide(yolo[0].boxes, crack)
                st.session_state.yolo_plot = cv2.cvtColor(yolo[0].plot(), cv2.COLOR_BGR2RGB)
                st.session_state.edges = edges

# ======================================================
# AI CHAT RESPONSE
# ======================================================
if st.session_state.status:
    msg = ""
    badge_class = ""
    icon = ""
    
    if st.session_state.status == "GOOD":
        msg = "This weld looks **excellent**! I did not detect any critical defects that would compromise structural integrity."
        badge_class = "status-good"
        icon = "✅"
    elif st.session_state.status == "DEFECTIVE":
        msg = "I detected **significant defects** that could affect weld integrity and require immediate attention."
        badge_class = "status-defective"
        icon = "⚠️"
    else:
        msg = "I'm **not fully confident** in this assessment. Human expert review is recommended for final verification."
        badge_class = "status-uncertain"
        icon = "⚡"

    st.markdown(f"""
    <div class="chat-ai fade-in">
        <strong style="font-size: 1.1rem;">🤖 WeldAI Analysis</strong>
        <div class="status-badge {badge_class}">{icon} {st.session_state.status}</div>
        <p style="margin-top: 1rem; font-size: 1rem; line-height: 1.6;">{msg}</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.defects:
        defect_html = '<div class="chat-ai fade-in"><strong style="font-size: 1.05rem;">🔍 Detected Issues:</strong><br><br>'
        for i, d in enumerate(st.session_state.defects):
            if d[0] == "crack":
                defect_html += f'<div class="defect-item">🔴 Possible surface crack detected</div>'
            else:
                defect_html += f'<div class="defect-item">🔸 {model.names[d[0]].capitalize()} (Confidence: {d[1]*100:.0f}%)</div>'
        defect_html += '</div>'
        st.markdown(defect_html, unsafe_allow_html=True)

    with st.expander("🔬 View Detailed Analysis Visuals"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**YOLO Detection Results**")
            st.image(st.session_state.yolo_plot)
        with col2:
            st.markdown("**Edge Detection Analysis**")
            st.image(st.session_state.edges, clamp=True)

# ======================================================
# HUMAN FEEDBACK (CHAT STYLE)
# ======================================================
if st.session_state.status:
    st.markdown("""
    <div class="chat-human fade-in">
        <strong style="font-size: 1.1rem;">👤 Your Expert Feedback</strong><br>
        <p style="margin-top: 0.5rem; opacity: 0.95;">Help improve WeldAI by confirming or correcting the analysis</p>
    </div>
    """, unsafe_allow_html=True)

    feedback = st.radio(
        "Select your assessment:",
        ["✅ AI is correct", "❌ AI is wrong (no defect found)", "➕ AI missed a defect"],
        label_visibility="collapsed"
    )

    if feedback == "❌ AI is wrong (no defect found)":
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("✅ Confirm: This weld is GOOD", use_container_width=True):
                df = pd.read_csv(label_file)
                df.loc[len(df)] = [
                    os.path.basename(st.session_state.image_path),
                    None, None, None, None, "no_defect"
                ]
                df.to_csv(label_file, index=False)
                st.success("✅ Feedback saved! WeldAI has been corrected and will learn from this.")

    if feedback == "➕ AI missed a defect":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        name = st.text_input("🏷️ Defect type name:", placeholder="e.g., porosity, undercut, slag inclusion")
        st.markdown("**Draw a box around the missed defect:**")
        canvas = st_canvas(
            background_image=st.session_state.image,
            drawing_mode="rect",
            stroke_color="#ef4444",
            stroke_width=3,
            fill_color="rgba(239, 68, 68, 0.2)",
            height=st.session_state.image.height,
            width=st.session_state.image.width,
            key="fb"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("💾 Save Correction", use_container_width=True):
                if canvas.json_data and name.strip():
                    df = pd.read_csv(label_file)
                    for obj in canvas.json_data["objects"]:
                        df.loc[len(df)] = [
                            os.path.basename(st.session_state.image_path),
                            obj["left"], obj["top"],
                            obj["width"], obj["height"],
                            name.lower()
                        ]
                    df.to_csv(label_file, index=False)
                    st.success("💾 Correction saved! Thank you for improving WeldAI's accuracy.")
                else:
                    st.warning("⚠️ Please provide both a defect name and draw a bounding box.")

# ======================================================
# RESET
# ======================================================
st.markdown("---")
col1, col2, col3 = st.columns([1,2,1])
with col2:
    if st.button("🔄 Start New Inspection", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ======================================================
# FOOTER
# ======================================================
st.markdown(f"""
<div style="text-align: center; margin-top: 3rem; padding: 1.5rem; color: {text_dim}; font-size: 0.85rem;">
    <p>Powered by YOLO & OpenCV | Built with ❤️ for weld quality assurance</p>
</div>
""", unsafe_allow_html=True)