import streamlit as st
from PIL import Image

# Page settings
st.set_page_config(
    page_title="Spam Email Detection",
    page_icon="📧",
    layout="wide"
)

# Load Images
banner = Image.open("images/spam.gif")
email_icon = Image.open("images/email_icon.webp")
detect_icon = Image.open("images/detect_icon.png")
ml_icon = Image.open("images/ml_icon.png")

# Custom CSS
st.markdown("""
    <style>
        .main-container {
            padding: 2rem 1rem;
        }
        .hero-text {
            font-size: 2.8em;
            font-weight: 800;
            color: #333;
            text-align: center;
        }
        .sub-text {
            font-size: 1.2em;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .feature-box {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            text-align: center;
        }
        .feature-title {
            font-size: 1.2em;
            font-weight: 600;
            margin-top: 1rem;
            color: #222;
        }
        .feature-desc {
            color: #555;
            font-size: 0.95em;
        }
        .footer {
            text-align: center;
            color: #aaa;
            font-size: 0.9em;
            margin-top: 3rem;
        }
    </style>
""", unsafe_allow_html=True)

# Main Container
# st.markdown('<div class="main-container">', unsafe_allow_html=True)

# --- Hero Section ---
st.markdown('<div class="hero-text">📧 Spam Email Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Detect spam emails in real-time using intelligent machine learning algorithms.</div>', unsafe_allow_html=True)

# Center and resize the banner image using columns
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image('images/spam.gif')

# --- Features Section ---
st.markdown("### 🔍 Key Features")
col1, col2, col3 = st.columns(3)

with col1:
    st.image(email_icon, width=70)
    st.markdown('<div class="feature-title">Easy Input</div>', unsafe_allow_html=True)
    st.markdown('<div class="feature-desc">Paste email text or upload a file for detection.</div>', unsafe_allow_html=True)

with col2:
    st.image(detect_icon, width=70)
    st.markdown('<div class="feature-title">Fast Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="feature-desc">Get instant results with clear spam/not-spam output.</div>', unsafe_allow_html=True)

with col3:
    st.image(ml_icon, width=70)
    st.markdown('<div class="feature-title">ML Powered</div>', unsafe_allow_html=True)
    st.markdown('<div class="feature-desc">Backed by natural language processing and classification algorithms.</div>', unsafe_allow_html=True)

# --- How to Use Section ---
st.markdown("### 🚀 How to Use")
st.markdown("""
1. Navigate to the **'Detection'** page from the sidebar.  
2. Paste the email or upload a `.txt` file.  
3. Click on **Detect Spam**.  
4. View the result with confidence score.
""")

# --- About ---
with st.expander("ℹ️ About This Project"):
    st.markdown("""
    This project is developed as a demonstration of using machine learning for real-world NLP applications.
    
    **Technologies used**:
    - Streamlit
    - Scikit-learn
    - Natural Language Processing (NLP)
    - TF-IDF / Count Vectorizer
    """)

# Footer
st.markdown('<div class="footer">© 2025 Spam Detection App | Created by Your Name</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
