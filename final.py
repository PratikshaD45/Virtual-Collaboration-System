# import re
# import streamlit as st
# from typing import Tuple, Any
# import pandas as pd
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# import google.generativeai as genai
# from pathlib import Path
# from fpdf import FPDF
# import io
# import time

# # Set page configuration with custom theme
# st.set_page_config(
#     page_title="Innovation Hub - Collaborate, Evaluate, Impact",
#     page_icon="🚀",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for professional UI
# st.markdown("""
# <style>
#     /* Main color palette */
#     :root {
#         --primary: #2563EB;
#         --primary-light: #DBEAFE;
#         --primary-dark: #1E40AF;
#         --secondary: #10B981;
#         --secondary-light: #D1FAE5;
#         --secondary-dark: #047857;
#         --accent: #8B5CF6;
#         --accent-light: #EDE9FE;
#         --accent-dark: #6D28D9;
#         --neutral-50: #F9FAFB;
#         --neutral-100: #F3F4F6;
#         --neutral-200: #E5E7EB;
#         --neutral-300: #D1D5DB;
#         --neutral-600: #4B5563;
#         --neutral-700: #374151;
#         --neutral-800: #1F2937;
#         --neutral-900: #111827;
#         --danger: #EF4444;
#         --warning: #F59E0B;
#     }

#     /* Typography */
#     h1, h2, h3, h4, h5, h6 {
#         font-family: 'Inter', sans-serif !important;
#     }
    
#     .main-header {
#         font-size: 2.5rem !important;
#         font-weight: 800 !important;
#         color: var(--neutral-800) !important;
#         margin-bottom: 1.5rem !important;
#         letter-spacing: -0.025em !important;
#     }
    
#     .sub-header {
#         font-size: 1.8rem !important;
#         font-weight: 700 !important;
#         color: var(--neutral-800) !important;
#         margin-top: 2rem !important;
#         margin-bottom: 1rem !important;
#         letter-spacing: -0.025em !important;
#     }
    
#     /* Feature Cards */
#     .feature-card {
#         background-color: white;
#         border-radius: 12px;
#         padding: 1.5rem;
#         margin-bottom: 1.5rem;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 10px 15px rgba(0, 0, 0, 0.03);
#         transition: transform 0.2s, box-shadow 0.2s;
#         cursor: pointer;
#         border: 1px solid var(--neutral-200);
#         height: 100%;
#     }
    
#     .feature-card:hover {
#         transform: translateY(-5px);
#         box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
#     }
    
#     .feature-card.collaborators {
#         border-top: 5px solid var(--primary);
#     }
    
#     .feature-card.evaluate {
#         border-top: 5px solid var(--secondary);
#     }
    
#     .feature-card.impact {
#         border-top: 5px solid var(--accent);
#     }
    
#     .feature-icon {
#         font-size: 2.5rem;
#         margin-bottom: 1rem;
#     }
    
#     .feature-title {
#         font-size: 1.5rem !important;
#         font-weight: 700 !important;
#         margin-bottom: 0.75rem !important;
#         color: var(--neutral-800) !important;
#     }
    
#     .feature-description {
#         color: var(--neutral-600) !important;
#         font-size: 1rem !important;
#         line-height: 1.5 !important;
#     }
    
#     /* Profile Cards */
#     .card {
#         background-color: white;
#         border-radius: 12px;
#         padding: 1.5rem;
#         margin-bottom: 1.5rem;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
#         border-left: 5px solid var(--primary);
#     }
    
#     .profile-name {
#         font-size: 1.3rem !important;
#         font-weight: 700 !important;
#         color: var(--primary-dark) !important;
#         margin-bottom: 0.75rem !important;
#     }
    
#     .profile-field-label {
#         font-weight: 600 !important;
#         color: var(--neutral-700) !important;
#     }
    
#     /* Buttons */
#     .stButton>button {
#         background-color: var(--primary);
#         color: white;
#         font-weight: 500;
#         border-radius: 8px;
#         padding: 0.5rem 1rem;
#         border: none;
#         transition: background-color 0.2s;
#     }
    
#     .stButton>button:hover {
#         background-color: var(--primary-dark);
#     }
    
#     .download-btn {
#         background-color: var(--secondary) !important;
#     }
    
#     .download-btn:hover {
#         background-color: var(--secondary-dark) !important;
#     }
    
#     /* Chat UI */
#     .chat-message {
#         padding: 1rem;
#         border-radius: 10px;
#         margin-bottom: 1rem;
#         display: flex;
#         flex-direction: column;
#     }
    
#     .user-message {
#         background-color: var(--primary-light);
#         border-left: 5px solid var(--primary);
#     }
    
#     .assistant-message {
#         background-color: var(--secondary-light);
#         border-left: 5px solid var(--secondary);
#     }
    
#     /* Form elements */
#     .stTextInput>div>div>input {
#         border-radius: 8px;
#         border: 1px solid var(--neutral-300);
#         padding: 0.75rem 1rem;
#         font-size: 1rem;
#     }
    
#     .stTextInput>div>div>input:focus {
#         border-color: var(--primary);
#         box-shadow: 0 0 0 2px var(--primary-light);
#     }
    
#     /* Containers */
#     .upload-section {
#         padding: 1.5rem;
#         background-color: white;
#         border-radius: 12px;
#         margin-bottom: 1.5rem;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
#     }
    
#     .info-box {
#         padding: 1rem;
#         background-color: var(--secondary-light);
#         border-radius: 8px;
#         margin-bottom: 1rem;
#         border-left: 4px solid var(--secondary);
#     }
    
#     .error-box {
#         padding: 1rem;
#         background-color: #FFEBEE;
#         border-radius: 8px;
#         margin-bottom: 1rem;
#         border-left: 4px solid var(--danger);
#     }
    
#     /* Progress bar */
#     .stProgress > div > div > div > div {
#         background-color: var(--primary);
#     }
    
#     /* Tabs */
#     .stTabs [data-baseweb="tab-list"] {
#         gap: 8px;
#     }
    
#     .stTabs [data-baseweb="tab"] {
#         height: 50px;
#         white-space: pre-wrap;
#         background-color: var(--neutral-100);
#         border-radius: 8px 8px 0 0;
#         gap: 1px;
#         padding: 10px 16px;
#         color: var(--neutral-700);
#     }
    
#     .stTabs [aria-selected="true"] {
#         background-color: white;
#         color: var(--primary);
#         border-top: 3px solid var(--primary);
#         font-weight: 600;
#     }
    
#     /* File uploader */
#     .stFileUploader > div > button {
#         background-color: var(--primary);
#     }
    
#     .stFileUploader > div > button:hover {
#         background-color: var(--primary-dark);
#     }
    
#     /* Welcome section */
#     .welcome-container {
#         text-align: center;
#         padding: 3rem 2rem;
#         background-color: white;
#         border-radius: 12px;
#         margin-top: 2rem;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
#     }
    
#     .welcome-title {
#         color: var(--primary);
#         font-size: 2rem;
#         font-weight: 700;
#         margin-bottom: 1.5rem;
#     }
    
#     .welcome-subtitle {
#         font-size: 1.2rem;
#         margin: 1rem 0;
#         color: var(--neutral-700);
#     }
    
#     .welcome-features {
#         text-align: left;
#         display: inline-block;
#         margin: 1.5rem 0;
#     }
    
#     .welcome-features li {
#         margin-bottom: 0.5rem;
#         color: var(--neutral-700);
#     }
    
#     /* Footer */
#     .footer {
#         text-align: center;
#         margin-top: 3rem;
#         padding-top: 1.5rem;
#         border-top: 1px solid var(--neutral-200);
#         color: var(--neutral-600);
#         font-size: 0.9rem;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Configuration
# MODEL_NAME = "all-MiniLM-L6-v2"
# GEMINI_MODEL = "gemini-1.5-flash"
# API_KEY = "AIzaSyBYfzDovXQK6E8jZsrOBieoSY_X6jCUktU"  # Store this securely

# # Initialize Google API
# @st.cache_resource
# def init_genai() -> Any:
#     genai.configure(api_key=API_KEY)
#     return genai.GenerativeModel(GEMINI_MODEL)

# # Load and cache the embedding model
# @st.cache_resource
# def load_encoder() -> SentenceTransformer:
#     return SentenceTransformer(MODEL_NAME)

# @st.cache_data
# def load_data(uploaded_file: Path) -> Tuple[pd.DataFrame, Any, SentenceTransformer, np.ndarray]:
#     try:
#         with st.spinner("Processing data and creating embeddings..."):
#             progress_bar = st.progress(0)
#             progress_bar.progress(10)
#             time.sleep(0.5)
            
#             df = pd.read_csv(uploaded_file, dtype_backend="pyarrow", encoding="ISO-8859-1")
#             # Normalize column names to lowercase and remove BOM
#             df.columns = df.columns.str.replace(r"ï»¿", "", regex=True).str.strip().str.lower()
            
#             # Update progress
#             progress_bar.progress(30)
#             time.sleep(0.5)
            
#             # Combine text fields for embeddings
#             df["combined_text"] = df.astype(str).agg(" ".join, axis=1)
            
#             # Update progress
#             progress_bar.progress(40)
#             time.sleep(0.5)
            
#             model = load_encoder()
            
#             # Update progress
#             progress_bar.progress(50)
#             time.sleep(0.5)
            
#             # Show a message during embedding
#             st.info("Generating embeddings for semantic search. This may take a moment...")
            
#             embeddings = model.encode(
#                 df["combined_text"].tolist(),
#                 batch_size=32,
#                 show_progress_bar=False,
#                 convert_to_numpy=True
#             )
            
#             # Update progress
#             progress_bar.progress(80)
#             time.sleep(0.5)
            
#             # Create FAISS index
#             dimension = embeddings.shape[1]
#             index = faiss.IndexFlatL2(dimension)
#             faiss.normalize_L2(embeddings)
#             index.add(embeddings)
            
#             # Update progress
#             progress_bar.progress(100)
#             time.sleep(0.5)
            
#             # Remove progress elements
#             progress_bar.empty()
            
#             st.success(f"✅ Successfully loaded {len(df)} profiles and created search index!")
            
#             return df, index, model, embeddings
    
#     except Exception as e:
#         st.error(f"Error loading data: {str(e)}")
#         raise

# # Function to clean text: remove emojis and non-ASCII characters
# def clean_text(text):
#     if pd.isna(text):
#         return "N/A"
#     # Remove all characters not in the printable ASCII range (32 to 126)
#     return re.sub(r"[^\x20-\x7E]", "", str(text))

# # Function to generate PDF
# def generate_pdf(results, query):
#     # In PDF generation section:
#     pdf.multi_cell(0, 8, f"{clean_text(profile.get('LinkedIn Name', 'N/A'))}")  # Updated key
#     pdf = FPDF()
#     pdf.set_auto_page_break(auto=True, margin=15)
#     pdf.add_page()
    
#     # Add header
#     pdf.set_font("Arial", "B", 16)
#     pdf.cell(200, 10, "Top 5 Matching Profiles", ln=True, align="C")
#     pdf.set_font("Arial", "I", 12)
#     pdf.cell(200, 10, f"Search Query: '{query}'", ln=True, align="C")
#     pdf.line(10, pdf.get_y(), 200, pdf.get_y())
#     pdf.ln(10)
    
#     for idx, profile in enumerate(results, start=1):
#         pdf.set_font("Arial", "B", 14)
#         pdf.set_fill_color(230, 230, 230)
#         pdf.cell(200, 10, f"Profile {idx}", ln=True, align="L", fill=True)
#         pdf.ln(5)
#         pdf.set_font("Arial", "B", 12)
        
#         # Name with larger font
#         pdf.cell(40, 8, "Name:", 0)
#         pdf.set_font("Arial", size=12)
#         pdf.multi_cell(0, 8, f"{clean_text(profile.get('linkedin name', 'N/A'))}")
        
#         # Other fields
#         pdf.set_font("Arial", "B", 11)
#         pdf.cell(40, 8, "Industry:", 0)
#         pdf.set_font("Arial", size=11)
#         pdf.multi_cell(0, 8, f"{clean_text(profile.get('industry', 'N/A'))}")
        
#         pdf.set_font("Arial", "B", 11)
#         pdf.cell(40, 8, "Current Role(s):", 0)
#         pdf.set_font("Arial", size=11)
#         pdf.multi_cell(0, 8, f"{clean_text(profile.get('current role(s)', 'N/A'))}")
        
#         pdf.set_font("Arial", "B", 11)
#         pdf.cell(40, 8, "Location:", 0)
#         pdf.set_font("Arial", size=11)
#         pdf.multi_cell(0, 8, f"{clean_text(profile.get('location', 'N/A'))}")
        
#         # About section with smaller font for longer text
#         pdf.set_font("Arial", "B", 11)
#         pdf.cell(40, 8, "About:", 0)
#         pdf.ln(8)
#         pdf.set_font("Arial", size=10)
#         about_text = clean_text(profile.get('about', 'N/A'))
#         pdf.multi_cell(0, 6, about_text)
        
#         # Profile link
#         pdf.ln(3)
#         pdf.set_font("Arial", "B", 11)
#         pdf.cell(40, 8, "Profile Link:", 0)
#         pdf.set_font("Arial", size=10)
#         pdf.set_text_color(0, 0, 255)  # Blue color for links
#         pdf.multi_cell(0, 8, f"{clean_text(profile.get('profile link', 'N/A'))}")
#         pdf.set_text_color(0, 0, 0)  # Reset to black
        
#         pdf.ln(5)
#         pdf.line(10, pdf.get_y(), 200, pdf.get_y())
#         pdf.ln(10)
    
#     # Add footer with date
#     pdf.set_y(-15)
#     pdf.set_font("Arial", "I", 8)
#     pdf.cell(0, 10, f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}", 0, 0, "C")
    
#     # Instead of outputting to a file, output to a string and then encode to bytes.
#     pdf_str = pdf.output(dest="S")
#     pdf_bytes = pdf_str.encode("latin1")
#     pdf_output = io.BytesIO(pdf_bytes)
#     return pdf_output

# # Function to display profile card
# def display_profile_card(profile, index):
#     # Update all profile.get() calls to use lowercase keys:
#     st.markdown(f"""
#     <div class="card">
#         <p class="profile-name">{index}. {clean_text(profile.get('linkedin name', 'N/A'))}</p>
#         <p><span class="profile-field-label">Industry:</span> {clean_text(profile.get('industry', 'N/A'))}</p>
#         <p><span class="profile-field-label">Current Role(s):</span> {clean_text(profile.get('current role(s)', 'N/A'))}</p>
#         <p><span class="profile-field-label">Location:</span> {clean_text(profile.get('location', 'N/A'))}</p>
#         <details>
#             <summary><span class="profile-field-label">About</span></summary>
#             <p>{clean_text(profile.get('about', 'N/A'))}</p>
#         </details>
#         <p><span class="profile-field-label">Profile Link:</span> <a href="{clean_text(profile.get('profile link', 'N/A'))}" target="_blank">{clean_text(profile.get('profile link', 'N/A'))}</a></p>
#     </div>
#     """, unsafe_allow_html=True)

# # Function to display feature cards
# def display_feature_cards():
#     st.markdown('<h2 class="sub-header">Choose a Tool</h2>', unsafe_allow_html=True)
    
#     # Create three columns for the feature cards
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.markdown("""
#         <div class="feature-card collaborators" id="find-collaborators">
#             <div class="feature-icon">👥</div>
#             <h3 class="feature-title">Find Collaborators</h3>
#             <p class="feature-description">
#                 Discover the perfect partners for your project using our AI-powered profile matching system. 
#                 Find experts with complementary skills and shared interests.
#             </p>
#         </div>
#         """, unsafe_allow_html=True)
        
#         # Add a hidden button that will be triggered by JavaScript
#         if st.button("Find Collaborators", key="btn_collaborators"):
#             st.session_state.active_feature = "find_collaborators"
#             st.rerun()
    
#     with col2:
#         st.markdown("""
#         <div class="feature-card evaluate" id="evaluate-idea">
#             <div class="feature-icon">💡</div>
#             <h3 class="feature-title">Evaluate Your Idea</h3>
#             <p class="feature-description">
#                 Get comprehensive feedback on your business or project idea. Our AI will analyze market potential, 
#                 identify strengths and weaknesses, and suggest improvements.
#             </p>
#         </div>
#         """, unsafe_allow_html=True)
        
#         if st.button("Evaluate Idea", key="btn_evaluate"):
#             st.session_state.active_feature = "evaluate_idea"
#             st.rerun()
    
#     with col3:
#         st.markdown("""
#         <div class="feature-card impact" id="environmental-impact">
#             <div class="feature-icon">🌱</div>
#             <h3 class="feature-title">Calculate Environmental Impact</h3>
#             <p class="feature-description">
#                 Assess the environmental footprint of your project or product. Get insights on sustainability 
#                 metrics and recommendations for reducing your carbon footprint.
#             </p>
#         </div>
#         """, unsafe_allow_html=True)
        
#         if st.button("Environmental Impact", key="btn_impact"):
#             st.session_state.active_feature = "environmental_impact"
#             st.rerun()

# # Function to display the idea evaluation form
# def display_idea_evaluation():
#     st.markdown('<h2 class="sub-header">💡 Evaluate Your Idea</h2>', unsafe_allow_html=True)
    
#     st.markdown("""
#     <div style="background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);">
#         <p style="color: #4B5563; margin-bottom: 1.5rem;">
#             Get comprehensive feedback on your business or project idea. Our AI will analyze market potential, 
#             identify strengths and weaknesses, and suggest improvements.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     with st.form("idea_evaluation_form"):
#         st.text_input("Idea Name", placeholder="Enter a name for your idea or project")
#         st.text_area("Idea Description", placeholder="Describe your idea in detail. What problem does it solve? Who is it for?", height=150)
        
#         col1, col2 = st.columns(2)
#         with col1:
#             st.selectbox("Industry", options=["Technology", "Healthcare", "Education", "Finance", "Retail", "Manufacturing", "Entertainment", "Other"])
#         with col2:
#             st.selectbox("Development Stage", options=["Concept", "Prototype", "MVP", "Early Market", "Growth", "Mature"])
        
#         st.text_area("Target Audience", placeholder="Describe your target audience or customer segment", height=100)
#         st.text_area("Competitive Landscape", placeholder="List any existing competitors or similar solutions", height=100)
        
#         st.submit_button("Evaluate My Idea")
    
#     # Add a back button
#     if st.button("← Back to Tools"):
#         st.session_state.active_feature = None
#         st.rerun()

# # Function to display the environmental impact calculator
# def display_environmental_impact():
#     st.markdown('<h2 class="sub-header">🌱 Calculate Environmental Impact</h2>', unsafe_allow_html=True)
    
#     st.markdown("""
#     <div style="background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);">
#         <p style="color: #4B5563; margin-bottom: 1.5rem;">
#             Assess the environmental footprint of your project or product. Get insights on sustainability 
#             metrics and recommendations for reducing your carbon footprint.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     with st.form("environmental_impact_form"):
#         st.text_input("Project/Product Name", placeholder="Enter the name of your project or product")
        
#         st.subheader("Energy Consumption")
#         col1, col2 = st.columns(2)
#         with col1:
#             st.number_input("Electricity Usage (kWh/month)", min_value=0, value=0)
#         with col2:
#             st.selectbox("Energy Source", options=["Grid Electricity", "Solar", "Wind", "Hydro", "Mixed Renewable", "Other"])
        
#         st.subheader("Materials")
#         st.text_area("Materials Used", placeholder="List the main materials used in your product/project", height=100)
        
#         st.subheader("Transportation")
#         col1, col2 = st.columns(2)
#         with col1:
#             st.number_input("Shipping Distance (km)", min_value=0, value=0)
#         with col2:
#             st.selectbox("Transportation Method", options=["Road", "Rail", "Sea", "Air", "Multiple"])
        
#         st.subheader("Waste & Recycling")
#         st.text_area("Waste Generated", placeholder="Describe the waste generated and how it's handled", height=100)
        
#         st.submit_button("Calculate Impact")
    
#     # Add a back button
#     if st.button("← Back to Tools"):
#         st.session_state.active_feature = None
#         st.rerun()

# # Add the IDEAL_MATCH and COMPATIBLE_MATCH dictionaries
# # IDEAL_MATCH = {
# #     "ESFP": "ESFJ",
# #     "ESTP": "ESTJ",
# #     "ESTJ": "ESTP",
# #     "ESFJ": "ISTP",
# #     "ISTJ": "INFJ",
# #     "ISTP": "ISFP",
# #     "ISFJ": "ESFJ",
# #     "ISFP": "ESFP",
# #     "ENTJ": "INTJ",
# #     "ENTP": "ENTJ",
# #     "ENFJ": "ENFJ",
# #     "ENFP": "ENTJ",
# #     "INTJ": "INTP",
# #     "INTP": "ENTP",
# #     "INFJ": "ISTJ",
# #     "INFP": "INFJ"
# # }

# # # Other compatible matches get a bonus of 0.4
# # COMPATIBLE_MATCH = {
# #     "ESFP": ["ESTP", "ISFP"],
# #     "ESTP": ["ESFJ", "INFJ"],
# #     "ESTJ": ["ESFJ", "ISTJ"],
# #     "ESFJ": ["ESTJ", "ESTP"],
# #     "ISTJ": ["ISTP", "ISFJ"],
# #     "ISTP": ["INFP", "ESFP"],
# #     "ISFJ": ["ISFP", "ISTJ"],
# #     "ISFP": ["ISFJ"],
# #     "ENTJ": ["ENTP", "ENFJ"],
# #     "ENTP": ["ENFP", "ENFJ"],
# #     "ENFJ": ["INFJ", "ENFP"],
# #     "ENFP": ["INTJ", "INTP"],
# #     "INTJ": ["INFJ", "INFP"],
# #     "INTP": ["INFP", "ENFP"],
# #     "INFJ": ["INFP", "INTJ"],
# #     "INFP": ["ISFJ", "ENFJ"]
# # }


# MBTI_COMPATIBILITY = {
#     "ISTJ": ["ESTP", "ESFP", "ISFJ", "INTJ"],
#     "ISFJ": ["ESFP", "ESTP", "ISTJ", "INFJ"],
#     "INFJ": ["ENFP", "ENTP", "INFP", "INTJ"],
#     "INTJ": ["ENFP", "ENTP", "INTP", "ENTJ"],

#     "ISTP": ["ESTJ", "ESFJ", "ENTP", "ISFP"],
#     "ISFP": ["ESFJ", "ESTJ", "ISFP", "INFP"],
#     "INFP": ["INFJ", "INTJ", "ENFP", "ENTP"],
#     "INTP": ["ENTJ", "ESTJ", "ISTJ", "ENFJ"],

#     "ESTP": ["ISTJ", "ISFJ", "ESFP", "ENTP"],
#     "ESFP": ["ISTJ", "ISFJ", "ESTP", "ENFP"],
#     "ENFP": ["INFJ", "INTJ", "ENTP", "ENTJ"],
#     "ENTP": ["INFJ", "INTJ", "ENFP", "ENTJ"],

#     "ESTJ": ["ISTP", "ISFP", "ESFJ", "ENTJ"],
#     "ESFJ": ["ISTP", "ISFP", "ESTJ", "ENFJ"],
#     "ENFJ": ["INFP", "INTP", "ENTP", "ESFJ"],
#     "ENTJ": ["INTP", "ENTP", "ENFP", "ESTJ"]
# }

# # Function to calculate compatibility score
# def calculate_compatibility(user_mbti, target_mbti):
#     if user_mbti == target_mbti:
#         return "High Self-Match"
#     if target_mbti in MBTI_COMPATIBILITY.get(user_mbti, []):
#         return "High Compatibility"
    
#     # Cognitive function compatibility fallback
#     user_functions = {
#         'I': ['Ni', 'Te', 'Fi', 'Se'], 'E': ['Ne', 'Ti', 'Fe', 'Si'],
#         'N': ['Ni', 'Ne'], 'S': ['Si', 'Se'],
#         'T': ['Ti', 'Te'], 'F': ['Fi', 'Fe'],
#         'J': ['Ni', 'Si', 'Ti', 'Fi'], 'P': ['Ne', 'Se', 'Te', 'Fe']
#     }
#     shared_functions = len(set(user_functions[user_mbti[0]]) & set(user_functions[target_mbti[0]]))
#     return f"Moderate Compatibility ({shared_functions} shared cognitive functions)"

# def semantic_search(query, df, index, model, k=5):
#     query_embedding = model.encode([query])
#     distances, indices = index.search(query_embedding, k)
#     return [(df.iloc[idx], distances[0][i]) for i, idx in enumerate(indices[0]) if idx < len(df)]

# def combined_search(semantic_results, user_mbti):
#     results = []
#     for profile, distance in semantic_results:
#         compatibility = calculate_compatibility(user_mbti, profile.get("mbti personality", ""))
#         results.append((profile, distance, compatibility))

#      # Sort by compatibility priority then distance
#     priority = {"High Self-Match": 0, "High Compatibility": 1, "Moderate Compatibility": 2}
#     #return sorted(results, key=lambda x: (priority.get(x[2].split()[0], x[1]))[:5])
#     return sorted(results, key=lambda x: (priority.get(x[2], 3), x[1]))[:5]

    



# # Initialize session state for chat history and active feature
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# if "active_feature" not in st.session_state:
#     st.session_state.active_feature = None

# # Initialize chat model
# chat_model = init_genai()

# # Main application
# st.markdown('<h1 class="main-header">🚀 Innovation Hub</h1>', unsafe_allow_html=True)

# # If no feature is active, show the feature cards
# if st.session_state.active_feature is None:
#     display_feature_cards()
# # If "Find Collaborators" is active, show the profile search
# elif st.session_state.active_feature == "find_collaborators":
#     # Create two columns for the main layout
#     col1, col2 = st.columns([1, 3])
    
#     # Sidebar for configuration
#     with col1:
#         st.markdown('<div class="upload-section">', unsafe_allow_html=True)
#         st.markdown('<h3>📤 Upload Data</h3>', unsafe_allow_html=True)
#         uploaded_file = st.file_uploader("Upload LinkedIn Data (CSV)", type=["csv"])
        
#         if uploaded_file:
#             st.markdown('<div class="info-box">', unsafe_allow_html=True)
#             st.markdown('ℹ️ **Data Processing Info**')
#             st.markdown('- Embeddings are created for semantic search')
#             st.markdown('- FAISS index enables fast similarity search')
#             st.markdown('- All data is processed locally')
#             st.markdown('</div>', unsafe_allow_html=True)
#         st.markdown('</div>', unsafe_allow_html=True)
        
#         # Add a back button
#         if st.button("← Back to Tools"):
#             st.session_state.active_feature = None
#             st.rerun()
    
#     # Main content area
#     with col2:
#         st.markdown('<h2 class="sub-header">👥 Find Collaborators</h2>', unsafe_allow_html=True)
        
#         if uploaded_file:
#             df, index, model, embeddings = load_data(uploaded_file)
            
#             # Search section
#             st.markdown("""
#             <div style="background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); margin-bottom: 1.5rem;">
#                 <p style="color: #4B5563; margin-bottom: 1rem;">
#                     Find the perfect collaborators for your project using our AI-powered profile matching system.
#                     Enter skills, interests, or roles you're looking for to discover potential partners.
#                 </p>
#             </div>
#             """, unsafe_allow_html=True)
            
#             # Create two columns for search input and button
#             search_col1, search_col2 = st.columns([3, 1])
            
#             with search_col1:
#                 query = st.text_input("", placeholder="Enter search query (e.g., 'Python Developer', 'Machine Learning Expert')")
            
#             with search_col2:
#                 search_button = st.button("🔍 Search Profiles")
            
#             # Search results
#             if search_button and query:
#                 with st.spinner("Searching for matching profiles..."):
#                     semantic_results = semantic_search(query, df, index, model)
#                     st.session_state.semantic_results = semantic_results
#                     # Encode query and search
#                     # query_embedding = model.encode([query])
#                     # faiss.normalize_L2(query_embedding)
#                     # k = 5
#                     # distances, indices = index.search(query_embedding, k)
                    
#                     # # Get results
#                     # results = [df.iloc[idx].to_dict() for idx in indices[0]]
#                 # Display semantic results if available
#             if 'semantic_results' in st.session_state:
#                 st.subheader("📊 Top 5 Semantic Matches")
#                 for profile, distance in st.session_state.semantic_results[:5]:
#                     with st.expander(f"{profile['linkedin name']} - Distance: {distance:.2f}"):  # Updated key
#                         st.write(f"**💼 Role:** {profile['description']}")  # Ensure lowercase
#                         st.write(f"**📍 Location:** {profile['location']}")  # Ensure lowercase
#                         st.write(f"**🔗 Profile:** {profile['profile link']}")  # Ensure lowercase
#                         st.write(f"**📝 About:** {profile['about'][:200]}...")  # Ensure lowercase

        
#                 if st.session_state.get('show_mbti', False):
#                     st.session_state.show_mbti = True
                    
#                     # Calculate compatibility scores if MBTI data is available
#                     # if 'mbti' in df.columns:
#                     #     user_mbti = st.session_state.get('user_mbti', 'ENTP')  # Default to ENTP if not set
#                     #     for profile in results:
#                     #         profile_mbti = profile.get('mbti', 'ENTP')  # Default to ENTP if not available
#                     #         profile['compatibility_score'] = calculate_compatibility(user_mbti, profile_mbti)
                    
#                     # # Display results
#                     # st.markdown(f'<h2 class="sub-header">🎯 Top {len(results)} Matching Profiles</h2>', unsafe_allow_html=True)
                    
#                     # # Create columns for results and download button
#                     # res_col1, res_col2 = st.columns([3, 1])


#                     # MBTI Compatibility Section
#             if st.session_state.get('show_mbti', False):
#                 user_mbti = st.text_input("Enter your MBTI type (e.g., INTJ):", max_chars=4).strip().upper()
        
#                 if user_mbti and len(user_mbti) == 4 and user_mbti in MBTI_COMPATIBILITY:
#                     combined_results = combined_search(st.session_state.semantic_results, user_mbti)
            
#                     st.subheader(f"🌟 Top 5 MBTI-Compatible Matches for {user_mbti}")
#                     for profile, distance, compatibility in combined_results:
#                         with st.container():
#                             st.markdown(f"### {profile['LinkedIn Name']}")
#                             col1, col2 = st.columns([1, 3])
#                             with col1:
#                                 st.metric("Match Distance", f"{distance:.2f}")
#                                 st.markdown(f"**MBTI Type:** {profile['mbti personality']}")
#                             with col2:
#                                 st.markdown(f"**Compatibility:** {compatibility}")
#                                 st.markdown(f"**Profile Link:** {profile['Profile Link']}")
#                                 st.markdown(f"**Location:** {profile['Location']}")
#                                 st.markdown("---")
#                 elif user_mbti:
#                     st.error("Invalid MBTI type. Please enter a valid 4-letter MBTI code.")
                    
#                     res_col1, res_col2 = st.columns([3, 1])
#                     with res_col2:
#                         # Generate and offer PDF for download
#                         pdf_output = generate_pdf(combined_results, query)
#                         st.download_button(
#                             label="📄 Download as PDF",
#                             data=pdf_output,
#                             file_name="search_results.pdf",
#                             mime="application/pdf",
#                             key="download_pdf",
#                             help="Download these search results as a PDF document"
#                         )
                    
#                     # Display each profile
#                     for i, profile in enumerate(combined_results, start=1):
#                         display_profile_card(profile, i)
                        
#                         # Display compatibility score if available
#                         if 'compatibility_score' in profile:
#                             st.markdown(f"""
#                             <div style="background-color: #DBEAFE; padding: 0.5rem; border-radius: 8px; margin-bottom: 1rem;">
#                                 <p style="color: #1E40AF; font-weight: 600;">Compatibility Score: {profile['compatibility_score']:.2f}</p>
#                             </div>
#                             """, unsafe_allow_html=True)
            
#             # Assistant Chatbot
#             st.markdown('<h2 class="sub-header">🤖 Assistant Chatbot</h2>', unsafe_allow_html=True)
            
#             # Display chat history
#             for message in st.session_state.chat_history:
#                 if message["role"] == "user":
#                     st.markdown(f"""
#                     <div class="chat-message user-message">
#                         <p><strong>You:</strong></p>
#                         <p>{message["content"]}</p>
#                     </div>
#                     """, unsafe_allow_html=True)
#                 else:
#                     st.markdown(f"""
#                     <div class="chat-message assistant-message">
#                         <p><strong>Assistant:</strong></p>
#                         <p>{message["content"]}</p>
#                     </div>
#                     """, unsafe_allow_html=True)
            
#             # Chat input
#             user_input = st.text_input("", placeholder="Ask Assistant about networking, career, or interests...", key="chat_input")
            
#             # Create two columns for chat button and clear button
#             chat_col1, chat_col2 = st.columns([1, 3])
            
#             with chat_col1:
#                 chat_button = st.button("💬 Send Message")
            
#             with chat_col2:
#                 if st.button("🗑️ Clear Chat"):
#                     st.session_state.chat_history = []
#                     st.rerun()
            
#             if chat_button and user_input:
#                 # Add user message to history
#                 st.session_state.chat_history.append({"role": "user", "content": user_input})
                
#                 # Get response from Gemini
#                 with st.spinner("Assistant is thinking..."):
#                     try:
#                         # Create a context-aware prompt
#                         context = f"""
#                         You are a helpful career and networking assistant. The user has uploaded LinkedIn profile data and is looking for advice.
                        
#                         User question: {user_input}
#                         """
                        
#                         response = chat_model.generate_content(context)
#                         assistant_response = response.text
                        
#                         # Add assistant response to history
#                         st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                        
#                     except Exception as e:
#                         st.error(f"Error generating response: {str(e)}")
#                         assistant_response = "I'm sorry, I encountered an error while processing your request. Please try again."
#                         st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                
#                 # Rerun to update the chat display
#                 st.rerun()
        
#         else:
#             # Display welcome message when no file is uploaded
#             st.markdown("""
#             <div style="text-align: center; padding: 2rem; background-color: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); margin-top: 1rem;">
#                 <h2 style="color: #2563EB; font-size: 1.8rem; margin-bottom: 1.5rem;">Find Your Perfect Collaborators</h2>
#                 <p style="font-size: 1.2rem; margin: 1rem 0; color: #4B5563;">Upload a CSV file containing LinkedIn profiles to get started.</p>
#                 <div style="background-color: #DBEAFE; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0; text-align: left;">
#                     <h3 style="color: #1E40AF; font-size: 1.2rem; margin-bottom: 1rem;">How it works:</h3>
#                     <ol style="color: #4B5563; margin-left: 1.5rem;">
#                         <li style="margin-bottom: 0.5rem;">Upload your LinkedIn connections CSV export</li>
#                         <li style="margin-bottom: 0.5rem;">Search for specific skills, industries, or keywords</li>
#                         <li style="margin-bottom: 0.5rem;">Review the top matching profiles</li>
#                         <li style="margin-bottom: 0.5rem;">Connect with potential collaborators</li>
#                     </ol>
#                 </div>
#                 <p style="margin-top: 1rem; font-style: italic; color: #6B7280;">👈 Use the uploader in the sidebar to begin</p>
#             </div>
#             """, unsafe_allow_html=True)
# # If "Evaluate Idea" is active, show the idea evaluation form
# elif st.session_state.active_feature == "evaluate_idea":
#     display_idea_evaluation()
# # If "Environmental Impact" is active, show the environmental impact calculator
# elif st.session_state.active_feature == "environmental_impact":
#     display_environmental_impact()

# # Add footer
# st.markdown("""
# <div class="footer">
#     <p>Innovation Hub - Collaborate, Evaluate, Impact © 2025 | Built with Streamlit</p>
# </div>
# """, unsafe_allow_html=True)

# # Add JavaScript for card click functionality
# st.markdown("""
# <script>
#     // This script would handle card clicks, but Streamlit doesn't support custom JavaScript
#     // The functionality is handled by the buttons instead
# </script>
# """, unsafe_allow_html=True)
















import re
import streamlit as st
from typing import Tuple, Any
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from pathlib import Path
from fpdf import FPDF
import io
import time

st.set_page_config(
    page_title="Innovation Hub - Collaborate, Evaluate, Impact",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --primary: #2563EB;
        --primary-light: #DBEAFE;
        --primary-dark: #1E40AF;
        --secondary: #10B981;
        --secondary-light: #D1FAE5;
        --secondary-dark: #047857;
        --accent: #8B5CF6;
        --accent-light: #EDE9FE;
        --accent-dark: #6D28D9;
        --neutral-50: #F9FAFB;
        --neutral-100: #F3F4F6;
        --neutral-200: #E5E7EB;
        --neutral-300: #D1D5DB;
        --neutral-600: #4B5563;
        --neutral-700: #374151;
        --neutral-800: #1F2937;
        --neutral-900: #111827;
        --danger: #EF4444;
        --warning: #F59E0B;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
    }
    
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: var(--neutral-800) !important;
        margin-bottom: 1.5rem !important;
        letter-spacing: -0.025em !important;
    }
    
    .sub-header {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: var(--neutral-800) !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        letter-spacing: -0.025em !important;
    }
    
    .feature-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 10px 15px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s, box-shadow 0.2s;
        cursor: pointer;
        border: 1px solid var(--neutral-200);
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
    }
    
    .feature-card.collaborators {
        border-top: 5px solid var(--primary);
    }
    
    .feature-card.evaluate {
        border-top: 5px solid var(--secondary);
    }
    
    .feature-card.impact {
        border-top: 5px solid var(--accent);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.75rem !important;
        color: var(--neutral-800) !important;
    }
    
    .feature-description {
        color: var(--neutral-600) !important;
        font-size: 1rem !important;
        line-height: 1.5 !important;
    }
    
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 5px solid var(--primary);
    }
    
    .profile-name {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        color: var(--primary-dark) !important;
        margin-bottom: 0.75rem !important;
    }
    
    .profile-field-label {
        font-weight: 600 !important;
        color: var(--neutral-700) !important;
    }
    
    .stButton>button {
        background-color: var(--primary);
        color: white;
        font-weight: 500;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        transition: background-color 0.2s;
    }
    
    .stButton>button:hover {
        background-color: var(--primary-dark);
    }
    
    .download-btn {
        background-color: var(--secondary) !important;
    }
    
    .download-btn:hover {
        background-color: var(--secondary-dark) !important;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    
    .user-message {
        background-color: var(--primary-light);
        border-left: 5px solid var(--primary);
    }
    
    .assistant-message {
        background-color: var(--secondary-light);
        border-left: 5px solid var(--secondary);
    }
    
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid var(--neutral-300);
        padding: 0.75rem 1rem;
        font-size: 1rem;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 2px var(--primary-light);
    }
    
    .upload-section {
        padding: 1.5rem;
        background-color: white;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .info-box {
        padding: 1rem;
        background-color: var(--secondary-light);
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid var(--secondary);
    }
    
    .error-box {
        padding: 1rem;
        background-color: #FFEBEE;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid var(--danger);
    }
    
    .stProgress > div > div > div > div {
        background-color: var(--primary);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: var(--neutral-100);
        border-radius: 8px 8px 0 0;
        gap: 1px;
        padding: 10px 16px;
        color: var(--neutral-700);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: var(--primary);
        border-top: 3px solid var(--primary);
        font-weight: 600;
    }
    
    .stFileUploader > div > button {
        background-color: var(--primary);
    }
    
    .stFileUploader > div > button:hover {
        background-color: var(--primary-dark);
    }
    
    .welcome-container {
        text-align: center;
        padding: 3rem 2rem;
        background-color: white;
        border-radius: 12px;
        margin-top: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .welcome-title {
        color: var(--primary);
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }
    
    .welcome-subtitle {
        font-size: 1.2rem;
        margin: 1rem 0;
        color: var(--neutral-700);
    }
    
    .welcome-features {
        text-align: left;
        display: inline-block;
        margin: 1.5rem 0;
    }
    
    .welcome-features li {
        margin-bottom: 0.5rem;
        color: var(--neutral-700);
    }
    
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--neutral-200);
        color: var(--neutral-600);
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-1.5-flash"
API_KEY = "AIzaSyBYfzDovXQK6E8jZsrOBieoSY_X6jCUktU"

@st.cache_resource
def init_genai() -> Any:
    genai.configure(api_key=API_KEY)
    return genai.GenerativeModel(GEMINI_MODEL)

@st.cache_resource
def load_encoder() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def load_data(uploaded_file: Path) -> Tuple[pd.DataFrame, Any, SentenceTransformer, np.ndarray]:
    try:
        with st.spinner("Processing data and creating embeddings..."):
            progress_bar = st.progress(0)
            progress_bar.progress(10)
            time.sleep(0.5)
            
            df = pd.read_csv(uploaded_file, dtype_backend="pyarrow", encoding="ISO-8859-1")
            df.columns = df.columns.str.replace(r"ï»¿", "", regex=True).str.strip().str.lower()
            
            progress_bar.progress(30)
            time.sleep(0.5)
            
            df["combined_text"] = df.astype(str).agg(" ".join, axis=1)
            
            progress_bar.progress(40)
            time.sleep(0.5)
            
            model = load_encoder()
            
            progress_bar.progress(50)
            time.sleep(0.5)
            
            st.info("Generating embeddings for semantic search. This may take a moment...")
            
            embeddings = model.encode(
                df["combined_text"].tolist(),
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            progress_bar.progress(80)
            time.sleep(0.5)
            
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            
            progress_bar.progress(100)
            time.sleep(0.5)
            
            progress_bar.empty()
            
            st.success(f"✅ Successfully loaded {len(df)} profiles and created search index!")
            
            return df, index, model, embeddings
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        raise

def clean_text(text):
    if pd.isna(text):
        return "N/A"
    return re.sub(r"[^\x20-\x7E]", "", str(text))

def generate_pdf(semantic_results, combined_results, query, user_mbti):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Innovation Hub Collaboration Report", ln=True, align="C")
    pdf.set_font("Arial", "I", 12)
    pdf.cell(200, 10, f"Search Query: '{query}' | MBTI Filter: {user_mbti}", ln=True, align="C")
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(15)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Semantic Search Results", ln=True)
    pdf.ln(8)
    
    for idx, (profile, distance) in enumerate(semantic_results, start=1):
        pdf = add_profile_to_pdf(pdf, profile, distance, idx, "Semantic")
    
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "MBTI-Compatible Results", ln=True)
    pdf.ln(8)
    
    for idx, (profile, distance, compatibility) in enumerate(combined_results, start=1):
        pdf = add_profile_to_pdf(pdf, profile, distance, idx, "MBTI", compatibility)
    
    pdf.set_y(-15)
    pdf.set_font("Arial", "I", 8)
    pdf.cell(0, 10, f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}", 0, 0, "C")
    
    pdf_str = pdf.output(dest="S")
    return io.BytesIO(pdf_str.encode("latin1"))

def add_profile_to_pdf(pdf, profile, distance, index, search_type, compatibility=None):
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, f"{search_type} Match #{index}", ln=True)
    pdf.set_font("Arial", "", 10)
    
    pdf.cell(40, 8, "Name:", 0)
    pdf.multi_cell(0, 8, f"{clean_text(profile.get('linkedin name', 'N/A'))}")
    
    pdf.cell(40, 8, "Distance Score:", 0)
    pdf.multi_cell(0, 8, f"{distance:.2f}")
    
    if compatibility:
        pdf.cell(40, 8, "Compatibility:", 0)
        pdf.multi_cell(0, 8, compatibility)
    
    pdf.cell(40, 8, "Industry:", 0)
    pdf.multi_cell(0, 8, f"{clean_text(profile.get('industry', 'N/A'))}")
    
    pdf.cell(40, 8, "Current Role:", 0)
    pdf.multi_cell(0, 8, f"{clean_text(profile.get('current role(s)', 'N/A'))}")
    
    pdf.cell(40, 8, "Location:", 0)
    pdf.multi_cell(0, 8, f"{clean_text(profile.get('location', 'N/A'))}")
    
    pdf.cell(40, 8, "Profile Link:", 0)
    pdf.set_text_color(0, 0, 255)
    pdf.multi_cell(0, 8, f"{clean_text(profile.get('profile link', 'N/A'))}")
    pdf.set_text_color(0, 0, 0)
    
    pdf.ln(10)
    return pdf
def display_profile_card(profile, index):
    st.markdown(f"""
    <div class="card">
        <p class="profile-name">{index}. {clean_text(profile.get('linkedin name', 'N/A'))}</p>
        <p><span class="profile-field-label">Industry:</span> {clean_text(profile.get('industry', 'N/A'))}</p>
        <p><span class="profile-field-label">Current Role(s):</span> {clean_text(profile.get('current role(s)', 'N/A'))}</p>
        <p><span class="profile-field-label">Location:</span> {clean_text(profile.get('location', 'N/A'))}</p>
        <details>
            <summary><span class="profile-field-label">About</span></summary>
            <p>{clean_text(profile.get('about', 'N/A'))}</p>
        </details>
        <p><span class="profile-field-label">Profile Link:</span> <a href="{clean_text(profile.get('profile link', 'N/A'))}" target="_blank">{clean_text(profile.get('profile link', 'N/A'))}</a></p>
    </div>
    """, unsafe_allow_html=True)

def display_feature_cards():
    st.markdown('<h2 class="sub-header">Choose a Tool</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card collaborators" id="find-collaborators">
            <div class="feature-icon">👥</div>
            <h3 class="feature-title">Find Collaborators</h3>
            <p class="feature-description">
                Discover the perfect partners for your project using our AI-powered profile matching system. 
                Find experts with complementary skills and shared interests.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Find Collaborators", key="btn_collaborators"):
            st.session_state.active_feature = "find_collaborators"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="feature-card evaluate" id="evaluate-idea">
            <div class="feature-icon">💡</div>
            <h3 class="feature-title">Evaluate Your Idea</h3>
            <p class="feature-description">
                Get comprehensive feedback on your business or project idea. Our AI will analyze market potential, 
                identify strengths and weaknesses, and suggest improvements.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Evaluate Idea", key="btn_evaluate"):
            st.session_state.active_feature = "evaluate_idea"
            st.rerun()
    
    with col3:
        st.markdown("""
        <div class="feature-card impact" id="environmental-impact">
            <div class="feature-icon">🌱</div>
            <h3 class="feature-title">Calculate Environmental Impact</h3>
            <p class="feature-description">
                Assess the environmental footprint of your project or product. Get insights on sustainability 
                metrics and recommendations for reducing your carbon footprint.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Environmental Impact", key="btn_impact"):
            st.session_state.active_feature = "environmental_impact"
            st.rerun()

def display_idea_evaluation():
    st.markdown('<h2 class="sub-header">💡 Evaluate Your Idea</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);">
        <p style="color: #4B5563; margin-bottom: 1.5rem;">
            Get comprehensive feedback on your business or project idea. Our AI will analyze market potential, 
            identify strengths and weaknesses, and suggest improvements.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("idea_evaluation_form"):
        st.text_input("Idea Name", placeholder="Enter a name for your idea or project")
        st.text_area("Idea Description", placeholder="Describe your idea in detail. What problem does it solve? Who is it for?", height=150)
        
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("Industry", options=["Technology", "Healthcare", "Education", "Finance", "Retail", "Manufacturing", "Entertainment", "Other"])
        with col2:
            st.selectbox("Development Stage", options=["Concept", "Prototype", "MVP", "Early Market", "Growth", "Mature"])
        
        st.text_area("Target Audience", placeholder="Describe your target audience or customer segment", height=100)
        st.text_area("Competitive Landscape", placeholder="List any existing competitors or similar solutions", height=100)
        
        st.submit_button("Evaluate My Idea")
    
    if st.button("← Back to Tools"):
        st.session_state.active_feature = None
        st.rerun()

def display_environmental_impact():
    st.markdown('<h2 class="sub-header">🌱 Calculate Environmental Impact</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);">
        <p style="color: #4B5563; margin-bottom: 1.5rem;">
            Assess the environmental footprint of your project or product. Get insights on sustainability 
            metrics and recommendations for reducing your carbon footprint.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("environmental_impact_form"):
        st.text_input("Project/Product Name", placeholder="Enter the name of your project or product")
        
        st.subheader("Energy Consumption")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Electricity Usage (kWh/month)", min_value=0, value=0)
        with col2:
            st.selectbox("Energy Source", options=["Grid Electricity", "Solar", "Wind", "Hydro", "Mixed Renewable", "Other"])
        
        st.subheader("Materials")
        st.text_area("Materials Used", placeholder="List the main materials used in your product/project", height=100)
        
        st.subheader("Transportation")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Shipping Distance (km)", min_value=0, value=0)
        with col2:
            st.selectbox("Transportation Method", options=["Road", "Rail", "Sea", "Air", "Multiple"])
        
        st.subheader("Waste & Recycling")
        st.text_area("Waste Generated", placeholder="Describe the waste generated and how it's handled", height=100)
        
        st.submit_button("Calculate Impact")
    
    if st.button("← Back to Tools"):
        st.session_state.active_feature = None
        st.rerun()

MBTI_COMPATIBILITY = {
    "ISTJ": ["ESTP", "ESFP", "ISFJ", "INTJ"],
    "ISFJ": ["ESFP", "ESTP", "ISTJ", "INFJ"],
    "INFJ": ["ENFP", "ENTP", "INFP", "INTJ"],
    "INTJ": ["ENFP", "ENTP", "INTP", "ENTJ"],
    "ISTP": ["ESTJ", "ESFJ", "ENTP", "ISFP"],
    "ISFP": ["ESFJ", "ESTJ", "ISFP", "INFP"],
    "INFP": ["INFJ", "INTJ", "ENFP", "ENTP"],
    "INTP": ["ENTJ", "ESTJ", "ISTJ", "ENFJ"],
    "ESTP": ["ISTJ", "ISFJ", "ESFP", "ENTP"],
    "ESFP": ["ISTJ", "ISFJ", "ESTP", "ENFP"],
    "ENFP": ["INFJ", "INTJ", "ENTP", "ENTJ"],
    "ENTP": ["INFJ", "INTJ", "ENFP", "ENTJ"],
    "ESTJ": ["ISTP", "ISFP", "ESFJ", "ENTJ"],
    "ESFJ": ["ISTP", "ISFP", "ESTJ", "ENFJ"],
    "ENFJ": ["INFP", "INTP", "ENTP", "ESFJ"],
    "ENTJ": ["INTP", "ENTP", "ENFP", "ESTJ"]
}

def calculate_compatibility(user_mbti, target_mbti):
    if user_mbti == target_mbti:
        return "High Self-Match"
    if target_mbti in MBTI_COMPATIBILITY.get(user_mbti, []):
        return "High Compatibility"
    user_functions = {
        'I': ['Ni', 'Te', 'Fi', 'Se'], 'E': ['Ne', 'Ti', 'Fe', 'Si'],
        'N': ['Ni', 'Ne'], 'S': ['Si', 'Se'],
        'T': ['Ti', 'Te'], 'F': ['Fi', 'Fe'],
        'J': ['Ni', 'Si', 'Ti', 'Fi'], 'P': ['Ne', 'Se', 'Te', 'Fe']
    }
    shared_functions = len(set(user_functions[user_mbti[0]]) & set(user_functions[target_mbti[0]]))
    return f"Moderate Compatibility ({shared_functions} shared cognitive functions)"

def semantic_search(query, df, index, model, k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [(df.iloc[idx], distances[0][i]) for i, idx in enumerate(indices[0]) if idx < len(df)]

def combined_search(semantic_results, user_mbti):
    results = []
    for profile, distance in semantic_results:
        compatibility = calculate_compatibility(user_mbti, profile.get("mbti personality", ""))
        results.append((profile, distance, compatibility))
    priority = {"High Self-Match": 0, "High Compatibility": 1, "Moderate Compatibility": 2}
    return sorted(results, key=lambda x: (priority.get(x[2], 3), x[1]))[:5]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "active_feature" not in st.session_state:
    st.session_state.active_feature = None

chat_model = init_genai()

st.markdown('<h1 class="main-header">🚀 Innovation Hub</h1>', unsafe_allow_html=True)

if st.session_state.active_feature is None:
    display_feature_cards()
elif st.session_state.active_feature == "find_collaborators":
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown('<h3>📤 Upload Data</h3>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload LinkedIn Data (CSV)", type=["csv"])
        
        if uploaded_file:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown('ℹ️ **Data Processing Info**')
            st.markdown('- Embeddings are created for semantic search')
            st.markdown('- FAISS index enables fast similarity search')
            st.markdown('- All data is processed locally')
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("← Back to Tools"):
            st.session_state.active_feature = None
            st.rerun()
    
    with col2:
        st.markdown('<h2 class="sub-header">👥 Find Collaborators</h2>', unsafe_allow_html=True)
    
    if uploaded_file:
        df, index, model, embeddings = load_data(uploaded_file)
        
        st.markdown("""
        <div style="background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); margin-bottom: 1.5rem;">
            <p style="color: #4B5563; margin-bottom: 1rem;">
                Find the perfect collaborators for your project using our AI-powered profile matching system.
                Enter skills, interests, or roles you're looking for to discover potential partners.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        search_col1, search_col2 = st.columns([3, 1])
        with search_col1:
            query = st.text_input("", placeholder="Enter search query (e.g., 'Python Developer', 'Machine Learning Expert')")
        with search_col2:
            search_button = st.button("🔍 Search Profiles")

        if search_button and query:
            with st.spinner("Searching for matching profiles..."):
                semantic_results = semantic_search(query, df, index, model)
                st.session_state.semantic_results = semantic_results
                st.session_state.show_mbti = False

        if 'semantic_results' in st.session_state:
            st.subheader("📊 Top 5 Semantic Matches")
            for profile, distance in st.session_state.semantic_results[:5]:
                with st.expander(f"{profile['linkedin name']} - Distance: {distance:.2f}"):
                    st.write(f"**💼 Role:** {profile['description']}")
                    st.write(f"**📍 Location:** {profile['location']}")
                    st.write(f"**🔗 Profile:** {profile['profile link']}")
                    st.write(f"**📝 About:** {profile['about'][:200]}...")

            if not st.session_state.get('show_mbti', False):
                if st.button("🧠 Add MBTI Compatibility Filter", type="primary"):
                    st.session_state.show_mbti = True

            if st.session_state.get('show_mbti', False):
                user_mbti = st.text_input("Enter your MBTI type (e.g., INTJ):", max_chars=4).strip().upper()
                
                if user_mbti:
                    if len(user_mbti) == 4 and user_mbti in MBTI_COMPATIBILITY:
                        combined_results = combined_search(st.session_state.semantic_results, user_mbti)
                        
                        st.subheader(f"🌟 Top 5 MBTI-Compatible Matches for {user_mbti}")
                        for profile, distance, compatibility in combined_results:
                            with st.expander(f"{profile['linkedin name']} - {compatibility}"):
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    st.metric("Semantic Distance", f"{distance:.2f}")
                                    st.metric("MBTI Type", profile.get('mbti personality', 'N/A'))
                                with col2:
                                    st.write(f"**Industry:** {profile.get('industry', 'N/A')}")
                                    st.write(f"**Current Role:** {profile.get('current role(s)', 'N/A')}")
                                    st.write(f"**Location:** {profile.get('location', 'N/A')}")
                                    st.write(f"**Profile Link:** {profile.get('profile link', 'N/A')}")
                        
                        if st.button("📄 Download Full Report"):
                            pdf_output = generate_pdf(
                                semantic_results=st.session_state.semantic_results[:5],
                                combined_results=combined_results,
                                query=query,
                                user_mbti=user_mbti
                            )
                            st.download_button(
                                label="Download Full Report",
                                data=pdf_output,
                                file_name="full_report.pdf",
                                mime="application/pdf"
                            )
                    else:
                        st.error("Please enter a valid 4-letter MBTI type (e.g., INTJ, ENFP)")

            st.markdown('<h2 class="sub-header">🤖 Assistant Chatbot</h2>', unsafe_allow_html=True)
            
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <p><strong>You:</strong></p>
                        <p>{message["content"]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <p><strong>Assistant:</strong></p>
                        <p>{message["content"]}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            user_input = st.text_input("", placeholder="Ask Assistant about networking, career, or interests...", key="chat_input")
            
            chat_col1, chat_col2 = st.columns([1, 3])
            with chat_col1:
                chat_button = st.button("💬 Send Message")
            with chat_col2:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()
            
            if chat_button and user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                with st.spinner("Assistant is thinking..."):
                    try:
                        context = f"You are a helpful career and networking assistant. The user has uploaded LinkedIn profile data and is looking for advice.\n\nUser question: {user_input}"
                        response = chat_model.generate_content(context)
                        assistant_response = response.text
                        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        assistant_response = "I'm sorry, I encountered an error while processing your request. Please try again."
                        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                st.rerun()
        
        else:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background-color: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); margin-top: 1rem;">
                <h2 style="color: #2563EB; font-size: 1.8rem; margin-bottom: 1.5rem;">Find Your Perfect Collaborators</h2>
                <p style="font-size: 1.2rem; margin: 1rem 0; color: #4B5563;">Upload a CSV file containing LinkedIn profiles to get started.</p>
                <div style="background-color: #DBEAFE; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0; text-align: left;">
                    <h3 style="color: #1E40AF; font-size: 1.2rem; margin-bottom: 1rem;">How it works:</h3>
                    <ol style="color: #4B5563; margin-left: 1.5rem;">
                        <li style="margin-bottom: 0.5rem;">Upload your LinkedIn connections CSV export</li>
                        <li style="margin-bottom: 0.5rem;">Search for specific skills, industries, or keywords</li>
                        <li style="margin-bottom: 0.5rem;">Review the top matching profiles</li>
                        <li style="margin-bottom: 0.5rem;">Connect with potential collaborators</li>
                    </ol>
                </div>
                <p style="margin-top: 1rem; font-style: italic; color: #6B7280;">👈 Use the uploader in the sidebar to begin</p>
            </div>
            """, unsafe_allow_html=True)

elif st.session_state.active_feature == "evaluate_idea":
    display_idea_evaluation()
elif st.session_state.active_feature == "environmental_impact":
    display_environmental_impact()

st.markdown("""
<div class="footer">
    <p>Innovation Hub - Collaborate, Evaluate, Impact © 2025 | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<script>
</script>
""", unsafe_allow_html=True)




