# import streamlit as st
# from typing import Tuple, Any
# import pandas as pd
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# import google.generativeai as genai
# from pathlib import Path
# from faiss import swigfaiss as faiss
# from faiss import normalize_L2  # Correct import for normalize_L2
# dimension = 768  # Set the dimension to a default value or calculate it based on your data
# index = faiss.IndexFlatL2(dimension)

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

# # Correct imports at top:
# import faiss
# from faiss import normalize_L2  # Keep this if needed, but better to use faiss.normalize_L2

# # [...] rest of imports

# # In load_data():
# @st.cache_data
# def load_data(uploaded_file: Path) -> Tuple[pd.DataFrame, Any, SentenceTransformer, np.ndarray]:
#     try:
#         df = pd.read_csv(uploaded_file, dtype_backend="pyarrow")
#         df["combined_text"] = df.astype(str).agg(" ".join, axis=1)
        
#         model = load_encoder()
#         embeddings = model.encode(
#             df["combined_text"].tolist(),
#             batch_size=32,
#             show_progress_bar=True,
#             convert_to_numpy=True
#         )
        
#         # FAISS Index Handling
#         dimension = embeddings.shape[1]
#         embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
#         faiss.normalize_L2(embeddings)
        
#         index = faiss.IndexFlatL2(dimension)
#         index.add(embeddings)
        
#         return df, index, model, embeddings
#     except Exception as e:
#         st.error(f"Error loading data: {e}")
#         return None, None, None, None

# # Initialize app
# st.set_page_config(
#     page_title="Profile Search & Assistant Chatbot",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Initialize chat model
# chat_model = init_genai()

# # Streamlit UI setup
# st.title("🔍 AI-Powered Profile Search & Assistant Chatbot")

# # Sidebar for configuration
# st.sidebar.header("Upload Data")

# # File upload for dataset
# uploaded_file = st.sidebar.file_uploader("LinkedinData.csv", type=["csv"])

# if uploaded_file:
#     df, index, model, embeddings = load_data(uploaded_file)

#     # Search bar
#     query = st.text_input("Enter search query (e.g., 'Python Developer', 'Machine Learning Expert'):")

#     if st.button("Search"):
#         if query:
#             query_embedding = model.encode([query])
#             k = 5
#             distances, indices = index.search(query_embedding, k)

#             # Display results
#             st.subheader("🎯 Top 5 Matching Profiles")
#             for idx in indices[0]:
#                 profile = df.iloc[idx]
#                 st.write(f"**👤 Name:** {profile['LinkedIn Name']}")
#                 st.write(f"**💼 Description:** {profile['Description']}")
#                 st.write(f"**🛠 About:** {profile['About']}")
#                 st.write(f"**📆 Current Role(s):** {profile['Current Role(s)']} years")
#                 st.write(f"**📍 Location:** {profile['Location']}")
#                 st.write(f"**📍 Profile Link:** {profile['Profile Link']}")
#                 st.markdown("---")  # Separator

#     # Assistant Chatbot for finding like-minded people
#     st.subheader("🤖 Assistant Chatbot")
#     user_input = st.text_input("Ask Assistant about networking, career, or interests:")

#     if st.button("Chat with Assistant"):
#         if user_input:
#             response = chat_model.generate_content(user_input)
#             st.write("**Assistant:**")
#             st.write(response.text)




import re
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import plotly.express as px
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Configuration
MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-1.5-flash"
API_KEY = "AIzaSyBYfzDovXQK6E8jZsrOBieoSY_X6jCUktU"

# Initialize models
@st.cache_resource
def init_genai():
    genai.configure(api_key=API_KEY)
    return genai.GenerativeModel(GEMINI_MODEL)

@st.cache_resource
def load_encoder():
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        df["combined_text"] = df.astype(str).agg(" ".join, axis=1)
        model = load_encoder()
        embeddings = model.encode(df["combined_text"].tolist(), convert_to_numpy=True)
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return df, index, model, embeddings
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return None, None, None, None

def generate_validation_report(results, filename="validation_report.pdf"):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    # PDF Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 750, "Search Validation Report")
    
    # Metrics Section
    c.setFont("Helvetica", 12)
    y_position = 700
    c.drawString(72, y_position, f"Query: {results['query']}")
    y_position -= 20
    c.drawString(72, y_position, f"Precision@5: {results['precision']:.2f}")
    y_position -= 20
    c.drawString(72, y_position, f"Recall@5: {results['recall']:.2f}")
    
    # Top Candidates Table
    y_position -= 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, y_position, "Top Candidates Comparison")
    y_position -= 30
    
    headers = ["Rank", "FAISS Candidate", "Gemini Candidate", "Match"]
    col_widths = [50, 200, 200, 50]
    
    # Table Header
    c.setFont("Helvetica-Bold", 12)
    for i, header in enumerate(headers):
        c.drawString(72 + sum(col_widths[:i]), y_position, header)
    y_position -= 20
    
    # Table Rows
    c.setFont("Helvetica", 12)
    for idx, row in enumerate(results['comparison']):
        c.drawString(72, y_position, str(idx+1))
        c.drawString(122, y_position, row['faiss'])
        c.drawString(322, y_position, row['gemini'])
        c.drawString(522, y_position, "✓" if row['match'] else "✗")
        y_position -= 15
    
    c.save()
    buffer.seek(0)
    return buffer

def calculate_metrics(faiss_top, gemini_top, full_candidates):
    # Precision and Recall calculation
    relevant = set(gemini_top)
    retrieved = set(faiss_top)
    
    tp = len(relevant & retrieved)
    precision = tp / len(retrieved) if len(retrieved) > 0 else 0
    recall = tp / len(relevant) if len(relevant) > 0 else 0
    
    # Comparison matrix
    comparison = []
    for i in range(5):
        faiss_name = full_candidates.iloc[faiss_top[i]]['LinkedIn Name'] if i < len(faiss_top) else "N/A"
        gemini_name = full_candidates.iloc[gemini_top[i]]['LinkedIn Name'] if i < len(gemini_top) else "N/A"
        comparison.append({
            'faiss': faiss_name,
            'gemini': gemini_name,
            'match': faiss_top[i] in gemini_top[:i+1]
        })
    
    return {
        'precision': precision,
        'recall': recall,
        'comparison': comparison
    }

def visualize_results(faiss_scores, gemini_scores):
    fig = px.bar(
        x=["Precision@5", "Recall@5"],
        y=[faiss_scores['precision'], faiss_scores['recall']],
        labels={'x': 'Metric', 'y': 'Value'},
        title="Search Performance Metrics",
        color=["FAISS", "FAISS"],
        barmode="group"
    )
    fig.add_bar(
        x=["Precision@5", "Recall@5"],
        y=[gemini_scores['precision'], gemini_scores['recall']],
        name="Gemini"
    )
    st.plotly_chart(fig)

# Streamlit UI
st.set_page_config(page_title="Advanced Profile Validator", layout="wide")
st.title("🔍 Semantic Search Validation Suite")

uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])
query = st.text_input("Enter Search Query:")

if uploaded_file and query:
    df, index, model, embeddings = load_data(uploaded_file)
    
    if df is not None and st.button("Run Validation"):
        # Semantic Search
        query_embedding = model.encode([query])
        distances, indices = index.search(query_embedding, 20)
        faiss_top20 = indices[0].tolist()
        faiss_scores = 1 - (distances[0] / np.max(distances[0]))
        
        # Gemini Validation
        gemini_model = init_genai()
        candidate_pool = df.iloc[faiss_top20]
        validation_prompt = f"""
        Given the query '{query}', analyze these profiles and return the indices of 
        the top 5 most relevant candidates in exact format: [1,2,3,4,5]
        Candidates:\n{candidate_pool.to_string()}
        """
        gemini_response = gemini_model.generate_content(validation_prompt)
        gemini_top5 = [int(idx) for idx in re.findall(r'\b\d+\b', gemini_response.text) if int(idx) in candidate_pool.index][:5]
        
        # Calculate Metrics
        faiss_metrics = calculate_metrics(faiss_top20[:5], gemini_top5, df)
        gemini_metrics = calculate_metrics(gemini_top5, faiss_top20[:5], df)
        
        # Visualization
        visualize_results(faiss_metrics, gemini_metrics)
        
        # Detailed Comparison
        st.subheader("Candidate Matching Matrix")
        comparison_df = pd.DataFrame({
            'Rank': range(1,6),
            'FAISS Candidate': [df.iloc[idx]['LinkedIn Name'] for idx in faiss_top20[:5]],
            'Gemini Candidate': [df.iloc[idx]['LinkedIn Name'] for idx in gemini_top5[:5]],
            'Match': [idx in gemini_top5 for idx in faiss_top20[:5]]
        })
        st.dataframe(comparison_df)
        
        # Generate Report
        report_data = {
            'query': query,
            'precision': faiss_metrics['precision'],
            'recall': faiss_metrics['recall'],
            'comparison': faiss_metrics['comparison']
        }
        pdf_buffer = generate_validation_report(report_data)
        st.download_button(
            label="Download Full Report (PDF)",
            data=pdf_buffer,
            file_name="search_validation_report.pdf",
            mime="application/pdf"
        )

# Chatbot Integration
st.sidebar.subheader("Analysis Assistant")
analysis_query = st.sidebar.text_input("Ask about the results:")
if analysis_query and uploaded_file:
    gemini_model = init_genai()
    context = f"""
    Current analysis context:
    - Query: {query}
    - Dataset: {uploaded_file.name}
    - Top FAISS Candidate: {df.iloc[faiss_top20[0]]['LinkedIn Name']}
    """
    response = gemini_model.generate_content(context + analysis_query)
    st.sidebar.write("Assistant:", response.text)
# Chatbot section remains similar



# import streamlit as st
# import pandas as pd
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer

# # Load MBTI compatibility rules
# MBTI_COMPATIBILITY = {
#     "ESFP": ["ESFJ", "ESTP", "ISFP"],
#     "ESTP": ["ESTJ", "ESFP", "INFJ"],
#     "ESTJ": ["ESTP", "ESFJ", "ISTJ"],
#     "ESFJ": ["ISTP", "ESTJ", "ESTP"],
#     "ISTJ": ["INFJ", "ISTP", "ISFJ"],
#     "ISTP": ["ISFP", "INFP", "ESFP"],
#     "ISFJ": ["ESFJ", "ISFP", "ISTJ"],
#     "ISFP": ["ESFP", "ISFJ", "ESFJ"],
#     "ENTJ": ["INTJ", "ENTP", "ENFJ"],
#     "ENTP": ["ENTJ", "ENFP", "ENFJ"],
#     "ENFJ": ["ENFJ", "INFJ", "ENFP"],
#     "ENFP": ["ENTJ", "INTJ", "INTP"],
#     "INTJ": ["INTP", "INFJ", "INFP"],
#     "INTP": ["ENTP", "INFP", "ENFP"],
#     "INFJ": ["ISTJ", "INFP", "INTJ"],
#     "INFP": ["INFJ", "ISFJ", "ENFJ"]
# }

# # Load and cache the embedding model
# @st.cache_resource
# def load_encoder():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# @st.cache_data
# def load_data(uploaded_file):
#     df = pd.read_csv(uploaded_file)
#     df["combined_text"] = df.astype(str).agg(" ".join, axis=1)
#     model = load_encoder()
#     embeddings = model.encode(df["combined_text"].tolist(), batch_size=32, show_progress_bar=True, convert_to_numpy=True)
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(embeddings)
#     return df, index, model, embeddings

# def calculate_mbti_score(user_mbti, profile_mbti):
#     if profile_mbti in MBTI_COMPATIBILITY.get(user_mbti, []):
#         return 1  # High match
#     return 0  # No match

# def find_top_matches(query, user_mbti, df, index, model):
#     query_embedding = model.encode([query])
#     k = min(10, len(df))  # Ensure k does not exceed available data
#     distances, indices = index.search(query_embedding, k)

#     results = []
#     for i in range(k):
#         idx = indices[0][i]
#         if idx >= len(df):  # Prevent index out of range error
#             continue
#         profile = df.iloc[idx]
#         mbti_score = calculate_mbti_score(user_mbti, profile.get("MBTI Type", ""))
#         results.append((profile, distances[0][i], mbti_score))

#     # Sort by MBTI score first, then by semantic distance
#     results.sort(key=lambda x: (-x[2], x[1]))
#     return results[:5]  # Return top 5

# # Streamlit UI
# st.title("🔍 AI-Powered Profile Search & Assistant Chatbot")
# uploaded_file = st.sidebar.file_uploader("Upload LinkedIn Data", type=["csv"])

# if uploaded_file:
#     df, index, model, embeddings = load_data(uploaded_file)
#     query = st.text_input("Enter search query (e.g., 'Python Developer', 'Machine Learning Expert'):")
#     user_mbti = st.text_input("Enter your MBTI Type (e.g., INTJ, ENFP):").strip().upper()

#     if st.button("Search"):
#         if query and user_mbti:
#             top_matches = find_top_matches(query, user_mbti, df, index, model)
#             st.subheader("🎯 Top 5 Matching Profiles")
#             for profile, _, _ in top_matches:
#                 st.write(f"**👤 Name:** {profile['LinkedIn Name']}")
#                 st.write(f"**💼 Description:** {profile['Description']}")
#                 st.write(f"**📍 Location:** {profile['Location']}")
#                 st.write(f"**🔗 Profile Link:** {profile['Profile Link']}")
#                 st.write(f"**📝 About:** {profile['About']}")
#                 st.write(f"**🔮 MBTI Type:** {profile['MBTI Personality']}")
#                 st.markdown("---")





# import streamlit as st
# import pandas as pd
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# from faiss import normalize_L2

# # Custom CSS Styles
# st.markdown("""
# <style>
#     :root {
#         --primary: #2563EB;
#         --secondary: #10B981;
#         --success: #059669;
#         --danger: #DC2626;
#         --background: #F8FAFC;
#         --text: #1E293B;
#     }

#     .collab-card {
#         background: white;
#         border-radius: 12px;
#         padding: 1.5rem;
#         margin: 1rem 0;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#         border-left: 4px solid var(--primary);
#         transition: transform 0.2s;
#     }

#     .collab-card:hover {
#         transform: translateY(-3px);
#     }

#     .mbti-match {
#         border-left-color: var(--success) !important;
#         background: #F0FDF4;
#     }

#     .mbti-mismatch {
#         border-left-color: var(--danger) !important;
#         background: #FEF2F2;
#     }

#     .score-badge {
#         display: inline-block;
#         padding: 0.25rem 0.75rem;
#         border-radius: 20px;
#         font-weight: 500;
#         font-size: 0.85rem;
#     }

#     .semantic-score {
#         background: #DBEAFE;
#         color: var(--primary);
#     }

#     .mbti-score {
#         background: #D1FAE5;
#         color: var(--success);
#     }

#     .combined-score {
#         background: #E0E7FF;
#         color: #4F46E5;
#     }

#     .profile-name {
#         color: var(--text);
#         font-size: 1.2rem;
#         margin-bottom: 0.5rem;
#         font-weight: 600;
#     }

#     .mbti-pill {
#         padding: 0.25rem 0.75rem;
#         border-radius: 20px;
#         background: var(--secondary);
#         color: white;
#         font-weight: 500;
#         display: inline-block;
#     }
    
#     .section-header {
#         color: var(--text);
#         font-size: 1.5rem;
#         margin: 2rem 0 1rem;
#         padding-bottom: 0.5rem;
#         border-bottom: 2px solid var(--primary);
#     }
# </style>
# """, unsafe_allow_html=True)

# # MBTI Compatibility Rules
# IDEAL_MATCH = {
#     # ... (keep the existing IDEAL_MATCH dictionary unchanged)

#     "ESFP": ["ESFJ", "ESTP", "ISFP"],
#     "ESTP": ["ESTJ", "ESFP", "INFJ"],
#     "ESTJ": ["ESTP", "ESFJ", "ISTJ"],
#     "ESFJ": ["ISTP", "ESTJ", "ESTP"],
#     "ISTJ": ["INFJ", "ISTP", "ISFJ"],
#     "ISTP": ["ISFP", "INFP", "ESFP"],
#     "ISFJ": ["ESFJ", "ISFP", "ISTJ"],
#     "ISFP": ["ESFP", "ISFJ", "ESFJ"],
#     "ENTJ": ["INTJ", "ENTP", "ENFJ"],
#     "ENTP": ["ENTJ", "ENFP", "ENFJ"],
#     "ENFJ": ["ENFJ", "INFJ", "ENFP"],
#     "ENFP": ["ENTJ", "INTJ", "INTP"],
#     "INTJ": ["INTP", "INFJ", "INFP"],
#     "INTP": ["ENTP", "INFP", "ENFP"],
#     "INFJ": ["ISTJ", "INFP", "INTJ"],
#     "INFP": ["INFJ", "ISFJ", "ENFJ"]
# }


# def calculate_mbti_score(user_mbti, profile_mbti):
#     """Calculate MBTI compatibility score (1.0 for ideal matches, 0 otherwise)"""
#     return 1.0 if profile_mbti.upper() in IDEAL_MATCH.get(user_mbti.upper(), []) else 0.0

# @st.cache_data
# def load_data(uploaded_file):
#     try:
#         df = pd.read_csv(uploaded_file)
#         if 'MBTI Personality' not in df.columns:
#             st.error("MBTI column not found in the dataset")
#             return None, None, None, None
            
#         df["combined_text"] = df.astype(str).agg(" ".join, axis=1)
#         model = SentenceTransformer("all-MiniLM-L6-v2")
#         embeddings = model.encode(df["combined_text"].tolist(), convert_to_numpy=True)
        
#         # FAISS Index
#         embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
#         normalize_L2(embeddings)
        
#         index = faiss.IndexFlatL2(embeddings.shape[1])
#         index.add(embeddings)
#         return df, index, model, embeddings
    
#     except Exception as e:
#         st.error(f"Error loading data: {e}")
#         return None, None, None, None

# # Streamlit App
# st.set_page_config(page_title="Collaborator Finder", layout="wide")
# st.title("🤝 AI-Powered Collaborator Finder")

# # File Upload
# uploaded_file = st.sidebar.file_uploader("Upload LinkedIn Data CSV", type=["csv"])
# if uploaded_file:
#     df, index, model, embeddings = load_data(uploaded_file)
    
#     if df is not None:
#         # Semantic Search
#         query = st.text_input("Search for collaborators (e.g., 'Python Developer', 'UX Designer'):")
        
#         if st.button("Find Collaborators"):
#             if query:
#                 # Semantic Search
#                 query_embedding = model.encode([query])
#                 normalize_L2(query_embedding)
                
#                 # Get top 20 matches for re-ranking
#                 k = 20
#                 distances, indices = index.search(query_embedding, k)
                
#                 # Store results in session state
#                 st.session_state.semantic_results = [
#                     {**df.iloc[idx].to_dict(), "distance": float(dist)}
#                     for idx, dist in zip(indices[0], distances[0])
#                 ]
                
#                 # Show initial results
#                 st.markdown('<div class="section-header">🔍 Top Semantic Matches</div>', unsafe_allow_html=True)
#                 cols = st.columns(5)
#                 for i, profile in enumerate(st.session_state.semantic_results[:5]):
#                     with cols[i % 5]:
#                         st.markdown(f"""
#                         <div class="collab-card">
#                             <div class="profile-name">{profile.get('LinkedIn Name', '')}</div>
#                             <p><strong>Role:</strong> {profile.get('Current Role(s)', '')}</p>
#                             <p><strong>Skills:</strong> {profile.get('Skills', '')[:50]}...</p>
#                             <div class="mbti-pill">{profile.get('MBTI Personality', 'N/A')}</div>
#                         </div>
#                         """, unsafe_allow_html=True)

#         # MBTI Input
#         if 'semantic_results' in st.session_state:
#             st.markdown("---")
#             user_mbti = st.text_input("Enter your MBTI personality type (e.g., ENTP):").upper()
            
#             if st.button("Apply Personality Filter"):
#                 if len(user_mbti) != 4 or not user_mbti.isalpha():
#                     st.error("Please enter a valid 4-letter MBTI type")
#                 else:
#                     # Calculate combined scores
#                     ranked_profiles = []
#                     for profile in st.session_state.semantic_results:
#                         # Convert distance to similarity score (0-1 range)
#                         semantic_score = 1 / (1 + profile['distance'])
                        
#                         # Calculate MBTI score
#                         mbti_score = calculate_mbti_score(
#                             user_mbti, 
#                             profile.get('MBTI Personality', '')
#                         )
                        
#                         # Combine scores (50/50 weight)
#                         combined_score = (semantic_score * 0.5) + (mbti_score * 0.5)
                        
#                         ranked_profiles.append({
#                             **profile,
#                             "combined_score": combined_score,
#                             "mbti_score": mbti_score
#                         })
                    
#                     # Sort by combined score
#                     ranked_profiles.sort(key=lambda x: x['combined_score'], reverse=True)
                    
#                     # Display results
#                     st.markdown('<div class="section-header">🌟 Top Combined Matches (Skills + Personality)</div>', unsafe_allow_html=True)
#                     cols = st.columns(5)
#                     for i, profile in enumerate(ranked_profiles[:5]):
#                         with cols[i % 5]:
#                             card_class = "mbti-match" if profile['mbti_score'] > 0 else "mbti-mismatch"
#                             st.markdown(f"""
#                             <div class="collab-card {card_class}">
#                                 <div class="profile-name">{profile.get('LinkedIn Name', '')}</div>
#                                 <div class="score-badge combined-score">Match Score: {profile['combined_score']:.2f}</div>
#                                 <div class="mbti-pill">{profile.get('MBTI Personality', 'N/A')}</div>
#                                 <p><strong>Role:</strong> {profile.get('Current Role(s)', '')}</p>
#                                 <p><strong>Skills:</strong> {profile.get('Skills', '')[:50]}...</p>
#                                 <div class="score-badge {'mbti-score' if profile['mbti_score'] > 0 else 'semantic-score'}">
#                                     {'Ideal Personality Match ✅' if profile['mbti_score'] > 0 else 'Skills Match ⚠️'}
#                                 </div>
#                             </div>
#                             """, unsafe_allow_html=True)

# # How to Use
# else:
#     st.markdown("""
#     <div class="collab-card">
#         <h3>How to Use:</h3>
#         <ol>
#             <li>Upload a CSV file containing LinkedIn profiles with MBTI data</li>
#             <li>Search for collaborators using skills or role keywords</li>
#             <li>Enter your MBTI personality type to find personality-compatible matches</li>
#             <li>Review combined matches considering both skills and personality</li>
#         </ol>
#         <div class="mbti-pill" style="margin-top: 1rem;">Example CSV Format:</div>
#         <p>LinkedIn Name, Current Role(s), Skills, About, MBTI Personality</p>
#     </div>
#     """, unsafe_allow_html=True)








# # MBTI Personality Type
# import streamlit as st
# import pandas as pd
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer

# # Load MBTI compatibility rules (expand with your full compatibility matrix)
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


# @st.cache_resource
# def load_encoder():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# @st.cache_data
# def load_data(uploaded_file):
#     df = pd.read_csv(uploaded_file)
#     df["combined_text"] = df.astype(str).agg(" ".join, axis=1)
#     model = load_encoder()
#     embeddings = model.encode(df["combined_text"].tolist(), batch_size=32, show_progress_bar=True, convert_to_numpy=True)
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(embeddings)
#     return df, index, model, embeddings

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
#         compatibility = calculate_compatibility(user_mbti, profile.get("MBTI Personality", ""))
#         results.append((profile, distance, compatibility))
    
#     # Sort by compatibility priority then distance
#     priority = {"High Self-Match": 0, "High Compatibility": 1, "Moderate Compatibility": 2}
#     #return sorted(results, key=lambda x: (priority.get(x[2].split()[0], x[1]))[:5])
#     return sorted(results, key=lambda x: (priority.get(x[2], 3), x[1]))[:5]





# # Streamlit UI
# st.title("🔍 AI-Powered Profile Search & MBTI Compatibility")
# uploaded_file = st.sidebar.file_uploader("Upload LinkedIn Data", type=["csv"])

# if uploaded_file:
#     df, index, model, embeddings = load_data(uploaded_file)
#     query = st.text_input("Enter search query (e.g., 'Python Developer', 'Marketing Manager'):")

#     if st.button("🔍 Basic Semantic Search"):
#         if query:
#             semantic_results = semantic_search(query, df, index, model)
#             st.session_state.semantic_results = semantic_results

#     # Display semantic results if available
#     if 'semantic_results' in st.session_state:
#         st.subheader("📊 Top 5 Semantic Matches")
#         for profile, distance in st.session_state.semantic_results[:5]:
#             with st.expander(f"{profile['LinkedIn Name']} - Distance: {distance:.2f}"):
#                 st.write(f"**💼 Role:** {profile['Description']}")
#                 st.write(f"**📍 Location:** {profile['Location']}")
#                 st.write(f"**🔗 Profile:** {profile['Profile Link']}")
#                 st.write(f"**📝 About:** {profile['About'][:200]}...")
        
#         if st.button("🧠 MBTI Compatibility Search", type="primary"):
#             st.session_state.show_mbti = True

#     # MBTI Compatibility Section
#     if st.session_state.get('show_mbti', False):
#         user_mbti = st.text_input("Enter your MBTI type (e.g., INTJ):", max_chars=4).strip().upper()
        
#         if user_mbti and len(user_mbti) == 4 and user_mbti in MBTI_COMPATIBILITY:
#             combined_results = combined_search(st.session_state.semantic_results, user_mbti)
            
#             st.subheader(f"🌟 Top 5 MBTI-Compatible Matches for {user_mbti}")
#             for profile, distance, compatibility in combined_results:
#                 with st.container():
#                     st.markdown(f"### {profile['LinkedIn Name']}")
#                     col1, col2 = st.columns([1, 3])
#                     with col1:
#                         st.metric("Match Distance", f"{distance:.2f}")
#                         st.markdown(f"**MBTI Type:** {profile['MBTI Personality']}")
#                     with col2:
#                         st.markdown(f"**Compatibility:** {compatibility}")
#                         st.markdown(f"**Profile Link:** {profile['Profile Link']}")
#                         st.markdown(f"**Location:** {profile['Location']}")
#                     st.markdown("---")
#         elif user_mbti:
#             st.error("Invalid MBTI type. Please enter a valid 4-letter MBTI code.")