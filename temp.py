import streamlit as st
from typing import Tuple, Any
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from pathlib import Path


# Configuration
MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-1.5-flash"
API_KEY = "AIzaSyBYfzDovXQK6E8jZsrOBieoSY_X6jCUktU"  # Store this securely

# Initialize Google API
@st.cache_resource
def init_genai() -> Any:
    genai.configure(api_key=API_KEY)
    return genai.GenerativeModel(GEMINI_MODEL)

# Load and cache the embedding model
@st.cache_resource
def load_encoder() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def load_data(uploaded_file: Path) -> Tuple[pd.DataFrame, Any, SentenceTransformer, np.ndarray]:
    try:
        # Read data efficiently
        df = pd.read_csv(uploaded_file, dtype_backend="pyarrow")
        
        # Combine text more efficiently
        df["combined_text"] = df.astype(str).agg(" ".join, axis=1)
        
        # Get encoder and compute embeddings
        model = load_encoder()
        embeddings = model.encode(
            df["combined_text"].tolist(),
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Create optimized FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        faiss.normalize_L2(embeddings)  # Normalize vectors for better search
        index.add(embeddings)
        
        return df, index, model, embeddings
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        raise

# Initialize app
st.set_page_config(
    page_title="Profile Search & Assistant Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize chat model
chat_model = init_genai()

# Streamlit UI setup
st.title("🔍 AI-Powered Profile Search & Assistant Chatbot")

# Sidebar for configuration
st.sidebar.header("Upload Data")

# File upload for dataset
uploaded_file = st.sidebar.file_uploader("LinkedinData.csv", type=["csv"])

if uploaded_file:
    df, index, model, embeddings = load_data(uploaded_file)

    # Search bar
    query = st.text_input("Enter search query (e.g., 'Python Developer', 'Machine Learning Expert'):")

    if st.button("Search"):
        if query:
            query_embedding = model.encode([query])
            k = 5
            distances, indices = index.search(query_embedding, k)

            # Display results
            st.subheader("🎯 Top 5 Matching Profiles")
            for idx in indices[0]:
                profile = df.iloc[idx]
                st.write(f"**👤 Name:** {profile['LinkedIn Name']}")
                st.write(f"**💼 Description:** {profile['Description']}")
                st.write(f"**🛠 About:** {profile['About']}")
                st.write(f"**📆 Current Role(s):** {profile['Current Role(s)']} years")
                st.write(f"**📍 Location:** {profile['Location']}")
                st.write(f"**📍 Profile Link:** {profile['Profile Link']}")
                st.markdown("---")  # Separator

    # Assistant Chatbot for finding like-minded people
    st.subheader("🤖 Assistant Chatbot")
    user_input = st.text_input("Ask Assistant about networking, career, or interests:")

    if st.button("Chat with Assistant"):
        if user_input:
            response = chat_model.generate_content(user_input)
            st.write("**Assistant:**")
            st.write(response.text)
