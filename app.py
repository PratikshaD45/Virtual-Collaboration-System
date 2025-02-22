import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random
import google.generativeai as genai  # Import Google API

# Google API Key
api_key = "AIzaSyBYfzDovXQK6E8jZsrOBieoSY_X6jCUktU"  # Replace with your actual API key
genai.configure(api_key=api_key)
chat_model = genai.GenerativeModel("gemini-1.5-flash")

# Function to load and process dataset
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df["combined_text"] = df.apply(lambda row: " ".join(row.values.astype(str)), axis=1)

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["combined_text"].tolist())

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return df, index, model, embeddings

# Streamlit UI setup
st.set_page_config(page_title="Profile Search & Assistant Chatbot", layout="wide")
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
