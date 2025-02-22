import streamlit as st
from typing import Tuple, Any
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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
        df = pd.read_csv(uploaded_file, dtype_backend="pyarrow")
        df["combined_text"] = df.astype(str).agg(" ".join, axis=1)
        model = load_encoder()
        embeddings = model.encode(
            df["combined_text"].tolist(),
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        return df, index, model, embeddings
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        raise

# Function to create and display network graph
def create_network_graph(df, indices, query):
    G = nx.Graph()
    G.add_node("Your Idea", size=20, color="red")
    for idx in indices[0]:
        profile = df.iloc[idx]
        name = profile['LinkedIn Name']
        G.add_node(name, size=15, color="blue")
        G.add_edge("Your Idea", name)
    for i in range(len(indices[0])):
        for j in range(i + 1, len(indices[0])):
            name1 = df.iloc[indices[0][i]]['LinkedIn Name']
            name2 = df.iloc[indices[0][j]]['LinkedIn Name']
            G.add_edge(name1, name2)
    pos = nx.spring_layout(G)
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        ))
    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        text=[node for node in G.nodes()],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=[G.nodes[node]['size'] for node in G.nodes()],
            color=[G.nodes[node]['color'] for node in G.nodes()],
            line_width=2
        )
    )
    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    st.plotly_chart(fig, use_container_width=True)

# RAG Function
def chat_with_rag(query, df, index, model):
    query_embedding = model.encode([query])
    k = 5
    distances, indices = index.search(query_embedding, k)
    retrieved_context = "\n".join(df.iloc[indices[0]]['combined_text'].tolist())
    prompt = f"Given the following relevant data:\n{retrieved_context}\n\nAnswer this query: {query}"
    response = chat_model.generate_content(prompt)
    return response.text

# Initialize app
st.set_page_config(
    page_title="Profile Search & Assistant Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize chat model
chat_model = init_genai()

# Streamlit UI setup
st.title("🔍 Connecting Leaders for Environmental Change")

# Sidebar for configuration
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("LinkedinData.csv", type=["csv"])

if uploaded_file:
    df, index, model, embeddings = load_data(uploaded_file)
    query = st.text_input("Enter search query (e.g., 'Tree Plantation ideas', 'Climate Change ideas'):")
    if st.button("Search"):
        if query:
            query_embedding = model.encode([query])
            k = 5
            distances, indices = index.search(query_embedding, k)
            st.subheader("🎯 Top 5 Matching Profiles")
            for idx in indices[0]:
                profile = df.iloc[idx]
                st.write(f"**👤 Name:** {profile['LinkedIn Name']}")
                st.write(f"**💼 Description:** {profile['Description']}")
                st.write(f"**🛠 About:** {profile['About']}")
                st.write(f"**📆 Current Role(s):** {profile['Current Role(s)']} years")
                st.write(f"**📍 Location:** {profile['Location']}")
                st.write(f"**📍 Profile Link:** {profile['Profile Link']}")
                st.markdown("---")
            st.subheader("🌐 Collaboration Network Graph")
            create_network_graph(df, indices, query)
    st.subheader("🤖 Assistant Chatbot")
    user_input = st.text_input("Ask Assistant about networking, career, or interests:")
    if st.button("Chat with Assistant"):
        if user_input:
            response = chat_with_rag(user_input, df, index, model)
            st.write("**Assistant:**")
            st.write(response)




