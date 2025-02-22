import streamlit as st
from typing import Tuple, Any
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from pathlib import Path
import networkx as nx
import plotly.graph_objects as go
import hashlib
from PIL import Image

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


# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

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

# Initialize session state for user authentication and data storage
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'top_5_people' not in st.session_state:
    st.session_state.top_5_people = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'users' not in st.session_state:
    st.session_state.users = {}

# Login, Signup, and Account Management
def login():
    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in st.session_state.users and st.session_state.users[username] == hash_password(password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Logged in successfully!")
        else:
            st.error("Invalid username or password.")

def signup():
    st.header("Signup")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    if st.button("Signup"):
        if new_username in st.session_state.users:
            st.error("Username already exists.")
        else:
            st.session_state.users[new_username] = hash_password(new_password)
            st.success("Account created successfully! Please login.")

# def change_password():
#     st.header("Change Password")
#     old_password = st.text_input("Old Password", type="password")
#     new_password = st.text_input("New Password", type="password")
#     if st.button("Change Password"):
#         if st.session_state.users[st.session_state.username] == hash_password(old_password):
#             st.session_state.users[st.session_state.username] = hash_password(new_password)
#             st.success("Password changed successfully!")
#         else:
#             st.error("Incorrect old password.")

def forgot_password():
    st.header("Forgot Password")
    username = st.text_input("Enter Username")
    if st.button("Reset Password"):
        if username in st.session_state.users:
            st.session_state.users[username] = hash_password("new_password")  # Reset to a default password
            st.success("Password reset to 'new_password'. Please change it after login.")
        else:
            st.error("Username not found.")

# Display login/signup options
if not st.session_state.logged_in:
    st.image("environmental-protection-326923_640.jpg", use_column_width=True)  # Add a banner image for login/signup
    login()
    signup()
    forgot_password()
else:
    st.sidebar.header(f"Welcome, {st.session_state.username}!")
    # //change_password()
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.sidebar.success("Logged out successfully!")

# Main app functionality
if st.session_state.logged_in:
    # Sidebar for configuration
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("LinkedinData.csv", type=["csv"])

    if uploaded_file:
        df, index, model, embeddings = load_data(uploaded_file)
        
        # Search for top 5 people
        query = st.text_input("Enter search query (e.g., 'Tree Plantation ideas', 'Climate Change ideas'):")
        if st.button("Search"):
            if query:
                query_embedding = model.encode([query])
                k = 5
                distances, indices = index.search(query_embedding, k)
                st.session_state.top_5_people = []
                for idx in indices[0]:
                    profile = df.iloc[idx]
                    st.session_state.top_5_people.append(profile)
                st.subheader("🎯 Top 5 Matching Profiles")
                for profile in st.session_state.top_5_people:
                    st.write(f"**👤 Name:** {profile['LinkedIn Name']}")
                    st.write(f"**💼 Description:** {profile['Description']}")
                    st.write(f"**🛠 About:** {profile['About']}")
                    st.write(f"**📆 Current Role(s):** {profile['Current Role(s)']} years")
                    st.write(f"**📍 Location:** {profile['Location']}")
                    st.write(f"**📍 Profile Link:** {profile['Profile Link']}")
                    st.markdown("---")
                st.subheader("🌐 Collaboration Network Graph")
                create_network_graph(df, indices, query)
    
        
        # Chatbot with chat history
        st.subheader("🤖 Assistant Chatbot")
        user_input = st.text_input("Ask Assistant about networking, career, or interests:")
        if st.button("Chat with Assistant"):
            if user_input:
                response = chat_with_rag(user_input, df, index, model)
                st.session_state.chat_history.append(f"**You:** {user_input}")
                st.session_state.chat_history.append(f"**Assistant:** {response}")
                st.write("**Chat History:**")
                for chat in st.session_state.chat_history:
                    st.write(chat)