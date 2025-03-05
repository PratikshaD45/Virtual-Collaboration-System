
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
import time
import datetime

# Configuration
MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-1.5-flash"
API_KEY = "AIzaSyBYfzDovXQK6E8jZsrOBieoSY_X6jCUktU"

# Custom CSS
def inject_custom_css():
    st.markdown("""
        <style>
            :root {
                --primary: #2A9D8F;
                --secondary: #264653;
                --accent: #E9C46A;
                --background: #F8F9FA;
                --text: #2C3E50;
                --gradient-start: #2A9D8F;
                --gradient-end: #264653;
            }
            
            .main {
                background: var(--background);
                font-family: 'Inter', sans-serif;
            }
            
            .header-text {
                font-size: 3rem !important;
                background: linear-gradient(45deg, var(--gradient-start), var(--gradient-end));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center;
                margin: 2rem 0;
                font-weight: 800;
                letter-spacing: -1px;
            }
            
            .card {
                padding: 1.5rem;
                background: white;
                border-radius: 20px;
                box-shadow: 0 4px 25px rgba(0, 0, 0, 0.06);
                margin-bottom: 1.5rem;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                border: none;
                position: relative;
                overflow: hidden;
            }
            
            .card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 4px;
                height: 100%;
                background: var(--primary);
                transition: all 0.3s;
            }
            
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 30px rgba(42, 157, 143, 0.15);
            }
            
            .card:hover::before {
                width: 6px;
                background: var(--accent);
            }
            
            .stButton>button {
                background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end)) !important;
                color: white !important;
                border-radius: 12px !important;
                padding: 0.85rem 2rem !important;
                transition: all 0.3s !important;
                border: none !important;
                font-weight: 600 !important;
                letter-spacing: 0.5px;
            }
            
            .stButton>button:hover {
                transform: scale(1.05);
                box-shadow: 0 5px 15px rgba(42, 157, 143, 0.3) !important;
            }
            
            .chat-container {
                background: white;
                border-radius: 20px;
                padding: 1.5rem;
                height: 600px;
                overflow-y: auto;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.03);
                border: 1px solid rgba(42, 157, 143, 0.1);
            }
            
            .user-message {
                background: var(--primary);
                color: white;
                padding: 1.2rem 1.8rem;
                border-radius: 20px 20px 0 20px;
                margin: 1rem 0;
                max-width: 70%;
                float: right;
                clear: both;
                position: relative;
                animation: slideIn 0.3s ease-out;
            }
            
            .bot-message {
                background: #F1FAFE;
                color: var(--text);
                padding: 1.2rem 1.8rem;
                border-radius: 20px 20px 20px 0;
                margin: 1rem 0;
                max-width: 70%;
                float: left;
                clear: both;
                animation: slideIn 0.3s ease-out;
            }
            
            @keyframes slideIn {
                from { transform: translateY(20px); opacity: 0; }
                to { transform: translateY(0); opacity: 1; }
            }
            
            .profile-img {
                width: 100px;
                height: 100px;
                border-radius: 50%;
                object-fit: cover;
                margin-right: 1.5rem;
                border: 3px solid var(--primary);
                box-shadow: 0 4px 15px rgba(42, 157, 143, 0.2);
            }
            
            .sidebar .sidebar-content {
                background: linear-gradient(180deg, var(--secondary), #1a2f38) !important;
                color: white !important;
                padding: 1.5rem !important;
                box-shadow: 4px 0 15px rgba(0, 0, 0, 0.08);
            }
            
            .loader {
                border: 4px solid #f3f3f3;
                border-radius: 50%;
                border-top: 4px solid var(--primary);
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 2rem auto;
            }
            
            .search-bar {
                padding: 1rem 1.5rem;
                border-radius: 15px;
                border: 2px solid rgba(42, 157, 143, 0.2);
                font-size: 1.1rem;
                transition: all 0.3s;
            }
            
            .search-bar:focus {
                border-color: var(--primary);
                box-shadow: 0 0 0 3px rgba(42, 157, 143, 0.1);
            }
            
            .timestamp {
                font-size: 0.75rem;
                color: #6c757d;
                margin-top: 0.5rem;
                display: block;
            }
            
            .skill-pill {
                background: rgba(42, 157, 143, 0.1) !important;
                color: var(--primary) !important;
                padding: 0.4rem 1rem !important;
                border-radius: 20px !important;
                transition: all 0.2s;
            }
            
            .skill-pill:hover {
                background: var(--primary) !important;
                color: white !important;
            }
        </style>
    """, unsafe_allow_html=True)

# Rest of the code remains the same until chat_interface function


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

def create_network_graph(df: pd.DataFrame, indices: np.ndarray, query: str):
    try:
        G = nx.Graph()
        for idx in indices[0]:
            node_name = df.iloc[idx]['LinkedIn Name']
            G.add_node(node_name)
        
        pos = nx.spring_layout(G, seed=42)
        fig = go.Figure()
        
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'))
        
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=20,
                color='#2A9D8F',
                line=dict(width=2, color='DarkSlateGrey')
            ),
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            name=''
        ))
        
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(b=0,l=0,r=0,t=0),
            height=500,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating network graph: {str(e)}")

def chat_with_rag(user_input: str, df: pd.DataFrame, index: faiss.IndexFlatL2, model: SentenceTransformer) -> str:
    try:
        query_embedding = model.encode([user_input])
        k = 3
        distances, indices = index.search(query_embedding, k)
        context = "\n".join([df.iloc[idx]['combined_text'] for idx in indices[0]])
        return f"Based on our network data: {context[:500]}..."
    except Exception as e:
        return f"Error generating response: {str(e)}"

def create_profile_card(profile):
    try:
        with st.container():
            st.markdown(f"""
                <div class="card">
                    <div style="display: flex; align-items: start; margin-bottom: 1rem;">
                        <img src="{profile.get('Profile Image', 'https://source.unsplash.com/featured/800x600/?nature')}" 
                             class="profile-img" 
                             onerror="this.src='https://source.unsplash.com/featured/800x600/?nature'">
                        <div style="flex: 1;">
                            <h3 style="margin: 0; color: var(--secondary);">{profile.get('LinkedIn Name', 'Unknown')}</h3>
                            <div style="display: flex; align-items: center; gap: 1rem; margin: 0.5rem 0;">
                                <div style="display: flex; align-items: center; color: var(--primary);">
                                    <svg style="width:18px; margin-right: 5px;" fill="currentColor" viewBox="0 0 24 24">
                                        <path d="M12 0C8.686 0 6 2.686 6 6c0 3.313 2.686 6 6 6s6-2.687 6-6c0-3.314-2.686-6-6-6zm0 2c2.21 0 4 1.79 4 4s-1.79 4-4 4-4-1.79-4-4 1.79-4 4-4zm6 14H6v-2c0-2.67 5.33-4 8-4s8 1.33 8 4v2z"/>
                                    </svg>
                                    <span>{profile.get('Location', 'Unknown')}</span>
                                </div>
                                <div style="display: flex; align-items: center; color: var(--primary);">
                                    <svg style="width:18px; margin-right: 5px;" fill="currentColor" viewBox="0 0 24 24">
                                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm.5-13h-1v6l4.25 2.53.75-1.23-3.5-2.07V7z"/>
                                    </svg>
                                    <span>{profile.get('Experience', 'N/A')}</span>
                                </div>
                            </div>
                            <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.5rem 0;">
                                {''.join([f'<span style="background: #E9F5F3; color: var(--primary); padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.9rem;">{skill}</span>' 
                                       for skill in profile.get('Skills', ['No skills listed'])[:5]])}
                            </div>
                        </div>
                    </div>
                    <div style="color: var(--text); line-height: 1.6; margin-bottom: 1rem;">
                        {profile.get('About', 'No description available')[:200]}...
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <a href="{profile.get('Profile Link', '#')}" target="_blank" style="text-decoration: none;">
                            <button style="background: var(--primary); color: white; border: none; padding: 0.6rem 1.5rem; border-radius: 8px; cursor: pointer; font-weight: 500;">
                                View Profile →
                            </button>
                        </a>
                        <div style="display: flex; gap: 1rem; color: var(--primary);">
                            <span>👥 {profile.get('Connections', 'N/A')}</span>
                            <span>🌟 {profile.get('Endorsements', 'N/A')}</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error rendering profile card: {str(e)}")


def chat_interface():
    st.markdown("### 💬 AI Collaboration Assistant")
    with st.container():
        st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.chat_history[-10:]:
            timestamp = datetime.datetime.now().strftime("%H:%M")
            st.markdown(f'''
                <div class="user-message">
                    👤 {message["user"]}
                    <span class="timestamp">{timestamp}</span>
                </div>
            ''', unsafe_allow_html=True)
            st.markdown(f'''
                <div class="bot-message">
                    🤖 {message["bot"]}
                    <span class="timestamp">{timestamp}</span>
                </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Auto-scroll to bottom
        st.markdown('''
            <script>
                window.addEventListener('load', function() {
                    const container = document.getElementById('chat-container');
                    container.scrollTop = container.scrollHeight;
                });
            </script>
        ''', unsafe_allow_html=True)
        
        with st.form("chat_form", clear_on_submit=True):
            cols = st.columns([6, 1])
            with cols[0]:
                user_input = st.text_input("Type your message...", 
                                        label_visibility="collapsed", 
                                        placeholder="Ask about collaboration opportunities...",
                                        key="chat_input")
            with cols[1]:
                if st.form_submit_button("🚀", use_container_width=True):
                    if user_input:
                        with st.spinner("Analyzing..."):
                            response = chat_with_rag(user_input, df, index, model)
                            st.session_state.chat_history.append({"user": user_input, "bot": response})
                            st.rerun()

# Modified auth screen
def auth_screen():
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.form("Login"):
            st.markdown("""
                <div style="text-align: center; margin-bottom: 2rem;">
                    <h1 style="color: var(--secondary); margin-bottom: 1rem;">🌿 Welcome Back</h1>
                    <div style="background: linear-gradient(45deg, var(--primary), var(--secondary)); 
                        width: 60px; height: 4px; margin: 0 auto 1.5rem; border-radius: 2px;"></div>
                    <p style="color: var(--text); opacity: 0.8;">Sign in to continue your sustainability journey</p>
                </div>
            """, unsafe_allow_html=True)
            
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            
            if st.form_submit_button("Login →", use_container_width=True):
                if username and password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Please enter both username and password")

    with col2:
        with st.container():
            st.markdown("""
                <div style="background: linear-gradient(135deg, var(--primary), var(--secondary)); 
                    padding: 3rem 2rem; border-radius: 20px; height: 100%; 
                    display: flex; flex-direction: column; justify-content: center;
                    box-shadow: 0 8px 25px rgba(42, 157, 143, 0.15);">
                    <div style="text-align: center; color: white;">
                        <h2 style="margin-bottom: 1rem;">🌍 New Here?</h2>
                        <p style="opacity: 0.9; margin-bottom: 2rem;">Join our network of sustainability leaders</p>
                        <div style="font-size: 3rem; margin-bottom: 1.5rem;">🚀</div>
                        <p style="opacity: 0.8; font-size: 0.9rem;">Create account to access:</p>
                        <div style="display: flex; justify-content: center; gap: 1rem; opacity: 0.9;">
                            <span>🔍 Smart Search</span>
                            <span>🤖 AI Assistant</span>
                            <span>🌐 Network Maps</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# Rest of the code remains the same with minor UI tweaks in the sidebar and profile cards

# Main app
if not st.session_state.logged_in:
    auth_screen()
else:
    with st.sidebar:
        st.markdown(f"""
            <div style="padding: 1.5rem; background: var(--secondary); color: white; border-radius: 15px; margin-bottom: 2rem;">
                <h2>👋 Welcome back, {st.session_state.username}</h2>
                <p style="opacity: 0.8;">Last login: {time.strftime('%Y-%m-%d')}</p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

    if uploaded_file:
        try:
            df, index, model, embeddings = load_data(uploaded_file)
            
            # Search Section
            st.markdown("<h1 class='header-text'>🔍 Discover Sustainability Leaders</h1>", unsafe_allow_html=True)
            
            with st.form("search_form"):
                cols = st.columns([4, 1])
                with cols[0]:
                    query = st.text_input("Search professionals by skills, interests, or expertise:",
                                        placeholder="e.g. 'Renewable Energy Policy Expert'",
                                        key="search_query")
                with cols[1]:
                    if st.form_submit_button("Search", use_container_width=True):
                        if query:
                            with st.spinner("Finding best matches..."):
                                query_embedding = model.encode([query])
                                k = 5
                                distances, indices = index.search(query_embedding, k)
                                st.session_state.top_5_people = [df.iloc[idx] for idx in indices[0]]
            
            # Results Section
            if 'top_5_people' in st.session_state and st.session_state.top_5_people:
                st.markdown("### 🌟 Top Matches")
                for profile in st.session_state.top_5_people:
                    create_profile_card(profile)
                
                # Network Visualization
                st.markdown("### 🌐 Connection Network")
                with st.spinner("Mapping professional relationships..."):
                    create_network_graph(df, indices, query)
            
            # Chat Section
            chat_interface()
        
        except Exception as e:
            st.error(f"Application error: {str(e)}")
    else:
        st.markdown("""
            <div style="text-align: center; padding: 4rem; opacity: 0.7;">
                <h2>📤 Upload Your Dataset</h2>
                <p>Begin by uploading a CSV file containing professional network data</p>
                <div style="margin-top: 2rem;">⬅️ Use the sidebar to get started</div>
            </div>
        """, unsafe_allow_html=True)


