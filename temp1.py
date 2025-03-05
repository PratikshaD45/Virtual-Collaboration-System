# import streamlit as st
# from typing import Tuple, Any
# import pandas as pd
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# import google.generativeai as genai
# from pathlib import Path
# import networkx as nx
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go

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
#         df = pd.read_csv(uploaded_file, dtype_backend="pyarrow")
#         df["combined_text"] = df.astype(str).agg(" ".join, axis=1)
#         model = load_encoder()
#         embeddings = model.encode(
#             df["combined_text"].tolist(),
#             batch_size=32,
#             show_progress_bar=True,
#             convert_to_numpy=True
#         )
#         dimension = embeddings.shape[1]
#         index = faiss.IndexFlatL2(dimension)
#         faiss.normalize_L2(embeddings)
#         index.add(embeddings)
#         return df, index, model, embeddings
#     except Exception as e:
#         st.error(f"Error loading data: {str(e)}")
#         raise

# # Function to create and display network graph
# def create_network_graph(df, indices, query):
#     G = nx.Graph()
#     G.add_node("Your Idea", size=20, color="red")
#     for idx in indices[0]:
#         profile = df.iloc[idx]
#         name = profile['LinkedIn Name']
#         G.add_node(name, size=15, color="blue")
#         G.add_edge("Your Idea", name)
#     for i in range(len(indices[0])):
#         for j in range(i + 1, len(indices[0])):
#             name1 = df.iloc[indices[0][i]]['LinkedIn Name']
#             name2 = df.iloc[indices[0][j]]['LinkedIn Name']
#             G.add_edge(name1, name2)
#     pos = nx.spring_layout(G)
#     edge_trace = []
#     for edge in G.edges():
#         x0, y0 = pos[edge[0]]
#         x1, y1 = pos[edge[1]]
#         edge_trace.append(go.Scatter(
#             x=[x0, x1, None], y=[y0, y1, None],
#             line=dict(width=0.5, color='#888'),
#             hoverinfo='none',
#             mode='lines'
#         ))
#     node_trace = go.Scatter(
#         x=[pos[node][0] for node in G.nodes()],
#         y=[pos[node][1] for node in G.nodes()],
#         text=[node for node in G.nodes()],
#         mode='markers+text',
#         hoverinfo='text',
#         marker=dict(
#             showscale=True,
#             colorscale='YlGnBu',
#             size=[G.nodes[node]['size'] for node in G.nodes()],
#             color=[G.nodes[node]['color'] for node in G.nodes()],
#             line_width=2
#         )
#     )
#     fig = go.Figure(data=edge_trace + [node_trace],
#                     layout=go.Layout(
#                         showlegend=False,
#                         hovermode='closest',
#                         margin=dict(b=0, l=0, r=0, t=0),
#                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
#                     ))
#     st.plotly_chart(fig, use_container_width=True)

# # RAG Function
# def chat_with_rag(query, df, index, model):
#     query_embedding = model.encode([query])
#     k = 5
#     distances, indices = index.search(query_embedding, k)
#     retrieved_context = "\n".join(df.iloc[indices[0]]['combined_text'].tolist())
#     prompt = f"Given the following relevant data:\n{retrieved_context}\n\nAnswer this query: {query}"
#     response = chat_model.generate_content(prompt)
#     return response.text

# # Initialize app
# st.set_page_config(
#     page_title="Profile Search & Assistant Chatbot",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Initialize chat model
# chat_model = init_genai()

# # Streamlit UI setup
# st.title("🔍 Connecting Leaders for Environmental Change")

# # Sidebar for configuration
# st.sidebar.header("Upload Data")
# uploaded_file = st.sidebar.file_uploader("LinkedinData.csv", type=["csv"])

# if uploaded_file:
#     df, index, model, embeddings = load_data(uploaded_file)
#     query = st.text_input("Enter search query (e.g., 'Tree Plantation ideas', 'Climate Change ideas'):")
#     if st.button("Search"):
#         if query:
#             query_embedding = model.encode([query])
#             k = 5
#             distances, indices = index.search(query_embedding, k)
#             st.subheader("🎯 Top 5 Matching Profiles")
#             for idx in indices[0]:
#                 profile = df.iloc[idx]
#                 st.write(f"**👤 Name:** {profile['LinkedIn Name']}")
#                 st.write(f"**💼 Description:** {profile['Description']}")
#                 st.write(f"**🛠 About:** {profile['About']}")
#                 st.write(f"**📆 Current Role(s):** {profile['Current Role(s)']} years")
#                 st.write(f"**📍 Location:** {profile['Location']}")
#                 st.write(f"**📍 Profile Link:** {profile['Profile Link']}")
#                 st.markdown("---")
#             st.subheader("🌐 Collaboration Network Graph")
#             create_network_graph(df, indices, query)
#     st.subheader("🤖 Assistant Chatbot")
#     user_input = st.text_input("Ask Assistant about networking, career, or interests:")
#     if st.button("Chat with Assistant"):
#         if user_input:
#             response = chat_with_rag(user_input, df, index, model)
#             st.write("**Assistant:**")
#             st.write(response)










# import streamlit as st
# from typing import Tuple, Any
# import pandas as pd
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# import google.generativeai as genai
# from pathlib import Path
# import networkx as nx
# import plotly.graph_objects as go
# import hashlib
# from PIL import Image
# import time
# from typing import Dict, List

# # Configuration
# MODEL_NAME = "all-MiniLM-L6-v2"
# GEMINI_MODEL = "gemini-1.5-flash"
# API_KEY = "AIzaSyBYfzDovXQK6E8jZsrOBieoSY_X6jCUktU"

# # Custom CSS
# def inject_custom_css():
#     st.markdown("""
#         <style>
#             :root {
#                 --primary: #2A9D8F;
#                 --secondary: #264653;
#                 --accent: #E9C46A;
#                 --background: #F8F9FA;
#                 --text: #2C3E50;
#             }
            
#             body {
#                 background: var(--background);
#             }

#             .main {
#                 background: var(--background);
#                 font-family: 'Segoe UI', sans-serif;
#             }
            
#             .header-text {
#                 font-size: 2.8rem !important;
#                 color: var(--secondary) !important;
#                 text-align: center;
#                 margin-bottom: 2rem !important;
#                 font-weight: 700;
#                 letter-spacing: -0.5px;
#             }
            
#             .card {
#                 padding: 1.5rem;
#                 background: white;
#                 border-radius: 15px;
#                 box-shadow: 0 2px 15px rgba(0, 0, 0, 0.08);
#                 margin-bottom: 1.5rem;
#                 transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
#                 border: 1px solid rgba(0, 0, 0, 0.05);
#             }
            
#             .card:hover {
#                 transform: translateY(-5px);
#                 box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
#             }
            
#             .stButton>button {
#                 background: var(--primary) !important;
#                 color: white !important;
#                 border-radius: 10px !important;
#                 padding: 0.75rem 1.5rem !important;
#                 transition: all 0.3s !important;
#                 border: none !important;
#                 font-weight: 600 !important;
#             }
            
#             .stButton>button:hover {
#                 background: #228176 !important;
#                 transform: scale(1.05);
#                 box-shadow: 0 3px 8px rgba(0, 0, 0, 0.15);
#             }
            
#             .chat-container {
#                 background: white;
#                 border-radius: 15px;
#                 padding: 1.5rem;
#                 height: 500px;
#                 overflow-y: auto;
#                 box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
#                 border: 1px solid rgba(0, 0, 0, 0.08);
#             }
            
#             .user-message {
#                 background: var(--primary);
#                 color: white;
#                 padding: 1rem 1.5rem;
#                 border-radius: 15px 15px 0 15px;
#                 margin: 0.8rem 0;
#                 max-width: 75%;
#                 float: right;
#                 clear: both;
#                 box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
#             }
            
#             .bot-message {
#                 background: #F1FAFE;
#                 color: var(--text);
#                 padding: 1rem 1.5rem;
#                 border-radius: 15px 15px 15px 0;
#                 margin: 0.8rem 0;
#                 max-width: 75%;
#                 float: left;
#                 clear: both;
#                 box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
#             }
            
#             .profile-img {
#                 width: 80px;
#                 height: 80px;
#                 border-radius: 50%;
#                 object-fit: cover;
#                 margin-right: 1.5rem;
#                 border: 3px solid var(--primary);
#             }
            
#             .sidebar .sidebar-content {
#                 background: var(--secondary) !important;
#                 color: white !important;
#                 padding: 1rem !important;
#             }

#             /* Additional CSS for dynamic UI enhancements */
#             .stTextInput>div>div>input {
#                 border-radius: 10px;
#                 border: 1px solid #ccc;
#                 padding: 0.8rem;
#             }
            
#             .stFileUploader label {
#                 font-weight: 600;
#                 color: var(--secondary);
#             }
            
#             .loader {
#                 border: 4px solid #f3f3f3;
#                 border-radius: 50%;
#                 border-top: 4px solid var(--primary);
#                 width: 40px;
#                 height: 40px;
#                 animation: spin 1s linear infinite;
#                 margin: 2rem auto;
#             }
            
#             @keyframes spin {
#                 0% { transform: rotate(0deg); }
#                 100% { transform: rotate(360deg); }
#             }
#         </style>
#     """, unsafe_allow_html=True)

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
#         df = pd.read_csv(uploaded_file, dtype_backend="pyarrow")
#         df["combined_text"] = df.astype(str).agg(" ".join, axis=1)
#         model = load_encoder()
#         embeddings = model.encode(
#             df["combined_text"].tolist(),
#             batch_size=32,
#             show_progress_bar=True,
#             convert_to_numpy=True
#         )
#         dimension = embeddings.shape[1]
#         index = faiss.IndexFlatL2(dimension)
#         faiss.normalize_L2(embeddings)
#         index.add(embeddings)
#         return df, index, model, embeddings
#     except Exception as e:
#         st.error(f"Error loading data: {str(e)}")
#         raise

# def create_network_graph(df: pd.DataFrame, indices: np.ndarray, query: str):
#     try:
#         G = nx.Graph()
#         for idx in indices[0]:
#             node_name = df.iloc[idx].get('LinkedIn Name', f'Profile {idx}')
#             G.add_node(node_name)
        
#         pos = nx.spring_layout(G, seed=42)
#         fig = go.Figure()
        
#         edge_x = []
#         edge_y = []
#         for edge in G.edges():
#             x0, y0 = pos[edge[0]]
#             x1, y1 = pos[edge[1]]
#             edge_x.extend([x0, x1, None])
#             edge_y.extend([y0, y1, None])
        
#         fig.add_trace(go.Scatter(
#             x=edge_x, y=edge_y,
#             line=dict(width=0.5, color='#888'),
#             hoverinfo='none',
#             mode='lines'))
        
#         node_x = []
#         node_y = []
#         node_text = []
#         for node in G.nodes():
#             x, y = pos[node]
#             node_x.append(x)
#             node_y.append(y)
#             node_text.append(node)
        
#         fig.add_trace(go.Scatter(
#             x=node_x, y=node_y,
#             mode='markers+text',
#             marker=dict(
#                 size=20,
#                 color='#2A9D8F',
#                 line=dict(width=2, color='DarkSlateGrey')
#             ),
#             text=node_text,
#             textposition="top center",
#             hoverinfo='text',
#             name=''
#         ))
        
#         fig.update_layout(
#             showlegend=False,
#             plot_bgcolor='rgba(0,0,0,0)',
#             margin=dict(b=0,l=0,r=0,t=0),
#             height=500,
#             xaxis=dict(showgrid=False, zeroline=False),
#             yaxis=dict(showgrid=False, zeroline=False)
#         )
#         st.plotly_chart(fig, use_container_width=True)
#     except Exception as e:
#         st.error(f"Error generating network graph: {str(e)}")

# def chat_with_rag(user_input: str, df: pd.DataFrame, index: faiss.IndexFlatL2, model: SentenceTransformer) -> str:
#     try:
#         query_embedding = model.encode([user_input])
#         k = 3
#         distances, indices = index.search(query_embedding, k)
#         context = "\n".join([df.iloc[idx]['combined_text'] for idx in indices[0]])
#         return f"Based on our network data: {context[:500]}..."
#     except Exception as e:
#         return f"Error generating response: {str(e)}"

# # ================== MODIFIED PROFILE CARD WITH MBTI ==================
# def create_profile_card(profile):
#     try:
#         user_mbti = st.session_state.get('user_mbti', 'ENFJ')  # Default if not set
#         profile_mbti = profile.get('MBTI', '').upper()[:4]
        
#         with st.container():
#             compatibility = calculate_mbti_score(user_mbti, profile_mbti) if profile_mbti else 0
#             score_color = "#2A9D8F" if compatibility >= 0.8 else "#E9C46A" if compatibility >= 0.6 else "#264653"
            
#             st.markdown(f"""
#                 <div class="card">
#                     <div style="display: flex; align-items: start; margin-bottom: 1rem; position: relative;">
#                         <div style="position: absolute; right: 0; top: 0; background: {score_color}; 
#                                     color: white; padding: 0.3rem 1rem; border-radius: 0 15px 0 15px;
#                                     font-weight: 600;">
#                             {compatibility*100:.0f}% Match
#                         </div>
#                         <img src="{profile.get('Profile Image', 'https://source.unsplash.com/featured/800x600/?nature')}" 
#                              class="profile-img" 
#                              onerror="this.src='https://source.unsplash.com/featured/800x600/?nature'">
#                         <div style="flex: 1;">
#                             <h3 style="margin: 0; color: var(--secondary);">{profile.get('LinkedIn Name', 'Unknown')}</h3>
#                             <div style="display: flex; align-items: center; gap: 1rem; margin: 0.5rem 0;">
#                                 {f'<div style="background: #E9F5F3; padding: 0.3rem 1rem; border-radius: 20px; font-size: 0.9rem;">MBTI: {profile_mbti}</div>' if profile_mbti else ''}
#                                 <!-- Rest of your existing profile card content -->
#                             </div>
#                             <!-- Keep existing profile card structure -->
#                         </div>
#                     </div>
#                 </div>
#             """, unsafe_allow_html=True)
#     except Exception as e:
#         st.error(f"Error rendering profile card: {str(e)}")



# # ================== MBTI COMPATIBILITY SYSTEM ==================
# MBTI_COMPATIBILITY: Dict[str, List[str]] = {
#     'ESFP': ['ESFJ', 'ESTP', 'ISFP'],
#     'ESTP': ['ESTJ', 'ESFP', 'INFJ'],
#     'ESTJ': ['ESTP', 'ESFJ', 'ISTJ'],
#     'ESFJ': ['ISTP', 'ESTJ', 'ESTP'],
#     'ISTJ': ['INFJ', 'ISTP', 'ISFJ'],
#     'ISTP': ['ISFP', 'INFP', 'ESFP'],
#     'ISFJ': ['ESFJ', 'ISFP', 'ISTJ'],
#     'ISFP': ['ESFP', 'ISFJ', 'ESFJ'],
#     'ENTJ': ['INTJ', 'ENTP', 'ENFJ'],
#     'ENTP': ['ENTJ', 'ENFP', 'ENFJ'],
#     'ENFJ': ['ENFJ', 'INFJ', 'ENFP'],
#     'ENFP': ['ENTJ', 'INTJ', 'INTP'],
#     'INTJ': ['INTP', 'INFJ', 'INFP'],
#     'INTP': ['ENTP', 'INFP', 'ENFP'],
#     'INFJ': ['ISTJ', 'INFP', 'INTJ'],
#     'INFP': ['INFJ', 'ISFJ', 'ENFJ']
# }

# def calculate_mbti_score(user_mbti: str, profile_mbti: str) -> float:
#     """Calculate compatibility score between two MBTI types"""
#     if user_mbti == profile_mbti:
#         return 1.0  # 100% match
#     if profile_mbti in MBTI_COMPATIBILITY.get(user_mbti, []):
#         return 0.8  # 80% match
#     return 0.6  # 60% base score

# # ================== ENHANCED UI COMPONENTS ==================
# def app_header():
#     """Custom header component"""
#     st.markdown("""
#         <div style="background: linear-gradient(135deg, #2A9D8F, #264653); 
#                     padding: 2rem; 
#                     border-radius: 0 0 20px 20px;
#                     margin-bottom: 2rem;
#                     box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
#             <div style="max-width: 1200px; margin: 0 auto;">
#                 <h1 style="color: white; margin: 0; font-size: 2.5rem;">🌐 EcoConnect Network</h1>
#                 <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0;">Connecting Sustainability Leaders Worldwide</p>
#             </div>
#         </div>
#     """, unsafe_allow_html=True)

# def app_footer():
#         """Custom footer component"""
#         st.markdown("""
#         <div style="background: #264653; 
#                     color: white;
#                     padding: 2rem;
#                     margin-top: 4rem;
#                     border-radius: 20px 20px 0 0;">
#             <div style="max-width: 1200px; margin: 0 auto; text-align: center;">
#                 <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 1rem;">
#                     <a href="#" style="color: white; text-decoration: none;">About</a>
#                     <a href="#" style="color: white; text-decoration: none;">Contact</a>
#                     <a href="#" style="color: white; text-decoration: none;">Privacy</a>
#                 </div>
#                 <hr style="opacity: 0.2;">
#                 <p style="opacity: 0.8; margin-top: 1rem;">© 2024 EcoConnect. Empowering sustainable collaboration.</p>
#             </div>
#         </div>
#         """, unsafe_allow_html=True)


# def chat_interface(df, index, model):
#     st.markdown("### 💬 Collaboration Assistant")
#     with st.container():
#         st.markdown('<div class="chat-container">', unsafe_allow_html=True)
#         for message in st.session_state.chat_history[-10:]:
#             st.markdown(f'<div class="user-message">👤 {message["user"]}</div>', unsafe_allow_html=True)
#             st.markdown(f'<div class="bot-message">🤖 {message["bot"]}</div>', unsafe_allow_html=True)
#         st.markdown('</div>', unsafe_allow_html=True)
        
#         with st.form("chat_form", clear_on_submit=True):
#             cols = st.columns([6, 1])
#             with cols[0]:
#                 user_input = st.text_input("Type your message...", 
#                                            label_visibility="collapsed", 
#                                            placeholder="Ask about collaboration opportunities...",
#                                            key="chat_input")
#             with cols[1]:
#                 if st.form_submit_button("Send", use_container_width=True):
#                     if user_input:
#                         with st.spinner("Analyzing..."):
#                             response = chat_with_rag(user_input, df, index, model)
#                             st.session_state.chat_history.append({"user": user_input, "bot": response})
#                             st.experimental_rerun()

# # Initialize app
# st.set_page_config(
#     page_title="EcoConnect - Leadership Network",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     page_icon="🌍"
# )

# # Inject custom CSS
# inject_custom_css()

# # Initialize session state
# if 'logged_in' not in st.session_state:
#     st.session_state.logged_in = False
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# if 'username' not in st.session_state:
#     st.session_state.username = "Guest"

# # Auth functions
# def auth_screen():
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         with st.form("Login"):
#             st.markdown("""
#                 <div style="text-align: center; margin-bottom: 2rem;">
#                     <h1 style="color: var(--secondary);">🌿 Welcome Back</h1>
#                     <p style="color: var(--text);">Sign in to continue</p>
#                 </div>
#             """, unsafe_allow_html=True)
            
#             username = st.text_input("Username", key="login_user")
#             password = st.text_input("Password", type="password", key="login_pass")
            
#             if st.form_submit_button("Login →", use_container_width=True):
#                 if username and password:
#                     st.session_state.logged_in = True
#                     st.session_state.username = username
#                     st.experimental_rerun()
#                 else:
#                     st.error("Please enter both username and password")

#     with col2:
#         with st.form("Signup"):
#             st.markdown("""
#                 <div style="text-align: center; margin-bottom: 2rem;">
#                     <h1 style="color: white;">🌱 Join Our Network</h1>
#                     <p style="color: rgba(255,255,255,0.8);">Connect with sustainability leaders worldwide</p>
#                 </div>
#             """, unsafe_allow_html=True)
            
#             new_user = st.text_input("Choose Username", key="signup_user")
#             new_pass = st.text_input("Create Password", type="password", key="signup_pass")
            
#             if st.form_submit_button("Get Started →", use_container_width=True):
#                 if new_user and new_pass:
#                     st.success("Account created! Please login")
#                 else:
#                     st.error("Please fill all fields")

# # Main app
# if not st.session_state.logged_in:
#     auth_screen()
# else:
#     app_header()
#     with st.sidebar:
#         st.markdown(f"""
#             <div style="padding: 1.5rem; background: var(--secondary); color: white; border-radius: 15px; margin-bottom: 2rem;">
#                 <h2>👋 Welcome back, {st.session_state.username}</h2>
#                 <p style="opacity: 0.8;">Last login: {time.strftime('%Y-%m-%d')}</p>
#             </div>
#         """, unsafe_allow_html=True)
        
#         uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])
#         st.markdown("---")
        
#         if st.button("🚪 Logout", use_container_width=True):
#             st.session_state.logged_in = False
#             st.experimental_rerun()

#     if uploaded_file:
#         try:
#             df, index, model, embeddings = load_data(uploaded_file)
            
#             # Search Section
#             st.markdown("<h1 class='header-text'>🔍 Discover Sustainability Leaders</h1>", unsafe_allow_html=True)
            
#             with st.form("search_form"):
#                 cols = st.columns([4, 1])
#                 with cols[0]:
#                     query = st.text_input("Search professionals by skills, interests, or expertise:",
#                                             placeholder="e.g. 'Renewable Energy Policy Expert'",
#                                             key="search_query")
#                 with cols[1]:
#                     if st.form_submit_button("Search", use_container_width=True):
#                         if query:
#                             with st.spinner("Finding best matches..."):
#                                 query_embedding = model.encode([query])
#                                 k = 5
#                                 distances, indices = index.search(query_embedding, k)
#                                 st.session_state.top_5_people = [df.iloc[idx] for idx in indices[0]]
            
#             # Results Section
#             if 'top_5_people' in st.session_state and st.session_state.top_5_people:
#                 st.markdown("### 🌟 Top Matches")
#                 for profile in st.session_state.top_5_people:
#                     create_profile_card(profile)
                
#                 # Network Visualization
#                 st.markdown("### 🌐 Connection Network")
#                 with st.spinner("Mapping professional relationships..."):
#                     create_network_graph(df, indices, query)
           
#             # Chat Section
#             chat_interface(df, index, model)
#         except Exception as e:
#             st.error(f"Application error: {str(e)}")
#     else:
#         st.markdown("""
#             <div style="text-align: center; padding: 4rem; opacity: 0.7;">
#                 <h2>📤 Upload Your Dataset</h2>
#                 <p>Begin by uploading a CSV file containing professional network data</p>
#                 <div style="margin-top: 2rem;">⬅️ Use the sidebar to get started</div>
#             </div>
#         """, unsafe_allow_html=True)

#         app_footer()
        





# import streamlit as st
# from typing import Tuple, Any, Dict, List
# import pandas as pd
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# import google.generativeai as genai
# from pathlib import Path
# import networkx as nx
# import plotly.graph_objects as go
# import hashlib
# from PIL import Image
# import time
# import datetime

# # Configuration
# MODEL_NAME = "all-MiniLM-L6-v2"
# GEMINI_MODEL = "gemini-1.5-flash"
# API_KEY = "AIzaSyBYfzDovXQK6E8jZsrOBieoSY_X6jCUktU"
# MBTI_COMPATIBILITY = {
#     'ESFP': ['ESFJ', 'ESTP', 'ISFP'],
#     'ESTP': ['ESTJ', 'ESFP', 'INFJ'],
#     'ESTJ': ['ESTP', 'ESFJ', 'ISTJ'],
#     'ESFJ': ['ISTP', 'ESTJ', 'ESTP'],
#     'ISTJ': ['INFJ', 'ISTP', 'ISFJ'],
#     'ISTP': ['ISFP', 'INFP', 'ESFP'],
#     'ISFJ': ['ESFJ', 'ISFP', 'ISTJ'],
#     'ISFP': ['ESFP', 'ISFJ', 'ESFJ'],
#     'ENTJ': ['INTJ', 'ENTP', 'ENFJ'],
#     'ENTP': ['ENTJ', 'ENFP', 'ENFJ'],
#     'ENFJ': ['ENFJ', 'INFJ', 'ENFP'],
#     'ENFP': ['ENTJ', 'INTJ', 'INTP'],
#     'INTJ': ['INTP', 'INFJ', 'INFP'],
#     'INTP': ['ENTP', 'INFP', 'ENFP'],
#     'INFJ': ['ISTJ', 'INFP', 'INTJ'],
#     'INFP': ['INFJ', 'ISFJ', 'ENFJ']
# }

# # Custom CSS
# def inject_custom_css():
#     st.markdown("""
#         <style>
#             :root {
#                 --primary: #2A9D8F;
#                 --secondary: #264653;
#                 --accent: #E9C46A;
#                 --background: #F8F9FA;
#                 --text: #2C3E50;
#             }
            
#             .main {
#                 background: var(--background);
#                 font-family: 'Inter', sans-serif;
#             }
            
#             .header {
#                 background: linear-gradient(135deg, var(--primary), var(--secondary));
#                 padding: 2rem;
#                 color: white;
#                 border-radius: 0 0 20px 20px;
#                 margin-bottom: 2rem;
#                 box-shadow: 0 4px 15px rgba(0,0,0,0.1);
#             }
            
#             .card {
#                 padding: 1.5rem;
#                 background: white;
#                 border-radius: 15px;
#                 box-shadow: 0 4px 25px rgba(0, 0, 0, 0.06);
#                 margin-bottom: 1.5rem;
#                 transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
#             }
            
#             .profile-img {
#                 width: 100px;
#                 height: 100px;
#                 border-radius: 50%;
#                 object-fit: cover;
#                 border: 3px solid var(--primary);
#                 box-shadow: 0 4px 15px rgba(42, 157, 143, 0.2);
#             }
            
#             .mbti-badge {
#                 background: #E9F5F3;
#                 color: var(--primary);
#                 padding: 0.3rem 1rem;
#                 border-radius: 20px;
#                 font-weight: 600;
#             }
            
#             .compatibility-score {
#                 position: absolute;
#                 right: 0;
#                 top: 0;
#                 padding: 0.3rem 1rem;
#                 border-radius: 0 15px 0 15px;
#                 font-weight: 600;
#                 color: white;
#             }
            
#             footer {
#                 background: var(--secondary);
#                 color: white;
#                 padding: 2rem;
#                 margin-top: 4rem;
#                 border-radius: 20px 20px 0 0;
#             }
#         </style>
#     """, unsafe_allow_html=True)

# # Initialize models
# @st.cache_resource
# def init_genai():
#     genai.configure(api_key=API_KEY)
#     return genai.GenerativeModel(GEMINI_MODEL)

# @st.cache_resource
# def load_encoder():
#     return SentenceTransformer(MODEL_NAME)

# @st.cache_data
# def load_data(uploaded_file):
#     df = pd.read_csv(uploaded_file)
#     df["combined_text"] = df.astype(str).agg(" ".join, axis=1)
#     model = load_encoder()
#     embeddings = model.encode(df["combined_text"].tolist(), convert_to_numpy=True)
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     faiss.normalize_L2(embeddings)
#     index.add(embeddings)
#     return df, index, model, embeddings

# def calculate_mbti_score(user_mbti, profile_mbti):
#     profile_mbti = profile_mbti.upper()[:4]
#     if user_mbti == profile_mbti: return 1.0
#     if profile_mbti in MBTI_COMPATIBILITY.get(user_mbti, []): return 0.8
#     return 0.6

# def create_profile_card(profile):
#     user_mbti = st.session_state.get('user_mbti', 'ENFJ')
#     profile_mbti = profile.get('MBTI', '')
#     score = calculate_mbti_score(user_mbti, profile_mbti)
#     score_color = "#2A9D8F" if score >= 0.8 else "#E9C46A" if score >= 0.6 else "#264653"
    
#     st.markdown(f"""
#         <div class="card">
#             <div style="position:relative;">
#                 <div class="compatibility-score" style="background:{score_color}">
#                     {score*100:.0f}% Match
#                 </div>
#                 <div style="display:flex; gap:1.5rem; align-items:start;">
#                     <img src="{profile.get('Profile Image', 'https://via.placeholder.com/100')}" 
#                          class="profile-img"
#                          onerror="this.src='https://via.placeholder.com/100'">
#                     <div style="flex:1;">
#                         <h3>{profile.get('LinkedIn Name', 'Unknown')}</h3>
#                         <div class="mbti-badge">MBTI: {profile_mbti.upper()[:4]}</div>
#                         <p>{profile.get('About', '')[:200]}...</p>
#                         <div style="display:flex; justify-content:space-between; align-items:center;">
#                             <a href="{profile.get('Profile Link', '#')}" target="_blank">
#                                 <button style="background:var(--primary); color:white; border:none; padding:0.5rem 1rem; border-radius:8px;">
#                                     View Profile →
#                                 </button>
#                             </a>
#                             <div style="display:flex; gap:1rem;">
#                                 <span>📍 {profile.get('Location', '')}</span>
#                                 <span>🌟 {profile.get('Endorsements', 0)}</span>
#                             </div>
#                         </div>
#                     </div>
#                 </div>
#             </div>
#         </div>
#     """, unsafe_allow_html=True)

# def app_header():
#     st.markdown("""
#         <div class="header">
#             <h1>🌐 EcoConnect Network</h1>
#             <p>Connecting Sustainability Leaders Worldwide</p>
#         </div>
#     """, unsafe_allow_html=True)

# def app_footer():
#     st.markdown("""
#         <footer>
#             <div style="text-align:center;">
#                 <div style="display:flex; justify-content:center; gap:2rem; margin-bottom:1rem;">
#                     <a href="#" style="color:white;">About</a>
#                     <a href="#" style="color:white;">Contact</a>
#                     <a href="#" style="color:white;">Privacy</a>
#                 </div>
#                 <hr style="opacity:0.2;">
#                 <p>© 2024 EcoConnect. Empowering sustainable collaboration.</p>
#             </div>
#         </footer>
#     """, unsafe_allow_html=True)

# # Main App
# st.set_page_config(page_title="EcoConnect", layout="wide", page_icon="🌍")
# inject_custom_css()

# if 'logged_in' not in st.session_state:
#     st.session_state.update({
#         'logged_in': False,
#         'chat_history': [],
#         'username': '',
#         'user_mbti': 'ENFJ'
#     })

# if not st.session_state.logged_in:
#     col1, col2 = st.columns(2)
#     with col1:
#         with st.form("Login"):
#             st.header("Welcome Back 🌿")
#             username = st.text_input("Username")
#             password = st.text_input("Password", type="password")
#             mbti = st.selectbox("MBTI Personality", options=list(MBTI_COMPATIBILITY.keys()))
#             if st.form_submit_button("Login"):
#                 if username and password:
#                     st.session_state.update({
#                         'logged_in': True,
#                         'username': username,
#                         'user_mbti': mbti
#                     })
#                     st.rerun()
#     with col2:
#         st.markdown("""
#             <div style="background:linear-gradient(135deg,#2A9D8F,#264653); padding:2rem; border-radius:15px; color:white;">
#                 <h2>New Here? 🌱</h2>
#                 <p>Join our network of sustainability leaders</p>
#                 <div style="font-size:3rem;">🚀</div>
#                 <p>Access smart search, AI assistant, and network mapping</p>
#             </div>
#         """, unsafe_allow_html=True)
# else:
#     app_header()
#     with st.sidebar:
#         st.markdown(f"""
#             <div style="background:var(--secondary); color:white; padding:1rem; border-radius:15px;">
#                 <h3>👋 {st.session_state.username}</h3>
#                 <p>MBTI: {st.session_state.user_mbti}</p>
#                 <p>Last login: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
#             </div>
#         """, unsafe_allow_html=True)
#         uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"])
#         if st.button("🚪 Logout"):
#             st.session_state.logged_in = False
#             st.rerun()

#     if uploaded_file:
#         df, index, model, embeddings = load_data(uploaded_file)
        
#         # Search Section
#         st.header("🔍 Discover Sustainability Leaders")
#         query = st.text_input("Search skills, interests, or expertise")
#         if query:
#             query_embed = model.encode([query])
#             distances, indices = index.search(query_embed, 5)
#             st.session_state.results = [df.iloc[i] for i in indices[0]]
        
#         if 'results' in st.session_state:
#             for profile in st.session_state.results:
#                 create_profile_card(profile)
            
#             # Network Visualization
#             st.header("🌐 Connection Network")
#             G = nx.Graph()
#             for profile in st.session_state.results:
#                 G.add_node(profile['LinkedIn Name'])
#             pos = nx.spring_layout(G)
#             fig = go.Figure()
#             fig.add_trace(go.Scatter(
#                 x=[pos[node][0] for node in G.nodes],
#                 y=[pos[node][1] for node in G.nodes],
#                 mode='markers+text',
#                 text=list(G.nodes),
#                 marker=dict(size=20, color='#2A9D8F')
#             ))
#             st.plotly_chart(fig, use_container_width=True)

#     app_footer()







import streamlit as st
from typing import Tuple, Any, Dict, List
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
            }
            
            body {
                background: var(--background);
            }

            .main {
                background: var(--background);
                font-family: 'Segoe UI', sans-serif;
            }
            
            .header-text {
                font-size: 2.8rem !important;
                color: var(--secondary) !important;
                text-align: center;
                margin-bottom: 2rem !important;
                font-weight: 700;
                letter-spacing: -0.5px;
            }
            
            .card {
                padding: 1.5rem;
                background: white;
                border-radius: 15px;
                box-shadow: 0 2px 15px rgba(0, 0, 0, 0.08);
                margin-bottom: 1.5rem;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                border: 1px solid rgba(0, 0, 0, 0.05);
            }
            
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
            }
            
            .stButton>button {
                background: var(--primary) !important;
                color: white !important;
                border-radius: 10px !important;
                padding: 0.75rem 1.5rem !important;
                transition: all 0.3s !important;
                border: none !important;
                font-weight: 600 !important;
            }
            
            .stButton>button:hover {
                background: #228176 !important;
                transform: scale(1.05);
                box-shadow: 0 3px 8px rgba(0, 0, 0, 0.15);
            }
            
            .chat-container {
                background: white;
                border-radius: 15px;
                padding: 1.5rem;
                height: 500px;
                overflow-y: auto;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                border: 1px solid rgba(0, 0, 0, 0.08);
            }
            
            .user-message {
                background: var(--primary);
                color: white;
                padding: 1rem 1.5rem;
                border-radius: 15px 15px 0 15px;
                margin: 0.8rem 0;
                max-width: 75%;
                float: right;
                clear: both;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            }
            
            .bot-message {
                background: #F1FAFE;
                color: var(--text);
                padding: 1rem 1.5rem;
                border-radius: 15px 15px 15px 0;
                margin: 0.8rem 0;
                max-width: 75%;
                float: left;
                clear: both;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            }
            
            .profile-img {
                width: 80px;
                height: 80px;
                border-radius: 50%;
                object-fit: cover;
                margin-right: 1.5rem;
                border: 3px solid var(--primary);
            }
            
            .sidebar .sidebar-content {
                background: var(--secondary) !important;
                color: white !important;
                padding: 1rem !important;
            }

            /* Additional CSS for dynamic UI enhancements */
            .stTextInput>div>div>input {
                border-radius: 10px;
                border: 1px solid #ccc;
                padding: 0.8rem;
            }
            
            .stFileUploader label {
                font-weight: 600;
                color: var(--secondary);
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
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    """, unsafe_allow_html=True)

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
            node_name = df.iloc[idx].get('LinkedIn Name', f'Profile {idx}')
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

# ================== MODIFIED PROFILE CARD WITH MBTI ==================
def create_profile_card(profile):
    try:
        user_mbti = st.session_state.get('user_mbti', 'ENFJ')  # Default if not set
        profile_mbti = profile.get('MBTI', '').upper()[:4]
        
        with st.container():
            compatibility = calculate_mbti_score(user_mbti, profile_mbti) if profile_mbti else 0
            score_color = "#2A9D8F" if compatibility >= 0.8 else "#E9C46A" if compatibility >= 0.6 else "#264653"
            
            st.markdown(f"""
                <div class="card">
                    <div style="display: flex; align-items: start; margin-bottom: 1rem; position: relative;">
                        <div style="position: absolute; right: 0; top: 0; background: {score_color}; 
                                    color: white; padding: 0.3rem 1rem; border-radius: 0 15px 0 15px;
                                    font-weight: 600;">
                            {compatibility*100:.0f}% Match
                        </div>
                        <img src="{profile.get('Profile Image', 'https://source.unsplash.com/featured/800x600/?nature')}" 
                             class="profile-img" 
                             onerror="this.src='https://source.unsplash.com/featured/800x600/?nature'">
                        <!-- IMAGE NOTE: Replace the above URL with your local image path or hosted URL if available -->
                        <div style="flex: 1;">
                            <h3 style="margin: 0; color: var(--secondary);">{profile.get('LinkedIn Name', 'Unknown')}</h3>
                            <div style="display: flex; align-items: center; gap: 1rem; margin: 0.5rem 0;">
                                {f'<div style="background: #E9F5F3; padding: 0.3rem 1rem; border-radius: 20px; font-size: 0.9rem;">MBTI: {profile_mbti}</div>' if profile_mbti else ''}
                                <!-- Rest of your existing profile card content -->
                            </div>
                            <!-- Keep existing profile card structure -->
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error rendering profile card: {str(e)}")

# ================== MBTI COMPATIBILITY SYSTEM ==================
MBTI_COMPATIBILITY: Dict[str, List[str]] = {
    'ESFP': ['ESFJ', 'ESTP', 'ISFP'],
    'ESTP': ['ESTJ', 'ESFP', 'INFJ'],
    'ESTJ': ['ESTP', 'ESFJ', 'ISTJ'],
    'ESFJ': ['ISTP', 'ESTJ', 'ESTP'],
    'ISTJ': ['INFJ', 'ISTP', 'ISFJ'],
    'ISTP': ['ISFP', 'INFP', 'ESFP'],
    'ISFJ': ['ESFJ', 'ISFP', 'ISTJ'],
    'ISFP': ['ESFP', 'ISFJ', 'ESFJ'],
    'ENTJ': ['INTJ', 'ENTP', 'ENFJ'],
    'ENTP': ['ENTJ', 'ENFP', 'ENFJ'],
    'ENFJ': ['ENFJ', 'INFJ', 'ENFP'],
    'ENFP': ['ENTJ', 'INTJ', 'INTP'],
    'INTJ': ['INTP', 'INFJ', 'INFP'],
    'INTP': ['ENTP', 'INFP', 'ENFP'],
    'INFJ': ['ISTJ', 'INFP', 'INTJ'],
    'INFP': ['INFJ', 'ISFJ', 'ENFJ']
}

def calculate_mbti_score(user_mbti: str, profile_mbti: str) -> float:
    """Calculate compatibility score between two MBTI types"""
    if user_mbti == profile_mbti:
        return 1.0  # 100% match
    if profile_mbti in MBTI_COMPATIBILITY.get(user_mbti, []):
        return 0.8  # 80% match
    return 0.6  # 60% base score

# ================== ENHANCED UI COMPONENTS ==================
def app_header():
    """Custom header component"""
    st.markdown("""
        <div style="background: linear-gradient(135deg, #2A9D8F, #264653); 
                    padding: 2rem; 
                    border-radius: 0 0 20px 20px;
                    margin-bottom: 2rem;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            <div style="max-width: 1200px; margin: 0 auto;">
                <h1 style="color: white; margin: 0; font-size: 2.5rem;">🌐 EcoConnect Network</h1>
                <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0;">Connecting Sustainability Leaders Worldwide</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

def app_footer():
    """Custom footer component"""
    st.markdown("""
        <div style="background: #264653; 
                    color: white;
                    padding: 2rem;
                    margin-top: 4rem;
                    border-radius: 20px 20px 0 0;">
            <div style="max-width: 1200px; margin: 0 auto; text-align: center;">
                <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 1rem;">
                    <a href="#" style="color: white; text-decoration: none;">About</a>
                    <a href="#" style="color: white; text-decoration: none;">Contact</a>
                    <a href="#" style="color: white; text-decoration: none;">Privacy</a>
                </div>
                <hr style="opacity: 0.2;">
                <p style="opacity: 0.8; margin-top: 1rem;">© 2024 EcoConnect. Empowering sustainable collaboration.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

def chat_interface(df, index, model):
    st.markdown("### 💬 Collaboration Assistant")
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.chat_history[-10:]:
            st.markdown(f'<div class="user-message">👤 {message["user"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bot-message">🤖 {message["bot"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        with st.form("chat_form", clear_on_submit=True):
            cols = st.columns([6, 1])
            with cols[0]:
                user_input = st.text_input("Type your message...", 
                                           label_visibility="collapsed", 
                                           placeholder="Ask about collaboration opportunities...",
                                           key="chat_input")
            with cols[1]:
                if st.form_submit_button("Send", use_container_width=True):
                    if user_input:
                        with st.spinner("Analyzing..."):
                            response = chat_with_rag(user_input, df, index, model)
                            st.session_state.chat_history.append({"user": user_input, "bot": response})
                            st.experimental_rerun()

# Initialize app
st.set_page_config(
    page_title="EcoConnect - Leadership Network",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌍"
)

# Inject custom CSS
inject_custom_css()

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'username' not in st.session_state:
    st.session_state.username = "Guest"

# Auth functions
def auth_screen():
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.form("Login"):
            st.markdown("""
                <div style="text-align: center; margin-bottom: 2rem;">
                    <h1 style="color: var(--secondary);">🌿 Welcome Back</h1>
                    <p style="color: var(--text);">Sign in to continue</p>
                </div>
            """, unsafe_allow_html=True)
            
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            
            if st.form_submit_button("Login →", use_container_width=True):
                if username and password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.experimental_rerun()
                else:
                    st.error("Please enter both username and password")

    with col2:
        with st.form("Signup"):
            st.markdown("""
                <div style="text-align: center; margin-bottom: 2rem;">
                    <h1 style="color: white;">🌱 Join Our Network</h1>
                    <p style="color: rgba(255,255,255,0.8);">Connect with sustainability leaders worldwide</p>
                </div>
            """, unsafe_allow_html=True)
            
            new_user = st.text_input("Choose Username", key="signup_user")
            new_pass = st.text_input("Create Password", type="password", key="signup_pass")
            
            if st.form_submit_button("Get Started →", use_container_width=True):
                if new_user and new_pass:
                    st.success("Account created! Please login")
                else:
                    st.error("Please fill all fields")

# Main app
if not st.session_state.logged_in:
    auth_screen()
else:
    app_header()
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
            st.experimental_rerun()

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
            chat_interface(df, index, model)
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

        app_footer()
