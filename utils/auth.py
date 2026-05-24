import hashlib
import streamlit as st

# Default seeded credentials with roles
DEFAULT_USERS = {
    "admin": {
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "Admin",
        "fullname": "Admin"
    },
    "analyst": {
        "password_hash": hashlib.sha256("analyst123".encode()).hexdigest(),
        "role": "Analyst",
        "fullname": "Senior Data Analyst"
    }
}

def verify_credentials(username, password):
    """
    Verifies username and password against seeded credentials.
    """
    username = username.strip().lower()
    if username in DEFAULT_USERS:
        input_hash = hashlib.sha256(password.encode()).hexdigest()
        if input_hash == DEFAULT_USERS[username]["password_hash"]:
            return DEFAULT_USERS[username]
    return None

def init_auth_session():
    """
    Initializes stream session states for authentication tracking.
    """
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "user_role" not in st.session_state:
        st.session_state.user_role = None
    if "fullname" not in st.session_state:
        st.session_state.fullname = None

def check_permission(required_role="Analyst"):
    """
    Checks if the currently logged in user has the required permission role.
    Admin overrides all permissions.
    """
    if not st.session_state.authenticated:
        return False
        
    current_role = st.session_state.user_role
    if current_role == "Admin":
        return True # Admin gets everything
        
    if required_role == "Analyst" and current_role == "Analyst":
        return True
        
    return False

def show_login_interface():
    """
    Renders a stunning glassmorphic login screen for user access.
    """
    init_auth_session()
    
    st.markdown("""
        <style>
            /* Scoped Dark Login Inputs & Light Text */
            div[data-baseweb="input"] {
                background-color: #1e1b4b !important;
                border: 1px solid #352e8c !important;
                color: #ffffff !important;
                border-radius: 12px !important;
                transition: all 0.2s ease-in-out !important;
            }
            div[data-baseweb="input"] input {
                color: #ffffff !important;
            }
            div[data-baseweb="input"] input::placeholder {
                color: #a5b4fc !important;
                opacity: 0.75 !important;
            }
            div[data-baseweb="input"]:focus-within {
                background-color: #12103e !important;
                border-color: #ff9f1c !important;
                box-shadow: 0 0 0 3px rgba(255, 159, 28, 0.15) !important;
            }
            
            /* Password visibility button override to fit dark inputs */
            div[data-testid="stTextInput"] button, div[data-baseweb="input"] button {
                background-color: transparent !important;
                border: none !important;
                color: #a5b4fc !important;
                box-shadow: none !important;
                padding: 0 !important;
                margin-right: 8px !important;
                width: auto !important;
                min-width: unset !important;
                height: auto !important;
            }
            div[data-testid="stTextInput"] button:hover, div[data-baseweb="input"] button:hover {
                color: #ffffff !important;
                background-color: transparent !important;
                border: none !important;
                transform: none !important;
                box-shadow: none !important;
            }
            div[data-testid="stTextInput"] button svg, div[data-baseweb="input"] svg {
                fill: #a5b4fc !important;
            }
            div[data-testid="stTextInput"] button:hover svg, div[data-baseweb="input"] button:hover svg {
                fill: #ffffff !important;
            }
        </style>
        
        <div class="login-wrapper">
            <h1 style='text-align: center; color: #352e8c; font-family: "Outfit", sans-serif; font-size: 3rem; margin-bottom: 5px; font-weight: 900; letter-spacing: -0.02em; text-transform: uppercase; background: linear-gradient(90deg, #352e8c, #5ac8fa); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>BUISNESSVERSE</h1>
            <p style='text-align: center; color: #4a467f; font-family: "Inter", sans-serif; margin-bottom: 30px; font-weight: 700; font-size: 1.2rem; text-transform: uppercase;'>ENTERPRISE SMART BUSINESS ANALYTICS & FORECASTING PLATFORM</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.container(border=True):
            st.subheader("System Authentication")
            
            username = st.text_input("Username", placeholder="e.g. admin or analyst")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            submit_btn = st.button("Access Platform", use_container_width=True)
            
            if submit_btn:
                user_data = verify_credentials(username, password)
                if user_data:
                    st.session_state.authenticated = True
                    st.session_state.username = username.lower()
                    st.session_state.user_role = user_data["role"]
                    st.session_state.fullname = user_data["fullname"]
                    st.toast(f"Welcome back, {user_data['fullname']}!", icon="🔥")
                    st.rerun()
                else:
                    st.error("Invalid security credentials. Please try again.")
        

