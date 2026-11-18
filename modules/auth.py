"""
Authentication and User Management Module for Community Safety & Trust AI.
Uses MongoDB Cloud (MongoDB Atlas) exclusively for User Registration and Login.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
import re
from typing import Optional, Tuple, Dict, Any

import bcrypt
from dotenv import load_dotenv, set_key
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError, ServerSelectionTimeoutError
import streamlit as st

# Automatically load .env if present
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE, override=True)

DEFAULT_DB_NAME = "community_safety_trust_db"
USER_COLLECTION = "users"
USERNAME_REGEX = re.compile(r"^[a-zA-Z0-9_]{3,30}$")
EMAIL_REGEX = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$")


def get_mongo_uri() -> Optional[str]:
    """Retrieve MongoDB Cloud connection URI from Streamlit secrets, environment, or .env."""
    # 1. Check Streamlit secrets
    try:
        if "MONGODB_URI" in st.secrets:
            return st.secrets["MONGODB_URI"]
        if "mongo" in st.secrets and "uri" in st.secrets["mongo"]:
            return st.secrets["mongo"]["uri"]
    except Exception:
        pass

    # 2. Check environment variables
    uri = os.getenv("MONGODB_URI") or os.getenv("MONGO_URI")
    if uri:
        return uri.strip()

    # 3. Reload dotenv if file exists and retry
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=True)
        uri = os.getenv("MONGODB_URI") or os.getenv("MONGO_URI")
        if uri:
            return uri.strip()

    return None


def get_db_name() -> str:
    """Retrieve database name from secrets or environment."""
    try:
        if "MONGODB_DB_NAME" in st.secrets:
            return st.secrets["MONGODB_DB_NAME"]
    except Exception:
        pass
    return os.getenv("MONGODB_DB_NAME", DEFAULT_DB_NAME).strip()


@st.cache_resource(show_spinner=False)
def get_mongo_client(uri: str) -> MongoClient:
    """Create and cache a MongoClient with a safe 5-second server timeout."""
    return MongoClient(
        uri,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        appname="CommunitySafetyTrustAI",
    )


def test_mongo_connection() -> Tuple[bool, str]:
    """Test connection to MongoDB Cloud."""
    uri = get_mongo_uri()
    if not uri:
        return False, "MONGODB_URI is not configured in .env or .streamlit/secrets.toml."
    try:
        client = get_mongo_client(uri)
        client.admin.command("ping")
        return True, "Successfully connected to MongoDB Cloud cluster!"
    except (ConnectionFailure, ServerSelectionTimeoutError) as err:
        return False, f"Could not connect to MongoDB Atlas cluster: {err}"
    except Exception as err:
        return False, f"Connection error: {err}"


def save_mongo_uri_to_env(uri: str, db_name: str = DEFAULT_DB_NAME) -> Tuple[bool, str]:
    """Save MongoDB Cloud URI to local .env file."""
    uri = uri.strip()
    if not uri:
        return False, "URI cannot be empty."
    if not uri.startswith("mongodb://") and not uri.startswith("mongodb+srv://"):
        return False, "Invalid URI format. Expected mongodb+srv:// or mongodb://"

    try:
        if not ENV_FILE.exists():
            ENV_FILE.write_text("# Community Safety & Trust AI Environment Config\n", encoding="utf-8")
        set_key(str(ENV_FILE), "MONGODB_URI", uri)
        if db_name:
            set_key(str(ENV_FILE), "MONGODB_DB_NAME", db_name.strip())
        os.environ["MONGODB_URI"] = uri
        if db_name:
            os.environ["MONGODB_DB_NAME"] = db_name.strip()
        # Invalidate cached client to pick up new connection
        get_mongo_client.clear()
        return True, "Configuration saved successfully!"
    except Exception as err:
        return False, f"Failed to save .env file: {err}"


def get_users_collection():
    """Retrieve the users collection and ensure required indexes exist."""
    uri = get_mongo_uri()
    if not uri:
        raise ValueError("MongoDB URI is not configured.")

    client = get_mongo_client(uri)
    db = client[get_db_name()]
    users = db[USER_COLLECTION]

    # Ensure unique indexes (safe to call repeatedly)
    try:
        users.create_index("username", unique=True, background=True)
        users.create_index("email", unique=True, background=True)
    except Exception:
        pass

    return users


def register_user(
    full_name: str,
    username: str,
    email: str,
    password: str,
    confirm_password: str,
) -> Tuple[bool, str]:
    """
    Validate and register a new user in MongoDB Cloud.
    Passwords are encrypted with bcrypt before saving.
    """
    # 1. Field validation
    full_name = full_name.strip()
    username = username.strip().lower()
    email = email.strip().lower()

    if not full_name:
        return False, "Please enter your full name."
    if not username:
        return False, "Please enter a username."
    if not USERNAME_REGEX.match(username):
        return False, "Username must be 3-30 characters long and contain only letters, numbers, or underscores."
    if not email or not EMAIL_REGEX.match(email):
        return False, "Please enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
    if password != confirm_password:
        return False, "Passwords do not match."

    # 2. Check MongoDB configuration
    uri = get_mongo_uri()
    if not uri:
        return False, "MongoDB Cloud URI is not configured. Please add MONGODB_URI to your .env or .streamlit/secrets.toml file."

    try:
        users = get_users_collection()

        # Check existing username or email
        existing_user = users.find_one({"$or": [{"username": username}, {"email": email}]})
        if existing_user:
            if existing_user.get("username") == username:
                return False, f"Username '{username}' is already taken. Please choose another."
            if existing_user.get("email") == email:
                return False, f"An account with email '{email}' already exists. Please sign in."

        # 3. Hash password securely
        password_bytes = password.encode("utf-8")
        salt = bcrypt.gensalt(rounds=12)
        password_hash = bcrypt.hashpw(password_bytes, salt).decode("utf-8")

        # 4. Insert user record
        now = datetime.now(timezone.utc)
        user_doc = {
            "full_name": full_name,
            "username": username,
            "email": email,
            "password_hash": password_hash,
            "created_at": now,
            "last_login": now,
        }
        users.insert_one(user_doc)
        return True, "Registration successful! You can now sign in."

    except DuplicateKeyError:
        return False, "A user with this username or email already exists."
    except (ConnectionFailure, ServerSelectionTimeoutError) as err:
        return False, f"Unable to reach MongoDB Cloud: {err}. Please verify your cluster URL and IP Whitelist."
    except Exception as err:
        return False, f"Registration failed: {err}"


def authenticate_user(
    identifier: str,
    password: str,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Authenticate user against MongoDB Cloud credentials.
    Identifier can be either username or email.
    """
    identifier = identifier.strip().lower()
    if not identifier:
        return False, "Please enter your username or email.", None
    if not password:
        return False, "Please enter your password.", None

    uri = get_mongo_uri()
    if not uri:
        return False, "MongoDB Cloud URI is not configured. Please configure MONGODB_URI.", None

    try:
        users = get_users_collection()
        user = users.find_one({"$or": [{"username": identifier}, {"email": identifier}]})
        if not user:
            return False, "Invalid username/email or password.", None

        stored_hash = user.get("password_hash", "")
        if not stored_hash:
            return False, "Account error. Please contact administrator.", None

        # Verify password hash
        if bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8")):
            # Update last login timestamp if _id is present
            if "_id" in user:
                try:
                    users.update_one(
                        {"_id": user["_id"]},
                        {"$set": {"last_login": datetime.now(timezone.utc)}},
                    )
                except Exception:
                    pass
            user_data = {
                "username": user.get("username"),
                "email": user.get("email"),
                "full_name": user.get("full_name", user.get("username")),
                "created_at": user.get("created_at"),
            }
            return True, "Login successful!", user_data
        else:
            return False, "Invalid username/email or password.", None

    except (ConnectionFailure, ServerSelectionTimeoutError) as err:
        return False, f"Unable to connect to MongoDB Cloud: {err}. Please check network or IP Whitelist.", None
    except Exception as err:
        return False, f"Authentication error: {err}", None


def render_auth_page():
    """
    Renders the unified Register & Login interface for Community Safety & Trust AI.
    Uses MongoDB Cloud Atlas exclusively for credential management.
    """
    # Professional AI Cybersecurity Light Theme & Styling
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

        /* Auth Page Global Background */
        .stApp {
            background-color: #F5F8FF !important;
            background-image: 
                radial-gradient(circle at 10% 12%, rgba(37, 99, 235, 0.08) 0%, transparent 45%),
                radial-gradient(circle at 90% 88%, rgba(6, 182, 212, 0.08) 0%, transparent 45%),
                radial-gradient(circle at 50% 50%, rgba(15, 118, 110, 0.04) 0%, transparent 50%),
                linear-gradient(rgba(37, 99, 235, 0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(37, 99, 235, 0.02) 1px, transparent 1px) !important;
            background-size: auto, auto, auto, 36px 36px, 36px 36px !important;
        }

        /* Auth Card Container */
        .st-key-auth_card {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 24px !important;
            padding: 2.2rem 2.4rem !important;
            box-shadow: 0 20px 45px -10px rgba(20, 33, 61, 0.08), 0 1px 3px rgba(20, 33, 61, 0.04) !important;
        }

        .st-key-auth_card div[data-testid="stForm"] {
            background: transparent !important;
            border: none !important;
            padding: 0.2rem 0 0.8rem 0 !important;
            box-shadow: none !important;
        }

        /* Typography */
        .auth-hero-title {
            color: #14213D;
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1.15;
            letter-spacing: -0.02em;
            margin: 0.6rem 0 0.7rem;
        }
        .auth-hero-title span {
            background: linear-gradient(135deg, #2563EB 0%, #06B6D4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .auth-hero-desc {
            color: #64748B;
            font-size: 0.92rem;
            line-height: 1.6;
            margin-bottom: 1.4rem;
        }

        /* High-Tech Badge */
        .auth-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            background: rgba(37, 99, 235, 0.08);
            border: 1px solid rgba(37, 99, 235, 0.22);
            border-radius: 999px;
            padding: 0.32rem 0.8rem;
            font-family: 'DM Mono', monospace;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            color: #2563EB;
            text-transform: uppercase;
        }

        /* Feature Micro-Cards */
        .auth-feature-pill {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.65rem;
            box-shadow: 0 2px 6px rgba(20, 33, 61, 0.025);
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        }
        .auth-feature-pill:hover {
            transform: translateY(-2px);
            border-color: #06B6D4;
            box-shadow: 0 6px 18px rgba(6, 182, 212, 0.12);
        }
        .auth-feature-icon {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.09) 0%, rgba(6, 182, 212, 0.12) 100%);
            display: grid;
            place-items: center;
            font-size: 1.1rem;
            flex-shrink: 0;
            border: 1px solid rgba(37, 99, 235, 0.12);
        }
        .auth-feature-title {
            font-size: 0.84rem;
            font-weight: 750;
            color: #172033;
            line-height: 1.25;
        }
        .auth-feature-desc {
            font-size: 0.72rem;
            color: #64748B;
            margin-top: 0.1rem;
        }

        /* Tabs Styling */
        .st-key-auth_card [data-baseweb="tab-list"] {
            background-color: #F1F5F9 !important;
            padding: 0.3rem !important;
            border-radius: 14px !important;
            gap: 0.35rem !important;
            border: 1px solid #E2E8F0 !important;
            margin-bottom: 0.9rem !important;
        }
        .st-key-auth_card [data-baseweb="tab"] {
            border-radius: 10px !important;
            padding: 0.55rem 1.1rem !important;
            font-weight: 700 !important;
            font-size: 0.85rem !important;
            color: #64748B !important;
            border: none !important;
            background: transparent !important;
            transition: all 0.2s ease !important;
        }
        .st-key-auth_card [data-baseweb="tab"][aria-selected="true"] {
            background-color: #FFFFFF !important;
            color: #14213D !important;
            box-shadow: 0 2px 8px rgba(20, 33, 61, 0.08) !important;
        }
        .st-key-auth_card [data-baseweb="tab-highlight"],
        .st-key-auth_card [data-baseweb="tab-border"] {
            display: none !important;
        }

        /* Inputs Styling */
        .st-key-auth_card div[data-testid="stTextInput"] input {
            background-color: #F1F5F9 !important;
            border: 1.5px solid #E2E8F0 !important;
            border-radius: 12px !important;
            color: #172033 !important;
            font-size: 0.88rem !important;
            font-weight: 550 !important;
            padding: 0.65rem 0.95rem !important;
            transition: all 0.2s ease !important;
        }
        .st-key-auth_card div[data-testid="stTextInput"] input:focus {
            background-color: #FFFFFF !important;
            border-color: #2563EB !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.16) !important;
        }
        .st-key-auth_card div[data-testid="stTextInput"] label {
            color: #14213D !important;
            font-weight: 700 !important;
            font-size: 0.82rem !important;
            letter-spacing: -0.01em !important;
        }

        /* Password visibility eye icon */
        .st-key-auth_card div[data-testid="stTextInput"] button {
            border: none !important;
            background: transparent !important;
            color: #64748B !important;
            box-shadow: none !important;
        }
        .st-key-auth_card div[data-testid="stTextInput"] button:hover {
            color: #2563EB !important;
            background: transparent !important;
        }

        /* Gradient Buttons for Primary Actions */
        .st-key-auth_card div[data-testid="stFormSubmitButton"] > button[kind="primary"],
        .st-key-auth_card .stButton > button[kind="primary"],
        div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, #2563EB 0%, #06B6D4 100%) !important;
            border: none !important;
            border-radius: 12px !important;
            color: #FFFFFF !important;
            font-size: 0.94rem !important;
            font-weight: 750 !important;
            letter-spacing: 0.02em !important;
            padding: 0.75rem 1.4rem !important;
            min-height: 2.9rem !important;
            box-shadow: 0 4px 16px rgba(37, 99, 235, 0.28) !important;
            transition: all 0.22s ease !important;
        }
        .st-key-auth_card div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover,
        .st-key-auth_card .stButton > button[kind="primary"]:hover,
        div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #1D4ED8 0%, #0891B2 100%) !important;
            box-shadow: 0 8px 24px rgba(37, 99, 235, 0.42) !important;
            transform: translateY(-1.5px) !important;
        }
        .st-key-auth_card div[data-testid="stFormSubmitButton"] > button[kind="primary"]:active,
        .st-key-auth_card .stButton > button[kind="primary"]:active {
            transform: translateY(0px) !important;
        }

        /* Secondary Buttons */
        .st-key-auth_card .stButton > button:not([kind="primary"]) {
            background: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 12px !important;
            color: #14213D !important;
            font-weight: 650 !important;
        }
        .st-key-auth_card .stButton > button:not([kind="primary"]):hover {
            background: #F1F5F9 !important;
            border-color: #94A3B8 !important;
        }

        /* Expander Styling */
        .st-key-auth_card div[data-testid="stExpander"] {
            background: #F8FAFC !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 14px !important;
            margin-bottom: 1.1rem !important;
            box-shadow: 0 1px 3px rgba(20, 33, 61, 0.02) !important;
        }
        .st-key-auth_card div[data-testid="stExpander"] summary {
            font-weight: 700 !important;
            color: #14213D !important;
            font-size: 0.84rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    mongo_uri = get_mongo_uri()
    col_hero, col_card = st.columns([1, 1.2], gap="large")

    # Left Column: AI Cybersecurity Visual Showcase
    with col_hero:
        # SVG AI Shield & Neural Circuit Illustration
        st.markdown(
            """
            <svg viewBox="0 0 460 280" fill="none" xmlns="http://www.w3.org/2000/svg" style="width:100%; max-width:440px; height:auto; display:block; margin: 0.5rem auto 1.2rem;">
              <defs>
                <linearGradient id="shieldGrad" x1="50" y1="20" x2="350" y2="280" gradientUnits="userSpaceOnUse">
                  <stop offset="0%" stop-color="#2563EB" />
                  <stop offset="60%" stop-color="#06B6D4" />
                  <stop offset="100%" stop-color="#0F766E" />
                </linearGradient>
                <linearGradient id="coreGrad" x1="190" y1="110" x2="270" y2="190" gradientUnits="userSpaceOnUse">
                  <stop offset="0%" stop-color="#3B82F6" />
                  <stop offset="100%" stop-color="#06B6D4" />
                </linearGradient>
                <linearGradient id="traceGrad" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0%" stop-color="#2563EB" stop-opacity="0.2" />
                  <stop offset="100%" stop-color="#06B6D4" stop-opacity="0.8" />
                </linearGradient>
                <radialGradient id="cyberGlow" cx="230" cy="140" r="130" gradientUnits="userSpaceOnUse">
                  <stop offset="0%" stop-color="#06B6D4" stop-opacity="0.18" />
                  <stop offset="60%" stop-color="#2563EB" stop-opacity="0.08" />
                  <stop offset="100%" stop-color="#F5F8FF" stop-opacity="0" />
                </radialGradient>
              </defs>

              <circle cx="230" cy="140" r="125" fill="url(#cyberGlow)" />
              <circle cx="230" cy="140" r="115" stroke="#E2E8F0" stroke-width="1.5" stroke-dasharray="4 6" opacity="0.8" />
              <circle cx="230" cy="140" r="80" stroke="#CBD5E1" stroke-width="1" stroke-dasharray="8 8" opacity="0.6" />
              <circle cx="230" cy="140" r="48" stroke="#06B6D4" stroke-width="1.2" stroke-opacity="0.4" />

              <path d="M70 50 H 130 L 165 85 H 200" stroke="url(#traceGrad)" stroke-width="1.5" stroke-linecap="round" fill="none" />
              <circle cx="70" cy="50" r="3.5" fill="#2563EB" />
              <circle cx="130" cy="50" r="2.5" fill="#06B6D4" />
              <circle cx="200" cy="85" r="3.5" fill="#06B6D4" />

              <path d="M50 220 H 120 L 160 185 H 195" stroke="url(#traceGrad)" stroke-width="1.5" stroke-linecap="round" fill="none" />
              <circle cx="50" cy="220" r="3.5" fill="#2563EB" />
              <circle cx="160" cy="185" r="2.5" fill="#06B6D4" />
              <circle cx="195" cy="185" r="3" fill="#0F766E" />

              <path d="M390 60 H 330 L 295 95 H 260" stroke="url(#traceGrad)" stroke-width="1.5" stroke-linecap="round" fill="none" />
              <circle cx="390" cy="60" r="3.5" fill="#06B6D4" />
              <circle cx="330" cy="60" r="2.5" fill="#2563EB" />
              <circle cx="260" cy="95" r="3.5" fill="#06B6D4" />

              <path d="M410 210 H 340 L 300 175 H 265" stroke="url(#traceGrad)" stroke-width="1.5" stroke-linecap="round" fill="none" />
              <circle cx="410" cy="210" r="3.5" fill="#0F766E" />
              <circle cx="300" cy="175" r="2.5" fill="#06B6D4" />
              <circle cx="265" cy="175" r="3" fill="#2563EB" />

              <polygon points="120,125 130,130 130,142 120,147 110,142 110,130" fill="#F1F5F9" stroke="#2563EB" stroke-width="1.5" />
              <polygon points="340,125 350,130 350,142 340,147 330,142 330,130" fill="#F1F5F9" stroke="#06B6D4" stroke-width="1.5" />

              <path d="M 230 35 L 300 68 C 300 145 265 200 230 225 C 195 200 160 145 160 68 Z" 
                    fill="#FFFFFF" 
                    stroke="url(#shieldGrad)" 
                    stroke-width="3" 
                    stroke-linejoin="round" />

              <path d="M 230 42 L 290 72 C 290 138 260 186 230 210 Z" fill="rgba(37, 99, 235, 0.05)" />
              <path d="M 230 42 L 170 72 C 170 138 200 186 230 210 Z" fill="rgba(6, 182, 212, 0.08)" />

              <circle cx="230" cy="132" r="32" fill="url(#coreGrad)" />
              <circle cx="230" cy="132" r="36" stroke="#06B6D4" stroke-width="1.5" stroke-dasharray="6 4" opacity="0.75" />

              <rect x="220" y="126" width="20" height="16" rx="3" fill="#FFFFFF" />
              <path d="M 223 126 V 121 C 223 117 237 117 237 121 V 126" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round" fill="none" />
              <circle cx="230" cy="133" r="2" fill="#14213D" />
              <path d="M 230 135 V 138" stroke="#14213D" stroke-width="1.5" stroke-linecap="round" />

              <path d="M 215 92 L 230 80 L 245 92" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none" />
              <circle cx="230" cy="64" r="3" fill="#06B6D4" />

              <text x="75" y="38" font-family="'DM Mono', monospace" font-size="9" fill="#64748B" letter-spacing="1">SEC_LEVEL // 01</text>
              <text x="315" y="48" font-family="'DM Mono', monospace" font-size="9" fill="#06B6D4" letter-spacing="1">AI_SHIELD // ON</text>
              <text x="60" y="240" font-family="'DM Mono', monospace" font-size="9" fill="#2563EB" letter-spacing="1">256-BIT BCRYPT</text>
              <text x="315" y="230" font-family="'DM Mono', monospace" font-size="9" fill="#0F766E" letter-spacing="1">MONGODB ATLAS</text>
            </svg>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="auth-badge">🛡️ Enterprise AI Defense Platform</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-hero-title">Protecting Trust With <span>Responsible AI</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-hero-desc">A unified intelligence workspace for visual screening, spatial awareness, and digital trust. Delivering enterprise-grade security and fraud mitigation.</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="auth-feature-pill">
                <div class="auth-feature-icon">💵</div>
                <div>
                    <div class="auth-feature-title">Neural Banknote Screening</div>
                    <div class="auth-feature-desc">Deep Convolutional Neural Network counterfeit detection</div>
                </div>
            </div>
            <div class="auth-feature-pill">
                <div class="auth-feature-icon">👁️</div>
                <div>
                    <div class="auth-feature-title">Spatial Vision & Crowd AI</div>
                    <div class="auth-feature-desc">Real-time YOLOv11 object identification and density review</div>
                </div>
            </div>
            <div class="auth-feature-pill">
                <div class="auth-feature-icon">📰</div>
                <div>
                    <div class="auth-feature-title">Cognitive Trust NLP Engine</div>
                    <div class="auth-feature-desc">Bilingual Hindi/English fake news and claim classification</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Right Column: Existing Authentication Form Card
    with col_card:
        with st.container(key="auth_card"):
            # Header Badge & Title
            st.markdown(
                """
                <div style="margin-bottom: 1.1rem; padding-bottom: 0.8rem; border-bottom: 1px solid #E2E8F0;">
                    <div style="display:flex; align-items:center; gap:0.75rem;">
                        <div style="font-size:1.6rem; background:linear-gradient(135deg, rgba(37,99,235,0.08) 0%, rgba(6,182,212,0.12) 100%); border:1px solid #E2E8F0; border-radius:12px; width:44px; height:44px; display:grid; place-items:center;">🛡️</div>
                        <div>
                            <div style="font-size:1.15rem; font-weight:800; color:#14213D; letter-spacing:0.02em; line-height:1.2;">COMMUNITY SAFETY & TRUST AI</div>
                            <div style="font-size:0.68rem; color:#2563EB; font-family:'DM Mono', monospace; font-weight:700; letter-spacing:0.06em; text-transform:uppercase;">Secure Intelligence Portal</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Check and display MongoDB Cloud status
            if not mongo_uri:
                st.warning(
                    "⚠️ **MongoDB Cloud is not configured.**\n\n"
                    "Please configure your MongoDB Atlas connection string below to enable registration and login."
                )
                with st.expander("⚙️ Connect MongoDB Atlas Cloud", expanded=True):
                    st.markdown(
                        """
                        Enter your MongoDB Atlas connection string:
                        ```text
                        mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
                        ```
                        *Credentials are securely stored locally in your `.env` file and excluded from Git.*
                        """
                    )
                    with st.form("mongo_config_form"):
                        input_uri = st.text_input(
                            "MongoDB Cloud URI",
                            type="password",
                            placeholder="mongodb+srv://user:pass@cluster.mongodb.net/...",
                        )
                        input_db = st.text_input(
                            "Database Name",
                            value=DEFAULT_DB_NAME,
                        )
                        submitted = st.form_submit_button("Save & Connect →", type="primary", use_container_width=True)
                        if submitted:
                            ok, msg = save_mongo_uri_to_env(input_uri, input_db)
                            if ok:
                                test_ok, test_msg = test_mongo_connection()
                                if test_ok:
                                    st.success("Connected to MongoDB Cloud successfully!")
                                    st.rerun()
                                else:
                                    st.warning(f"Saved to .env, but connection test failed: {test_msg}")
                            else:
                                st.error(msg)
            else:
                # Quick status indicator
                with st.expander("☁️ MongoDB Cloud Connection Status", expanded=False):
                    st.caption(f"Database: `{get_db_name()}` | Collection: `{USER_COLLECTION}`")
                    if st.button("Test MongoDB Atlas Connection", key="test_mongo_conn"):
                        with st.spinner("Testing connection to MongoDB Cloud..."):
                            ok, msg = test_mongo_connection()
                        if ok:
                            st.success(f"✅ {msg}")
                        else:
                            st.error(f"❌ {msg}")
                    # Allow updating URI if desired
                    st.divider()
                    st.caption("Need to update your connection string?")
                    with st.form("update_mongo_uri_form"):
                        new_uri = st.text_input("New MongoDB URI", type="password")
                        new_db = st.text_input("Database Name", value=get_db_name())
                        if st.form_submit_button("Update Configuration"):
                            ok, msg = save_mongo_uri_to_env(new_uri, new_db)
                            if ok:
                                st.success("Updated! Testing connection...")
                                test_ok, test_msg = test_mongo_connection()
                                if test_ok:
                                    st.success(test_msg)
                                    st.rerun()
                                else:
                                    st.warning(test_msg)
                            else:
                                st.error(msg)

            # Tabbed Authentication Interface
            tab_login, tab_register = st.tabs(["🔐 Sign In", "📝 Create Account"])

            with tab_login:
                st.markdown(
                    '<div style="margin: 0.6rem 0 1rem; color: #64748B; font-size: 0.88rem;">'
                    'Enter your credentials to access the safety intelligence platform.'
                    '</div>',
                    unsafe_allow_html=True,
                )
                with st.form("auth_login_form", clear_on_submit=False):
                    login_identifier = st.text_input(
                        "Username or Email",
                        placeholder="Enter your username or email address",
                        key="login_user_input",
                    )
                    login_password = st.text_input(
                        "Password",
                        type="password",
                        placeholder="Enter your password",
                        key="login_pass_input",
                    )
                    login_btn = st.form_submit_button(
                        "Sign In →",
                        type="primary",
                        use_container_width=True,
                        disabled=not bool(mongo_uri),
                    )
                    if login_btn:
                        with st.spinner("Authenticating with MongoDB Cloud..."):
                            success, message, user_data = authenticate_user(login_identifier, login_password)
                        if success:
                            st.session_state["authenticated"] = True
                            st.session_state["user"] = user_data
                            st.session_state["module"] = None
                            st.success(f"Welcome back, {user_data.get('full_name')}!")
                            st.rerun()
                        else:
                            st.error(message)

            with tab_register:
                st.markdown(
                    '<div style="margin: 0.6rem 0 1rem; color: #64748B; font-size: 0.88rem;">'
                    'Register a new account. Your credentials are securely hashed and stored in MongoDB Cloud.'
                    '</div>',
                    unsafe_allow_html=True,
                )
                with st.form("auth_register_form", clear_on_submit=False):
                    reg_full_name = st.text_input(
                        "Full Name",
                        placeholder="e.g. Alex Morgan",
                        key="reg_name_input",
                    )
                    reg_username = st.text_input(
                        "Username",
                        placeholder="3-30 letters, numbers, or underscores",
                        key="reg_username_input",
                    )
                    reg_email = st.text_input(
                        "Email Address",
                        placeholder="e.g. alex@example.com",
                        key="reg_email_input",
                    )
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        reg_password = st.text_input(
                            "Password",
                            type="password",
                            placeholder="At least 6 characters",
                            key="reg_password_input",
                        )
                    with col_p2:
                        reg_confirm = st.text_input(
                            "Confirm Password",
                            type="password",
                            placeholder="Repeat your password",
                            key="reg_confirm_input",
                        )

                    register_btn = st.form_submit_button(
                        "Create Account →",
                        type="primary",
                        use_container_width=True,
                        disabled=not bool(mongo_uri),
                    )
                    if register_btn:
                        with st.spinner("Registering user in MongoDB Cloud..."):
                            success, message = register_user(
                                reg_full_name,
                                reg_username,
                                reg_email,
                                reg_password,
                                reg_confirm,
                            )
                        if success:
                            st.success(f"✅ {message}")
                            st.info("Switch to the **Sign In** tab to log in with your new credentials.")
                        else:
                            st.error(f"❌ {message}")

            st.caption(
                "🔒 Passwords are cryptographically salted and hashed using bcrypt before transmission to MongoDB Cloud."
            )
