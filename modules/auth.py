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
    st.markdown(
        """
        <div style="text-align: center; margin-top: 1rem; margin-bottom: 2rem;">
            <div style="font-size: 3.2rem; margin-bottom: 0.5rem;">🛡️</div>
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: 0.04em; color: #0f172a;">
                COMMUNITY SAFETY & TRUST AI
            </h1>
            <p style="color: #64748b; font-size: 0.95rem; margin-top: 0.4rem;">
                Secure Access Portal • Authentication Powered by MongoDB Cloud
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mongo_uri = get_mongo_uri()
    col_left, col_center, col_right = st.columns([1, 2.2, 1])

    with col_center:
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
                '<div style="margin: 0.8rem 0 1.2rem; color: #475569; font-size: 0.88rem;">'
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
                '<div style="margin: 0.8rem 0 1.2rem; color: #475569; font-size: 0.88rem;">'
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
