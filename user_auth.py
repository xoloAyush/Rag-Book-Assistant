import os
import re
import uuid
import secrets
import time
from datetime import datetime, timezone, timedelta

import bcrypt
import jwt
import resend
import streamlit as st
from streamlit_cookies_controller import CookieController
from dotenv import load_dotenv
from pymongo.errors import PyMongoError

load_dotenv()

# ======================================================
# Authentication & JWT Configuration
# ======================================================

JWT_SECRET = os.getenv("JWT_SECRET", "super_secret_jwt_key_rag_book_assistant_2026_change_in_prod")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_EXPIRE_MINUTES", "1440"))  # 24 hours
JWT_RESET_EXPIRE_MINUTES = int(os.getenv("JWT_RESET_EXPIRE_MINUTES", "15"))      # 15 minutes
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "").strip()
FROM_EMAIL = os.getenv("FROM_EMAIL", "onboarding@resend.dev").strip()
APP_URL = os.getenv("APP_URL", "http://localhost:8501").strip()

AUTH_COOKIE_NAME = "rag_auth_jwt"
COOKIE_MAX_AGE = 7 * 24 * 3600  # 7 days in seconds


# ======================================================
# Password Cryptography (bcrypt)
# ======================================================

def hash_password(password: str) -> str:
    """Hash a plaintext password with a random bcrypt salt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a stored bcrypt hash."""
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )
    except Exception:
        return False


# ======================================================
# JWT Token Management (PyJWT)
# ======================================================

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a signed JWT access token."""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=JWT_ACCESS_EXPIRE_MINUTES))

    to_encode.update({
        "exp": expire,
        "iat": now,
        "token_type": "access",
    })
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_reset_token(email: str, expires_minutes: int | None = None) -> str:
    """Create a short-lived signed JWT token specifically for password reset."""
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=expires_minutes or JWT_RESET_EXPIRE_MINUTES)

    payload = {
        "sub": email.lower(),
        "email": email.lower(),
        "token_type": "password_reset",
        "jti": secrets.token_hex(16),
        "iat": now,
        "exp": expire,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str, expected_type: str = "access") -> dict | None:
    """
    Verify and decode a JWT token.
    Returns the decoded payload if valid and matches expected_type, otherwise None.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("token_type") != expected_type:
            return None
        return payload
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError, Exception):
        return None


# ======================================================
# Email Sending via Resend (with Graceful Dev Fallback)
# ======================================================

def send_reset_email(to_email: str, reset_token: str, base_url: str = APP_URL) -> tuple[bool, str]:
    """
    Send a password reset email using the Resend API.
    If RESEND_API_KEY is not configured or sending fails, returns (False, reason).
    """
    api_key = os.getenv("RESEND_API_KEY", "").strip() or RESEND_API_KEY
    if not api_key:
        return False, "RESEND_API_KEY is not configured in .env."

    reset_url = f"{base_url.rstrip('/')}/?reset_token={reset_token}"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; margin: 0; padding: 20px; }}
        .container {{ max-width: 580px; margin: 0 auto; background: #ffffff; border-radius: 12px; padding: 32px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
        .header {{ text-align: center; border-bottom: 1px solid #e9ecef; padding-bottom: 20px; }}
        .header h1 {{ margin: 0; color: #1e293b; font-size: 24px; }}
        .content {{ padding: 24px 0; color: #334155; line-height: 1.6; font-size: 15px; }}
        .btn {{ display: inline-block; background-color: #2563eb; color: #ffffff !important; text-decoration: none; padding: 12px 28px; border-radius: 8px; font-weight: 600; margin: 20px 0; text-align: center; }}
        .token-box {{ background-color: #f1f5f9; border-radius: 8px; padding: 14px; word-break: break-all; font-family: monospace; font-size: 13px; color: #0f172a; margin: 15px 0; }}
        .footer {{ border-top: 1px solid #e9ecef; padding-top: 16px; font-size: 12px; color: #94a3b8; text-align: center; }}
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>📚 RAG Book Assistant</h1>
        </div>
        <div class="content">
          <p>Hello,</p>
          <p>We received a request to reset the password for your <strong>RAG Book Assistant</strong> account associated with <code>{to_email}</code>.</p>
          <p>Click the button below to reset your password. This link is valid for <strong>{JWT_RESET_EXPIRE_MINUTES} minutes</strong>:</p>
          <div style="text-align: center;">
            <a href="{reset_url}" class="btn" target="_blank">Reset My Password</a>
          </div>
          <p>Or paste this reset token into the password reset form:</p>
          <div class="token-box">{reset_token}</div>
          <p>If you did not request this password reset, please ignore this email.</p>
        </div>
        <div class="footer">
          <p>&copy; {datetime.now(timezone.utc).year} RAG Book Assistant. All rights reserved.</p>
        </div>
      </div>
    </body>
    </html>
    """

    try:
        resend.api_key = api_key
        sender = os.getenv("FROM_EMAIL", "onboarding@resend.dev").strip() or FROM_EMAIL
        params = {
            "from": sender,
            "to": [to_email],
            "subject": "🔑 Reset your RAG Book Assistant password",
            "html": html_content,
        }
        resend.Emails.send(params)
        return True, "Password reset email sent successfully."
    except Exception as e:
        return False, f"Failed to send email via Resend: {str(e)}"


# ======================================================
# MongoDB User Operations
# ======================================================

def init_user_db(db):
    """Ensure indexes exist on users collection for fast & unique lookups."""
    try:
        db["users"].create_index("username", unique=True)
        db["users"].create_index("email", unique=True)
    except PyMongoError:
        pass


def validate_username(username: str) -> tuple[bool, str]:
    username = username.strip()
    if not username:
        return False, "Username cannot be empty."
    if len(username) < 3:
        return False, "Username must be at least 3 characters long."
    if len(username) > 50:
        return False, "Username cannot exceed 50 characters."
    if not re.match(r"^[\w\.\-\ @\+]+$", username, re.UNICODE):
        return False, "Username can contain letters, numbers, spaces, dots, hyphens, and underscores."
    return True, ""


def validate_email(email: str) -> tuple[bool, str]:
    email = email.strip().lower()
    email_regex = r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$"
    if not re.match(email_regex, email):
        return False, "Please enter a valid email address."
    return True, ""


def validate_password(password: str) -> tuple[bool, str]:
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
    return True, ""


def register_user(db, username: str, email: str, password: str) -> tuple[bool, str]:
    """Register a new user in MongoDB."""
    init_user_db(db)

    clean_email = email.strip().lower()
    is_valid_e, err_e = validate_email(clean_email)
    if not is_valid_e:
        return False, err_e

    # If username is left empty, auto-default to email prefix
    clean_username = username.strip()
    if not clean_username and clean_email:
        clean_username = clean_email.split("@")[0]

    # Normalize whitespace
    clean_username = re.sub(r"\s+", " ", clean_username)

    is_valid_u, err_u = validate_username(clean_username)
    if not is_valid_u:
        return False, err_u

    is_valid_p, err_p = validate_password(password)
    if not is_valid_p:
        return False, err_p

    # Check for existing user
    if db["users"].find_one({"username": {"$regex": f"^{re.escape(clean_username)}$", "$options": "i"}}):
        return False, f"Username '{clean_username}' is already taken."

    if db["users"].find_one({"email": clean_email}):
        return False, f"An account with email '{clean_email}' already exists."


    hashed = hash_password(password)
    user_doc = {
        "_id": str(uuid.uuid4()),
        "username": clean_username,
        "email": clean_email,
        "password_hash": hashed,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }

    try:
        db["users"].insert_one(user_doc)
        return True, "Registration successful! You can now log in."
    except PyMongoError as e:
        return False, f"Database error during registration: {str(e)}"


def authenticate_user(db, identifier: str, password: str) -> tuple[bool, str, dict | None, str | None]:
    """
    Authenticate user by username or email.
    Returns: (success, message, user_dict, access_token)
    """
    init_user_db(db)
    identifier = identifier.strip()

    if not identifier or not password:
        return False, "Please provide both username/email and password.", None, None

    # Try email or username
    user = db["users"].find_one({
        "$or": [
            {"email": identifier.lower()},
            {"username": {"$regex": f"^{re.escape(identifier)}$", "$options": "i"}},
        ]
    })

    if not user:
        return False, "Invalid username/email or password.", None, None

    if not verify_password(password, user.get("password_hash", "")):
        return False, "Invalid username/email or password.", None, None

    token = create_access_token({
        "sub": user["_id"],
        "username": user["username"],
        "email": user["email"],
    })

    safe_user = {
        "id": user["_id"],
        "username": user["username"],
        "email": user["email"],
    }
    return True, "Login successful!", safe_user, token


def request_password_reset(db, email: str, base_url: str = APP_URL) -> tuple[bool, str, str | None, bool]:
    """
    Process a forgot password request.
    Generates a signed JWT reset token and dispatches an email.
    Returns: (success, message, reset_token, email_sent)
    """
    init_user_db(db)
    clean_email = email.strip().lower()

    is_valid_e, err_e = validate_email(clean_email)
    if not is_valid_e:
        return False, err_e, None, False

    user = db["users"].find_one({"email": clean_email})
    if not user:
        return False, f"No user account found with email '{clean_email}'.", None, False

    reset_token = create_reset_token(clean_email, expires_minutes=JWT_RESET_EXPIRE_MINUTES)
    email_sent, email_msg = send_reset_email(clean_email, reset_token, base_url=base_url)

    if email_sent:
        return True, f"Password reset instructions have been sent to {clean_email}.", reset_token, True
    else:
        return True, f"Reset token generated successfully. ({email_msg})", reset_token, False


def reset_password_with_token(db, reset_token: str, new_password: str) -> tuple[bool, str]:
    """Reset user password using a verified reset token."""
    init_user_db(db)

    token_data = verify_token(reset_token.strip(), expected_type="password_reset")
    if not token_data:
        return False, "Invalid or expired password reset token. Please request a new one."

    email = token_data.get("email")
    if not email:
        return False, "Invalid token payload (missing email)."

    is_valid_p, err_p = validate_password(new_password)
    if not is_valid_p:
        return False, err_p

    user = db["users"].find_one({"email": email})
    if not user:
        return False, "Account associated with this token no longer exists."

    new_hash = hash_password(new_password)

    try:
        db["users"].update_one(
            {"_id": user["_id"]},
            {
                "$set": {
                    "password_hash": new_hash,
                    "updated_at": datetime.now(timezone.utc),
                }
            }
        )
        return True, "Password updated successfully! You can now log in with your new password."
    except PyMongoError as e:
        return False, f"Database error while resetting password: {str(e)}"


# ======================================================
# Browser Cookie Management (Persistent Authentication)
# ======================================================

def get_cookie_controller() -> CookieController:
    """Retrieve or initialize CookieController in st.session_state."""
    if "cookie_controller" not in st.session_state:
        st.session_state.cookie_controller = CookieController(key="rag_cookies")
    return st.session_state.cookie_controller


def set_auth_cookie(token: str):
    """Store JWT in browser cookie for persistent authentication across page refreshes."""
    try:
        controller = get_cookie_controller()
        controller.set(
            AUTH_COOKIE_NAME,
            token,
            max_age=COOKIE_MAX_AGE,
            path="/",
            same_site="lax",
        )
    except Exception as e:
        print(f"Error setting auth cookie: {e}")


def delete_auth_cookie():
    """Delete JWT from browser cookie."""
    try:
        controller = get_cookie_controller()
        controller.remove(AUTH_COOKIE_NAME, path="/", same_site="lax")
    except Exception as e:
        print(f"Error deleting auth cookie: {e}")


# ======================================================
# Streamlit Auth UI & Gateway
# ======================================================

def logout():
    """Clear authentication state, delete browser cookie, and rerun the app."""
    delete_auth_cookie()
    time.sleep(0.3)
    for key in ["jwt_token", "auth_user", "user_id", "session_id", "messages", "cookies_checked"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


def render_user_sidebar(db, user: dict | None):
    """Render user profile and logout controls in the sidebar."""
    if not user:
        return
    with st.sidebar:

        st.subheader("👤 User Profile")
        col_avatar, col_info = st.columns([1, 3])
        with col_avatar:
            st.markdown("## 🧑‍💻")
        with col_info:
            st.markdown(f"**{user.get('username', 'User')}**")
            st.caption(f"📧 {user.get('email', '')}")

        if st.button("🚪 Log Out", use_container_width=True, type="secondary"):
            logout()


def render_auth_interface(db):
    """Render Login, Sign Up, and Forgot/Reset Password interface."""
    st.title("📚 RAG Book Assistant")
    st.markdown("Please sign in or create an account to start chatting with your documents.")

    # Check if URL query params contain a reset_token
    query_params = st.query_params
    url_reset_token = query_params.get("reset_token", "")

    tab_titles = ["🔑 Sign In", "📝 Sign Up", "🔄 Forgot Password", "🔐 Reset Password"]
    tabs = st.tabs(tab_titles)

    # --------------------------------------------------
    # TAB 1: Sign In
    # --------------------------------------------------
    with tabs[0]:
        st.subheader("Sign In to Your Account")
        with st.form("login_form", clear_on_submit=False):
            identifier = st.text_input("Username or Email", placeholder="e.g. john or john@example.com")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit_login = st.form_submit_button("Sign In", use_container_width=True, type="primary")

            if submit_login:
                success, msg, user, token = authenticate_user(db, identifier, password)
                if success:
                    st.session_state.jwt_token = token
                    st.session_state.auth_user = user
                    st.session_state.user_id = user["username"]
                    set_auth_cookie(token)
                    st.success("✅ Signed in successfully! Loading assistant...")
                    time.sleep(0.4)
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")


    # --------------------------------------------------
    # TAB 2: Sign Up
    # --------------------------------------------------
    with tabs[1]:
        st.subheader("Create a New Account")
        with st.form("register_form", clear_on_submit=False):
            reg_username = st.text_input(
                "Username",
                placeholder="e.g. adhya or your name (optional)",
                help="Letters, numbers, spaces, dots, hyphens, and underscores are allowed. If left blank, it defaults to your email.",
            )
            reg_email = st.text_input("Email", placeholder="you@example.com")
            reg_password = st.text_input("Password", type="password", placeholder="At least 6 characters")
            reg_confirm_password = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
            submit_register = st.form_submit_button("Create Account", use_container_width=True, type="primary")

            if submit_register:
                if reg_password != reg_confirm_password:
                    st.error("❌ Passwords do not match.")
                else:
                    success, msg = register_user(db, reg_username, reg_email, reg_password)
                    if success:
                        st.success(f"✅ {msg}")
                        st.info("👉 Switch to the **Sign In** tab to log into your new account.")
                    else:
                        st.error(f"❌ {msg}")

    # --------------------------------------------------
    # TAB 3: Forgot Password (Request Token)
    # --------------------------------------------------
    with tabs[2]:
        st.subheader("Forgot Your Password?")
        st.write("Enter your registered email address to receive a secure password reset link and token.")

        with st.form("forgot_password_form", clear_on_submit=False):
            forgot_email = st.text_input("Registered Email Address", placeholder="you@example.com")
            submit_forgot = st.form_submit_button("Send Reset Instructions", use_container_width=True, type="primary")

            if submit_forgot:
                success, msg, token, email_sent = request_password_reset(db, forgot_email, base_url=APP_URL)
                if success:
                    if email_sent:
                        st.success(f"✅ {msg}")
                        st.info(f"Check your inbox for the reset link, or use the token in the **Reset Password** tab.")
                    else:
                        st.success("✅ Password reset token generated!")
                        st.warning(f"ℹ️ {msg}")
                        st.info("You can copy the reset token below and paste it into the **Reset Password** tab:")
                        st.code(token, language="text")
                else:
                    st.error(f"❌ {msg}")

    # --------------------------------------------------
    # TAB 4: Reset Password (Execute Reset)
    # --------------------------------------------------
    with tabs[3]:
        st.subheader("Set a New Password")
        st.write("Paste your reset token and enter your new password below.")

        with st.form("reset_password_form", clear_on_submit=False):
            token_input = st.text_input(
                "Reset Token",
                value=url_reset_token,
                placeholder="Paste the reset token received via email or generated above",
            )
            new_password = st.text_input("New Password", type="password", placeholder="At least 6 characters")
            confirm_new_password = st.text_input("Confirm New Password", type="password", placeholder="Repeat new password")
            submit_reset = st.form_submit_button("Update Password", use_container_width=True, type="primary")

            if submit_reset:
                if not token_input:
                    st.error("❌ Please provide the reset token.")
                elif new_password != confirm_new_password:
                    st.error("❌ Passwords do not match.")
                else:
                    success, msg = reset_password_with_token(db, token_input, new_password)
                    if success:
                        st.success(f"✅ {msg}")
                        # Clean up URL param if present
                        if "reset_token" in st.query_params:
                            del st.query_params["reset_token"]
                        st.info("👉 Switch to the **Sign In** tab to log in with your new password.")
                    else:
                        st.error(f"❌ {msg}")


def require_auth(db) -> dict:
    """
    Enforce JWT authentication for the Streamlit application.
    Checks:
      1. st.session_state.jwt_token (fast path)
      2. Browser cookie (persistent path across page reloads/refreshes)
    - If valid: restores session state and returns authenticated user dict.
    - If unauthenticated: renders auth interface and stops further execution with st.stop().
    """
    controller = get_cookie_controller()

    # On page refresh / initial cold load, allow the browser component to synchronize cookies
    if "cookies_checked" not in st.session_state:
        st.session_state.cookies_checked = True
        cookie_tok = controller.get(AUTH_COOKIE_NAME)
        if not cookie_tok and "jwt_token" not in st.session_state:
            time.sleep(0.25)
            cookie_tok = controller.get(AUTH_COOKIE_NAME)

    # 1. Fast check: token in st.session_state
    token = st.session_state.get("jwt_token")

    # 2. Refresh persistence: restore from browser cookie if not in session_state
    if not token:
        token = controller.get(AUTH_COOKIE_NAME)

    if token:
        payload = verify_token(token, expected_type="access")
        if payload:
            user = db["users"].find_one({"_id": payload.get("sub")})
            if user:
                safe_user = {
                    "id": user["_id"],
                    "username": user["username"],
                    "email": user["email"],
                }
                st.session_state.jwt_token = token
                st.session_state.auth_user = safe_user
                st.session_state.user_id = safe_user["username"]
                return safe_user

        # Token was provided but is invalid or expired: clear stale cookie
        delete_auth_cookie()

    # Not authenticated or token expired/invalid
    st.session_state.jwt_token = None
    st.session_state.auth_user = None
    if "user_id" in st.session_state:
        del st.session_state["user_id"]

    render_auth_interface(db)
    st.stop()

