"""Authentication helpers: PBKDF2 password hashing and JWT token handling.

Uses only stdlib hashing (``hashlib.pbkdf2_hmac``) for passwords and PyJWT for
signed tokens. The signing secret comes from the ``SPECTROGUARD_SECRET``
environment variable (a dev default is used if unset - fine for a prototype,
but set it in production).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time

import jwt

# Token signing secret.
SECRET_KEY = os.environ.get("SPECTROGUARD_SECRET", "spectroguard-dev-secret-change-me")
ALGORITHM = "HS256"
TOKEN_TTL_SECONDS = 7 * 24 * 3600  # 7 days


# --------------------------------------------------------------------------
# Passwords (PBKDF2-SHA256, salted)
# --------------------------------------------------------------------------
def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Hash a password. Returns ``(salt, hash_hex)``."""
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
    ).hex()
    return salt, digest


def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    """Constant-time comparison of a password against a stored hash."""
    _, digest = hash_password(password, salt)
    return hmac.compare_digest(digest, expected_hash)


# --------------------------------------------------------------------------
# Tokens (JWT)
# --------------------------------------------------------------------------
def create_token(user_id: str) -> str:
    """Issue a signed token containing the user id."""
    now = int(time.time())
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + TOKEN_TTL_SECONDS,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> str | None:
    """Validate a token and return the user id (or None if invalid/expired)."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]
    except jwt.PyJWTError:
        return None
