"""
gateway/auth.py — Authentication
==================================
Two auth modes, selected per request by inspecting the Authorization header:

  Mode 1 — Bearer JWT (Cognito):
    Authorization: Bearer <jwt>
    Validates signature against Cognito JWKS endpoint.
    Extracts sub (user_id) and custom:tenant_id from claims.
    Used by: external API consumers, web/mobile clients.

  Mode 2 — API Key:
    X-API-Key: <key>
    Key is stored in SSM as a SecureString and validated by hash comparison.
    Used by: internal service-to-service calls, CI pipelines.

Both modes return a unified AuthContext so downstream code is mode-agnostic.

WHY Cognito JWKS (not decode-and-trust):
  JWT signatures must be verified against the issuer's public key.
  Decoding without verification accepts any forged token.
  JWKS is fetched once and cached for the key lifetime — no per-request
  network call on the hot path.

WHY SSM for API keys (not env vars or headers alone):
  API keys are rotated in SSM without a deployment. Comparison uses
  hmac.compare_digest (constant-time) to prevent timing attacks.
"""

import hmac
import hashlib
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import httpx
from fastapi import HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from core import aws

log = logging.getLogger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)

# ── Auth context ───────────────────────────────────────────────────────────────

@dataclass
class AuthContext:
    """
    Unified auth result returned by both Cognito JWT and API key validation.
    Downstream code (rate limiter, agent router) uses this directly.
    """
    user_id:   str
    tenant_id: str = "default"
    auth_mode: str = "jwt"    # "jwt" | "api_key"
    scopes:    list[str] = None

    def __post_init__(self):
        if self.scopes is None:
            self.scopes = []


# ── JWKS cache ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_jwks(jwks_uri: str) -> dict:
    """
    Fetch and cache Cognito JWKS. Cached for the process lifetime.

    In production, rotate the cache when Cognito rotates keys by restarting
    the process, or add a TTL-based invalidation (not implemented here).
    Cache miss on first request adds ~100ms; all subsequent requests are free.
    """
    log.info(f"[AUTH] Fetching JWKS  uri={jwks_uri}")
    resp = httpx.get(jwks_uri, timeout=5.0)
    resp.raise_for_status()
    return resp.json()


def _validate_jwt(token: str) -> AuthContext:
    """
    Validate a Cognito JWT and extract claims.

    Reads the Cognito pool ID from SSM:
      /clinical-trial-agent/{env}/cognito/user_pool_id
      /clinical-trial-agent/{env}/cognito/region

    Raises HTTPException 401 on invalid or expired token.
    """
    try:
        import jwt as pyjwt  # pip install PyJWT[cryptography]
        from jwt import PyJWKClient

        env           = aws.get_env()
        user_pool_id  = aws.get_ssm_parameter(f"/clinical-trial-agent/{env}/cognito/user_pool_id", with_decryption=False)
        region        = aws.get_ssm_parameter(f"/clinical-trial-agent/{env}/cognito/region",        with_decryption=False)
        jwks_uri      = f"https://cognito-idp.{region}.amazonaws.com/{user_pool_id}/.well-known/jwks.json"
        issuer        = f"https://cognito-idp.{region}.amazonaws.com/{user_pool_id}"

        jwks_client = PyJWKClient(jwks_uri)
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        claims      = pyjwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=issuer,
            options={"verify_exp": True},
        )

        return AuthContext(
            user_id=claims.get("sub", "unknown"),
            tenant_id=claims.get("custom:tenant_id", "default"),
            auth_mode="jwt",
            scopes=claims.get("scope", "").split() if claims.get("scope") else [],
        )

    except Exception as exc:
        log.warning(f"[AUTH] JWT validation failed: {exc}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def _validate_api_key(api_key: str) -> AuthContext:
    """
    Validate an API key by comparing against the SSM-stored value.

    Uses hmac.compare_digest (constant-time) to prevent timing attacks
    where an attacker could determine correct key characters by measuring
    response latency of a naive string comparison.

    Reads from SSM:
      /clinical-trial-agent/{env}/platform/api_key   ← SecureString

    Raises HTTPException 401 on mismatch.
    """
    env             = aws.get_env()
    stored_key      = aws.get_ssm_parameter(f"/clinical-trial-agent/{env}/platform/api_key", with_decryption=True)

    # Constant-time comparison — prevents timing oracle attacks.
    key_matches = hmac.compare_digest(
        hashlib.sha256(api_key.encode()).digest(),
        hashlib.sha256(stored_key.encode()).digest(),
    )

    if not key_matches:
        log.warning("[AUTH] API key validation failed")
        raise HTTPException(status_code=401, detail="Invalid API key")

    # API key callers are treated as internal service accounts.
    return AuthContext(
        user_id="service-account",
        tenant_id="internal",
        auth_mode="api_key",
        scopes=["admin"],
    )


# ── FastAPI dependency ─────────────────────────────────────────────────────────

async def require_auth(
    request:     Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
) -> AuthContext:
    """
    FastAPI dependency — validates the request and returns AuthContext.

    Priority:
      1. X-API-Key header  → API key auth (internal services)
      2. Authorization: Bearer <jwt> → Cognito JWT auth (external clients)
      3. Neither → 401 Unauthorized

    Inject in route handlers:
      @router.post("/chat")
      async def chat(auth: AuthContext = Depends(require_auth)):
          ...
    """
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return _validate_api_key(api_key)

    if credentials and credentials.scheme.lower() == "bearer":
        return _validate_jwt(credentials.credentials)

    raise HTTPException(
        status_code=401,
        detail="Authentication required. Provide Authorization: Bearer <jwt> or X-API-Key: <key>",
    )
