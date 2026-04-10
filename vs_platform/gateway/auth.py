"""
gateway/auth.py — Authentication
==================================
Two auth modes, selected per request by inspecting headers:

  Mode 1 — API Key (always available):
    X-API-Key: <key>
    Validated against SSM SecureString /clinical-agent/dev/platform/api_key.

  Mode 2 — Bearer JWT (Cognito, optional):
    Authorization: Bearer <jwt>
    Only active when Cognito SSM params are present.

Both modes return a unified AuthContext so downstream code is mode-agnostic.
"""

import hmac
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from core import aws

log = logging.getLogger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)


@dataclass
class AuthContext:
    user_id:   str
    tenant_id: str = "default"
    auth_mode: str = "api_key"
    scopes:    list = field(default_factory=list)


# ── API key auth ───────────────────────────────────────────────────────────────

def _validate_api_key(api_key: str) -> AuthContext:
    """
    Validate an API key against SSM SecureString (constant-time comparison).
    Raises HTTPException 401 on mismatch.
    """
    stored_key = aws.get_ssm_parameter(
        f"/clinical-agent/{aws.ENV}/platform/api_key",
        with_decryption=True,
    )

    key_matches = hmac.compare_digest(
        hashlib.sha256(api_key.encode()).digest(),
        hashlib.sha256(stored_key.encode()).digest(),
    )

    if not key_matches:
        log.warning("[AUTH] API key validation failed")
        raise HTTPException(status_code=401, detail="Invalid API key")

    log.debug("[AUTH] API key validated")
    return AuthContext(
        user_id="service-account",
        tenant_id="internal",
        auth_mode="api_key",
        scopes=["admin"],
    )


# ── JWT / Cognito auth ─────────────────────────────────────────────────────────

def _cognito_configured() -> bool:
    """Return True only if Cognito SSM params are present."""
    try:
        aws.get_ssm_parameter(
            f"/clinical-agent/{aws.ENV}/cognito/user_pool_id",
            with_decryption=False,
        )
        return True
    except Exception:
        return False


def _validate_jwt(token: str) -> AuthContext:
    """
    Validate a Cognito JWT and extract claims.
    Raises HTTPException 401 on invalid or expired token.
    """
    try:
        import jwt as pyjwt
        from jwt import PyJWKClient

        user_pool_id = aws.get_ssm_parameter(f"/clinical-agent/{aws.ENV}/cognito/user_pool_id", with_decryption=False)
        region       = aws.get_ssm_parameter(f"/clinical-agent/{aws.ENV}/cognito/region",        with_decryption=False)
        jwks_uri     = f"https://cognito-idp.{region}.amazonaws.com/{user_pool_id}/.well-known/jwks.json"
        issuer       = f"https://cognito-idp.{region}.amazonaws.com/{user_pool_id}"

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


# ── FastAPI dependency ─────────────────────────────────────────────────────────

async def require_auth(
    request:     Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
) -> AuthContext:
    """
    FastAPI dependency — validates the request and returns AuthContext.

    Priority:
      1. X-API-Key header → API key auth (always available)
      2. Authorization: Bearer <jwt> → Cognito JWT (only when Cognito is configured)
      3. Neither → 401 Unauthorized
    """
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return _validate_api_key(api_key)

    if credentials and credentials.scheme.lower() == "bearer":
        if _cognito_configured():
            return _validate_jwt(credentials.credentials)
        else:
            raise HTTPException(
                status_code=401,
                detail="JWT auth is not configured. Use X-API-Key header instead.",
            )

    raise HTTPException(
        status_code=401,
        detail="Authentication required. Provide X-API-Key header.",
    )