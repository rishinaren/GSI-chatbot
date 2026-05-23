"""Optional Amazon Cognito JWT authentication for the standards RAG API."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Any

try:
    import jwt
    from jwt import PyJWKClient
except ImportError:  # pragma: no cover - optional dependency
    jwt = None  # type: ignore[assignment]
    PyJWKClient = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class AuthConfig:
    required: bool
    user_pool_id: str | None
    app_client_id: str | None
    region: str | None

    @property
    def enabled(self) -> bool:
        return bool(self.user_pool_id and self.app_client_id and self.region)


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    email: str | None = None
    organization_id: str | None = None


def load_auth_config_from_env() -> AuthConfig:
    required_flag = os.getenv("AUTH_REQUIRED", "").strip().lower()
    required = required_flag in {"1", "true", "yes", "on"}
    return AuthConfig(
        required=required,
        user_pool_id=os.getenv("COGNITO_USER_POOL_ID", "").strip() or None,
        app_client_id=os.getenv("COGNITO_APP_CLIENT_ID", "").strip() or None,
        region=os.getenv("COGNITO_REGION", os.getenv("AWS_REGION", "")).strip() or None,
    )


def auth_public_config(config: AuthConfig | None = None) -> dict[str, Any]:
    config = config or load_auth_config_from_env()
    return {
        "auth_required": config.required and config.enabled,
        "cognito_user_pool_id": config.user_pool_id,
        "cognito_app_client_id": config.app_client_id,
        "cognito_region": config.region,
    }


class AuthError(Exception):
    def __init__(self, message: str, *, status_code: int = 401) -> None:
        super().__init__(message)
        self.status_code = status_code


class CognitoTokenValidator:
    def __init__(self, config: AuthConfig) -> None:
        if not config.enabled:
            raise RuntimeError("Cognito auth is not configured.")
        if jwt is None or PyJWKClient is None:
            raise RuntimeError("Install the optional 'auth' dependencies to validate Cognito tokens.")
        self.config = config
        issuer = f"https://cognito-idp.{config.region}.amazonaws.com/{config.user_pool_id}"
        self.issuer = issuer
        self.jwks_client = PyJWKClient(f"{issuer}/.well-known/jwks.json")

    def validate(self, token: str) -> AuthenticatedUser:
        try:
            signing_key = self.jwks_client.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self.config.app_client_id,
                issuer=self.issuer,
                options={"verify_at_hash": False},
            )
        except Exception as exc:  # noqa: BLE001 - surface as auth failure
            raise AuthError("Invalid or expired authentication token.") from exc

        token_use = str(claims.get("token_use", ""))
        if token_use not in {"access", "id"}:
            raise AuthError("Unsupported token type.")

        user_id = str(claims.get("sub", "")).strip()
        if not user_id:
            raise AuthError("Token missing subject claim.")

        email = claims.get("email")
        organization_id = claims.get("custom:organization_id") or claims.get("organization_id")
        return AuthenticatedUser(
            user_id=user_id,
            email=str(email) if email else None,
            organization_id=str(organization_id) if organization_id else None,
        )


def authenticate_bearer_token(
    authorization_header: str | None,
    *,
    config: AuthConfig | None = None,
) -> AuthenticatedUser | None:
    config = config or load_auth_config_from_env()
    if not authorization_header:
        if config.required and config.enabled:
            raise AuthError("Authentication required.")
        return None

    scheme, _, token = authorization_header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError("Authorization header must use Bearer token.")

    if not config.enabled:
        return AuthenticatedUser(user_id="local-dev-user", email="dev@local")

    validator = CognitoTokenValidator(config)
    return validator.validate(token.strip())


def login_with_password(email: str, password: str, *, config: AuthConfig | None = None) -> dict[str, str]:
    """Exchange username/password for Cognito tokens via the USER_PASSWORD_AUTH flow."""
    config = config or load_auth_config_from_env()
    if not config.enabled:
        raise AuthError("Cognito auth is not configured.", status_code=503)

    payload = {
        "AuthFlow": "USER_PASSWORD_AUTH",
        "ClientId": config.app_client_id,
        "AuthParameters": {
            "USERNAME": email,
            "PASSWORD": password,
        },
    }
    url = f"https://cognito-idp.{config.region}.amazonaws.com/"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/x-amz-json-1.1",
            "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise AuthError("Login failed. Check email and password.") from exc

    result = body.get("AuthenticationResult") or {}
    access_token = result.get("AccessToken")
    id_token = result.get("IdToken")
    refresh_token = result.get("RefreshToken")
    if not access_token or not id_token:
        raise AuthError("Login failed. Cognito did not return tokens.")

    return {
        "access_token": str(access_token),
        "id_token": str(id_token),
        "refresh_token": str(refresh_token or ""),
        "expires_in": str(result.get("ExpiresIn", 3600)),
        "token_type": str(result.get("TokenType", "Bearer")),
        "issued_at": str(int(time.time())),
    }
