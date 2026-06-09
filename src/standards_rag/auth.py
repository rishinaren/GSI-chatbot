"""Optional Amazon Cognito JWT authentication for the standards RAG API."""

from __future__ import annotations

import json
import os
import time
import urllib.error
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
    enabled = config.required and config.enabled
    return {
        "auth_required": enabled,
        "signup_enabled": enabled,
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
                issuer=self.issuer,
                options={"verify_at_hash": False, "verify_aud": False},
            )
        except Exception as exc:  # noqa: BLE001 - surface as auth failure
            raise AuthError("Invalid or expired authentication token.") from exc

        token_use = str(claims.get("token_use", ""))
        if token_use not in {"access", "id"}:
            raise AuthError("Unsupported token type.")
        if token_use == "id":
            if str(claims.get("aud", "")).strip() != str(self.config.app_client_id):
                raise AuthError("Invalid authentication token audience.")
        if token_use == "access":
            if str(claims.get("client_id", "")).strip() != str(self.config.app_client_id):
                raise AuthError("Invalid authentication token client.")

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


def _map_cognito_error_message(raw_message: str, *, error_type: str = "") -> str:
    message = raw_message.strip()
    error_name = error_type.rsplit("#", maxsplit=1)[-1] if error_type else ""
    lowered = message.lower()

    if error_name == "UsernameExistsException" or "usernameexistsexception" in lowered:
        return "An account with this email already exists. Try signing in instead."
    if error_name == "InvalidPasswordException" or "invalidpasswordexception" in lowered:
        return (
            "Password does not meet requirements. Use at least 8 characters with uppercase, "
            "lowercase, numbers, and symbols."
        )
    if error_name == "InvalidParameterException" and "password" in lowered:
        return (
            "Password does not meet requirements. Use at least 8 characters with uppercase, "
            "lowercase, numbers, and symbols."
        )
    if error_name == "CodeMismatchException" or "codemismatchexception" in lowered:
        return "That verification code is incorrect. Please try again."
    if error_name == "ExpiredCodeException" or "expiredcodeexception" in lowered:
        return "That verification code has expired. Request a new code and try again."
    if error_name == "UserNotConfirmedException" or "usernotconfirmedexception" in lowered:
        return "Confirm your email with the verification code before signing in."
    if error_name == "NotAuthorizedException" or "notauthorizedexception" in lowered:
        if "incorrect username or password" in lowered:
            return "That email or password is incorrect. Please try again."
        return "Login failed. Check email and password."
    if message:
        return message
    return "Authentication request failed."


def _call_cognito_api(
    target: str,
    payload: dict[str, Any],
    *,
    config: AuthConfig,
    default_error: str,
    default_status: int = 400,
) -> dict[str, Any]:
    url = f"https://cognito-idp.{config.region}.amazonaws.com/"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/x-amz-json-1.1",
            "X-Amz-Target": target,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {}
        message = _map_cognito_error_message(
            str(body.get("message") or default_error),
            error_type=str(body.get("__type") or ""),
        )
        raise AuthError(message, status_code=exc.code or default_status) from exc
    except Exception as exc:  # noqa: BLE001
        raise AuthError(default_error, status_code=default_status) from exc


def _token_response(result: dict[str, Any]) -> dict[str, str]:
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


def login_with_password(email: str, password: str, *, config: AuthConfig | None = None) -> dict[str, str]:
    """Exchange username/password for Cognito tokens via the USER_PASSWORD_AUTH flow."""
    config = config or load_auth_config_from_env()
    if not config.enabled:
        raise AuthError("Cognito auth is not configured.", status_code=503)

    body = _call_cognito_api(
        "AWSCognitoIdentityProviderService.InitiateAuth",
        {
            "AuthFlow": "USER_PASSWORD_AUTH",
            "ClientId": config.app_client_id,
            "AuthParameters": {
                "USERNAME": email,
                "PASSWORD": password,
            },
        },
        config=config,
        default_error="Login failed. Check email and password.",
        default_status=401,
    )
    return _token_response(body.get("AuthenticationResult") or {})


def sign_up_with_password(email: str, password: str, *, config: AuthConfig | None = None) -> dict[str, Any]:
    """Register a new Cognito user with email and password."""
    config = config or load_auth_config_from_env()
    if not config.enabled:
        raise AuthError("Cognito auth is not configured.", status_code=503)

    body = _call_cognito_api(
        "AWSCognitoIdentityProviderService.SignUp",
        {
            "ClientId": config.app_client_id,
            "Username": email,
            "Password": password,
            "UserAttributes": [{"Name": "email", "Value": email}],
        },
        config=config,
        default_error="Sign up failed. Check your email and password.",
    )
    delivery = body.get("CodeDeliveryDetails") or {}
    return {
        "user_confirmed": bool(body.get("UserConfirmed")),
        "user_sub": str(body.get("UserSub") or ""),
        "delivery_medium": str(delivery.get("DeliveryMedium") or ""),
        "destination": str(delivery.get("Destination") or ""),
    }


def confirm_sign_up(email: str, confirmation_code: str, *, config: AuthConfig | None = None) -> dict[str, bool]:
    """Confirm a newly registered Cognito user with an email verification code."""
    config = config or load_auth_config_from_env()
    if not config.enabled:
        raise AuthError("Cognito auth is not configured.", status_code=503)

    _call_cognito_api(
        "AWSCognitoIdentityProviderService.ConfirmSignUp",
        {
            "ClientId": config.app_client_id,
            "Username": email,
            "ConfirmationCode": confirmation_code.strip(),
        },
        config=config,
        default_error="Email confirmation failed. Check the verification code.",
    )
    return {"confirmed": True}


def resend_confirmation_code(email: str, *, config: AuthConfig | None = None) -> dict[str, str]:
    """Resend the Cognito email verification code for an unconfirmed user."""
    config = config or load_auth_config_from_env()
    if not config.enabled:
        raise AuthError("Cognito auth is not configured.", status_code=503)

    body = _call_cognito_api(
        "AWSCognitoIdentityProviderService.ResendConfirmationCode",
        {
            "ClientId": config.app_client_id,
            "Username": email,
        },
        config=config,
        default_error="Could not resend the verification code.",
    )
    delivery = body.get("CodeDeliveryDetails") or {}
    return {
        "delivery_medium": str(delivery.get("DeliveryMedium") or ""),
        "destination": str(delivery.get("Destination") or ""),
    }
