"""Membership + subscription auth for the GSI chatbot.

Two audiences share the chatbot, with deliberately different login flows:

* **GSI members** — people who already hold a broader Geosynthetic Institute
  membership. They are *already vetted*, so they sign in with their GSI
  credentials and get in with **no email code**. Their credentials are checked
  against the GSI member directory. That directory is not yet wired up, so this
  module ships a ``PlaceholderMemberDirectory`` behind a clean ``MemberDirectory``
  seam — swap in the real GSI backend later without touching the routes/UI.

* **Subscribers** — people paying for the chatbot on its own ($1,200/yr, 1-month
  free trial). They are *not* vetted through GSI, so their sign-in requires an
  **email one-time code (2FA)**. Their accounts are **real Cognito users**
  (``CognitoSubscriberAuth`` behind the ``SubscriberAuthProvider`` seam), so an
  existing Cognito login keeps working and new signups verify their email
  through Cognito. The per-login code is layered on via ``EmailCodeService``
  (shown on-screen in demo mode); once SES is out of sandbox it can move to
  native Cognito email MFA. Billing is a ``BillingProvider`` placeholder (swap
  for Stripe).

Everything is demo-functional today, with obvious seams for the real
integrations. Session tokens are stdlib-signed HS256 JWTs (no external dep) so
the same ``Authorization: Bearer`` plumbing the rest of the API already uses
keeps working — ``authenticate_bearer_token`` validates them alongside Cognito.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing (single tier; billed annually). Kept here so the API + UI stay in sync.
# ---------------------------------------------------------------------------

PLAN = {
    "id": "gsi-chatbot-annual",
    "name": "GSI Chatbot — Annual",
    "amount": 1200,
    "currency": "USD",
    "interval": "year",
    "trial_days": 30,
    "monthly_equivalent": 100,
    "features": [
        "Unlimited standards Q&A grounded in ASTM, ISO & GRI",
        "Page-bound citations with source links",
        "Video walkthroughs surfaced inline",
        "Saved chats and projects",
    ],
}


# ---------------------------------------------------------------------------
# Tiny stdlib HS256 JWT (avoids an extra dependency for demo/session tokens).
# ---------------------------------------------------------------------------


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


def _jwt_encode(payload: dict[str, Any], secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = (
        _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        + "."
        + _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    )
    signature = hmac.new(secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url(signature)}"


def _jwt_decode(token: str, secret: str) -> dict[str, Any] | None:
    """Return the verified payload, or ``None`` if the token is not a valid
    membership token (bad shape, bad signature, or expired)."""
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
    except ValueError:
        return None
    signing_input = f"{header_b64}.{payload_b64}"
    expected = hmac.new(secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    try:
        provided = _b64url_decode(signature_b64)
    except Exception:  # noqa: BLE001 - malformed base64 → not our token
        return None
    if not hmac.compare_digest(expected, provided):
        return None
    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(payload, dict):
        return None
    exp = payload.get("exp")
    if isinstance(exp, (int, float)) and time.time() > float(exp):
        return None
    return payload


class MembershipError(Exception):
    """User-facing membership/auth error carrying an HTTP status code."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Password hashing (stdlib pbkdf2). Placeholder-grade but not plaintext.
# ---------------------------------------------------------------------------


def _hash_password(password: str, *, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return f"pbkdf2_sha256$120000${_b64url(salt)}${_b64url(digest)}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algo, iterations, salt_b64, hash_b64 = encoded.split("$")
    except ValueError:
        return False
    if algo != "pbkdf2_sha256":
        return False
    try:
        salt = _b64url_decode(salt_b64)
        expected = _b64url_decode(hash_b64)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
    except Exception:  # noqa: BLE001
        return False
    return hmac.compare_digest(digest, expected)


# ---------------------------------------------------------------------------
# GSI member directory (already-vetted members; no 2FA). Swap seam.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemberRecord:
    email: str
    name: str | None = None
    organization_id: str | None = None


class MemberDirectory(Protocol):
    """Interface for validating GSI member credentials.

    The real implementation will call the GSI member backend. Anything that
    implements ``verify`` can be dropped in without touching the routes.
    """

    def verify(self, email: str, password: str) -> MemberRecord | None:  # pragma: no cover - protocol
        ...


class PlaceholderMemberDirectory:
    """In-memory stand-in for the real GSI member database.

    Seeded with demo members so the "already a GSI member" flow is fully
    demo-able before the real directory is connected. Replace with a
    ``MemberDirectory`` that queries GSI once credentials/API access exist.
    """

    def __init__(self, members: dict[str, dict[str, Any]] | None = None) -> None:
        # email -> {"password_hash": str, "name": str, "organization_id": str}
        self._members: dict[str, dict[str, Any]] = members or {}

    @classmethod
    def with_demo_seed(cls) -> PlaceholderMemberDirectory:
        directory = cls()
        directory.add("member@gsi.org", "member1234", name="Demo GSI Member", organization_id="gsi")
        return directory

    def add(self, email: str, password: str, *, name: str | None = None, organization_id: str | None = None) -> None:
        self._members[email.strip().lower()] = {
            "password_hash": _hash_password(password),
            "name": name,
            "organization_id": organization_id,
        }

    def verify(self, email: str, password: str) -> MemberRecord | None:
        record = self._members.get(email.strip().lower())
        if not record or not _verify_password(password, record["password_hash"]):
            return None
        return MemberRecord(
            email=email.strip().lower(),
            name=record.get("name"),
            organization_id=record.get("organization_id"),
        )


# ---------------------------------------------------------------------------
# Subscriber accounts (self-serve paying customers) — backed by real Cognito.
# ---------------------------------------------------------------------------


class SubscriberAuthProvider(Protocol):
    """Identity backing for subscribers. The default implementation is the real
    Cognito user pool; tests inject a fake. All methods raise ``MembershipError``
    on failure (mapped to a user-facing message + HTTP status)."""

    def verify_password(self, email: str, password: str) -> None:  # pragma: no cover - protocol
        ...

    def sign_up(self, email: str, password: str) -> dict[str, Any]:  # pragma: no cover - protocol
        ...

    def confirm(self, email: str, code: str) -> None:  # pragma: no cover - protocol
        ...

    def resend_signup_code(self, email: str) -> dict[str, Any]:  # pragma: no cover - protocol
        ...


class CognitoSubscriberAuth:
    """Backs subscribers with the real Cognito user pool.

    Subscribers sign in with their actual Cognito credentials (so an existing
    account keeps working), and new subscribers are registered + email-verified
    through Cognito. A per-login one-time code is layered on top (see
    ``EmailCodeService``); once SES is out of sandbox this can move to native
    Cognito email MFA. Cognito's own tokens are only used here to *validate* the
    password — the session itself is a membership token minted after the code.
    """

    def _err(self, exc: Exception, default_status: int) -> MembershipError:
        status = getattr(exc, "status_code", default_status)
        return MembershipError(str(exc) or "Authentication failed.", status_code=status)

    def verify_password(self, email: str, password: str) -> None:
        from standards_rag.auth import AuthError, login_with_password

        try:
            login_with_password(email, password)  # tokens discarded; validity is all we need
        except AuthError as exc:
            raise self._err(exc, 401) from exc

    def sign_up(self, email: str, password: str) -> dict[str, Any]:
        from standards_rag.auth import AuthError, sign_up_with_password

        try:
            return sign_up_with_password(email, password)
        except AuthError as exc:
            raise self._err(exc, 400) from exc

    def confirm(self, email: str, code: str) -> None:
        from standards_rag.auth import AuthError, confirm_sign_up

        try:
            confirm_sign_up(email, code)
        except AuthError as exc:
            raise self._err(exc, 400) from exc

    def resend_signup_code(self, email: str) -> dict[str, Any]:
        from standards_rag.auth import AuthError, resend_confirmation_code

        try:
            return resend_confirmation_code(email)
        except AuthError as exc:
            raise self._err(exc, 400) from exc


# ---------------------------------------------------------------------------
# Email one-time code (2FA for subscribers). Swap seam for real email.
# ---------------------------------------------------------------------------


class EmailCodeSender(Protocol):
    def send(self, email: str, code: str) -> None:  # pragma: no cover - protocol
        ...


class LoggingEmailCodeSender:
    """Fallback sender: logs the code instead of emailing it.

    Used when no real email transport is configured. When ``demo_mode`` is on the
    code is also returned in API responses so the flow can be demoed without a
    mailbox.
    """

    def send(self, email: str, code: str) -> None:
        logger.info("[membership] email code for %s: %s", email, code)


class SesEmailCodeSender:
    """Emails the login code via Amazon SES (real per-login 2FA delivery).

    Enabled by setting ``MEMBERSHIP_CODE_EMAIL_FROM`` to a verified SES sender.
    Note: while SES is in sandbox it only delivers to verified recipients — see
    HANDOFF §6.8 for the production-access path.
    """

    def __init__(self, from_address: str, *, region: str | None = None) -> None:
        self._from = from_address
        self._region = region or os.getenv("AWS_REGION") or "us-east-1"

    def send(self, email: str, code: str) -> None:
        import boto3  # lazy: only needed when SES delivery is configured

        client = boto3.client("ses", region_name=self._region)
        client.send_email(
            Source=self._from,
            Destination={"ToAddresses": [email]},
            Message={
                "Subject": {"Data": "Your GSI Chatbot login code"},
                "Body": {
                    "Text": {
                        "Data": (
                            f"Your GSI Chatbot login code is {code}.\n\n"
                            "It expires in 10 minutes. If you didn't try to sign in, ignore this email."
                        )
                    },
                    "Html": {
                        "Data": (
                            f"<p>Your GSI Chatbot login code is <strong style=\"font-size:1.3em;"
                            f"letter-spacing:2px\">{code}</strong>.</p>"
                            "<p>It expires in 10 minutes. If you didn't try to sign in, ignore this email.</p>"
                        )
                    },
                },
            },
        )


@dataclass
class _PendingCode:
    code: str
    expires_at: float
    attempts: int = 0


class EmailCodeService:
    """Generates, stores, and verifies short-lived 6-digit login codes."""

    def __init__(
        self,
        sender: EmailCodeSender,
        *,
        ttl_seconds: int = 600,
        max_attempts: int = 5,
        strict_delivery: bool = False,
    ) -> None:
        self._sender = sender
        self._ttl = ttl_seconds
        self._max_attempts = max_attempts
        # When strict (i.e. not demo mode), a send failure surfaces as an error so
        # the user isn't silently stranded with no code. In demo mode the code is
        # also returned on-screen, so a failed send is best-effort.
        self._strict = strict_delivery
        self._pending: dict[str, _PendingCode] = {}

    def issue(self, email: str) -> str:
        code = f"{secrets.randbelow(1_000_000):06d}"
        self._pending[email.strip().lower()] = _PendingCode(code=code, expires_at=time.time() + self._ttl)
        try:
            self._sender.send(email, code)
        except Exception as exc:  # noqa: BLE001 - transport errors are surfaced/logged
            logger.warning("Failed to send login code to %s: %s", email, exc)
            if self._strict:
                raise MembershipError(
                    "We couldn't send your login code right now. Please try again in a moment.",
                    status_code=502,
                ) from exc
        return code

    def verify(self, email: str, code: str) -> bool:
        key = email.strip().lower()
        pending = self._pending.get(key)
        if pending is None:
            return False
        if time.time() > pending.expires_at:
            self._pending.pop(key, None)
            return False
        pending.attempts += 1
        if pending.attempts > self._max_attempts:
            self._pending.pop(key, None)
            return False
        if not hmac.compare_digest(pending.code, code.strip()):
            return False
        self._pending.pop(key, None)
        return True


# ---------------------------------------------------------------------------
# Billing (subscription checkout). Placeholder for Stripe. Swap seam.
# ---------------------------------------------------------------------------


class BillingProvider(Protocol):
    def create_subscription(self, email: str, payment: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        ...


class PlaceholderBillingProvider:
    """Fakes a subscription checkout. No card is charged, no data is stored.

    Swap for a Stripe-backed provider (create customer + subscription with a
    trial) once billing is live. Card details are accepted but intentionally
    discarded here.
    """

    def create_subscription(self, email: str, payment: dict[str, Any]) -> dict[str, Any]:
        now = int(time.time())
        return {
            "subscription_id": f"sub_demo_{secrets.token_hex(6)}",
            "status": "trialing",
            "plan_id": PLAN["id"],
            "trial_ends_at": now + PLAN["trial_days"] * 86_400,
            "created_at": now,
        }


# ---------------------------------------------------------------------------
# Token service (session tokens accepted by the rest of the API).
# ---------------------------------------------------------------------------


DEFAULT_TOKEN_TTL_SECONDS = 30 * 86_400
_MEMBERSHIP_ISSUER = "gsi-membership"


@dataclass(frozen=True)
class MembershipPrincipal:
    user_id: str
    email: str
    account_type: str  # "member" | "subscriber"
    name: str | None = None
    organization_id: str | None = None


class MembershipTokenService:
    def __init__(self, secret: str, *, ttl_seconds: int = DEFAULT_TOKEN_TTL_SECONDS) -> None:
        self._secret = secret
        self._ttl = ttl_seconds

    def mint(self, principal: MembershipPrincipal) -> str:
        now = int(time.time())
        payload = {
            "iss": _MEMBERSHIP_ISSUER,
            "token_use": "membership",
            "sub": principal.user_id,
            "email": principal.email,
            "account_type": principal.account_type,
            "name": principal.name,
            "organization_id": principal.organization_id,
            "iat": now,
            "exp": now + self._ttl,
        }
        return _jwt_encode(payload, self._secret)

    def validate(self, token: str) -> MembershipPrincipal | None:
        payload = _jwt_decode(token, self._secret)
        if not payload or payload.get("iss") != _MEMBERSHIP_ISSUER:
            return None
        user_id = str(payload.get("sub") or "").strip()
        email = str(payload.get("email") or "").strip()
        account_type = str(payload.get("account_type") or "").strip()
        if not user_id or account_type not in {"member", "subscriber"}:
            return None
        return MembershipPrincipal(
            user_id=user_id,
            email=email,
            account_type=account_type,
            name=(str(payload["name"]) if payload.get("name") else None),
            organization_id=(str(payload["organization_id"]) if payload.get("organization_id") else None),
        )


def _user_id_for(account_type: str, email: str) -> str:
    return f"{account_type}:{email.strip().lower()}"


# ---------------------------------------------------------------------------
# Service orchestration — the object the API talks to.
# ---------------------------------------------------------------------------


class MembershipService:
    def __init__(
        self,
        *,
        member_directory: MemberDirectory,
        subscriber_auth: SubscriberAuthProvider,
        code_service: EmailCodeService,
        token_service: MembershipTokenService,
        billing: BillingProvider,
        demo_mode: bool = True,
    ) -> None:
        self.member_directory = member_directory
        self.subscriber_auth = subscriber_auth
        self.code_service = code_service
        self.token_service = token_service
        self.billing = billing
        self.demo_mode = demo_mode

    # -- config -----------------------------------------------------------
    def public_config(self) -> dict[str, Any]:
        return {
            "member_login_enabled": True,
            "subscriber_enabled": True,
            "demo_mode": self.demo_mode,
            "plan": PLAN,
        }

    # -- session validation (used by auth.py) -----------------------------
    def validate_token(self, token: str) -> MembershipPrincipal | None:
        return self.token_service.validate(token)

    # -- GSI member path (no 2FA) -----------------------------------------
    def member_login(self, email: str, password: str) -> dict[str, Any]:
        record = self.member_directory.verify(email, password)
        if record is None:
            raise MembershipError("Those GSI credentials didn't match. Please try again.", status_code=401)
        principal = MembershipPrincipal(
            user_id=_user_id_for("member", record.email),
            email=record.email,
            account_type="member",
            name=record.name,
            organization_id=record.organization_id,
        )
        return {
            "access_token": self.token_service.mint(principal),
            "account_type": "member",
            "email": principal.email,
            "name": principal.name,
        }

    # -- Subscriber path (email 2FA) --------------------------------------
    def _challenge_response(self, email: str) -> dict[str, Any]:
        code = self.code_service.issue(email)
        response: dict[str, Any] = {
            "challenge": "email_code",
            "email": email.strip().lower(),
            "destination": _mask_email(email),
        }
        if self.demo_mode:
            # Demo convenience only: lets the flow be shown without a mailbox.
            response["demo_code"] = code
        return response

    def _subscriber_session(self, email: str) -> dict[str, Any]:
        principal = MembershipPrincipal(
            user_id=_user_id_for("subscriber", email),
            email=email.strip().lower(),
            account_type="subscriber",
        )
        return {
            "access_token": self.token_service.mint(principal),
            "account_type": "subscriber",
            "email": principal.email,
        }

    def subscriber_login_start(self, email: str, password: str) -> dict[str, Any]:
        # Validate the password against real Cognito, then require a login code.
        self.subscriber_auth.verify_password(email, password)
        return self._challenge_response(email)

    def subscriber_resend_code(self, email: str) -> dict[str, Any]:
        # Re-issue the per-login code (the password was already validated).
        return self._challenge_response(email)

    def subscriber_verify(self, email: str, code: str) -> dict[str, Any]:
        if not self.code_service.verify(email, code):
            raise MembershipError("That code is incorrect or has expired. Request a new one.", status_code=401)
        return self._subscriber_session(email)

    # -- Subscription checkout (placeholder billing + real Cognito signup) -
    def subscribe(self, email: str, password: str, payment: dict[str, Any], *, name: str | None = None) -> dict[str, Any]:
        email = email.strip().lower()
        if not email or "@" not in email:
            raise MembershipError("Enter a valid email address.")
        if len(password) < 8:
            raise MembershipError("Choose a password with at least 8 characters.")
        # Placeholder checkout (no card charged), then register the Cognito user.
        billing = self.billing.create_subscription(email, payment)
        signup = self.subscriber_auth.sign_up(email, password)
        result: dict[str, Any] = {
            "subscription": {
                "status": str(billing.get("status") or "trialing"),
                "trial_ends_at": billing.get("trial_ends_at"),
                "plan": PLAN,
            },
            "signup": {
                "user_confirmed": bool(signup.get("user_confirmed")),
                "destination": str(signup.get("destination") or _mask_email(email)),
                "email": email,
            },
        }
        # Cognito normally requires an email confirmation code before first login.
        if signup.get("user_confirmed"):
            result.update(self._subscriber_session(email))
        return result

    def subscriber_confirm_signup(self, email: str, code: str) -> dict[str, Any]:
        # Confirm the Cognito email code from signup, then start the session.
        self.subscriber_auth.confirm(email, code)
        return self._subscriber_session(email)

    def subscriber_resend_signup_code(self, email: str) -> dict[str, Any]:
        return self.subscriber_auth.resend_signup_code(email)


def _mask_email(email: str) -> str:
    email = email.strip().lower()
    local, _, domain = email.partition("@")
    if not domain:
        return email
    shown = local[0] if local else ""
    return f"{shown}{'*' * max(len(local) - 1, 1)}@{domain}"


# ---------------------------------------------------------------------------
# Env-driven construction (the swap seams live here).
# ---------------------------------------------------------------------------


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _build_member_directory_from_env() -> MemberDirectory:
    # Seam: when the real GSI member API is available, branch on an env var
    # (e.g. GSI_MEMBER_API_URL) and return a directory that queries it.
    directory = PlaceholderMemberDirectory.with_demo_seed()
    raw = os.getenv("MEMBERSHIP_DEMO_MEMBERS", "").strip()
    if raw:
        try:
            for entry in json.loads(raw):
                directory.add(
                    entry["email"],
                    entry["password"],
                    name=entry.get("name"),
                    organization_id=entry.get("organization_id"),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ignoring invalid MEMBERSHIP_DEMO_MEMBERS: %s", exc)
    return directory


def _membership_secret() -> str:
    secret = os.getenv("MEMBERSHIP_JWT_SECRET", "").strip()
    if secret:
        return secret
    # Deterministic dev fallback so tokens survive a reload in local/demo runs.
    logger.warning("MEMBERSHIP_JWT_SECRET not set — using an insecure dev default.")
    return "gsi-membership-dev-secret-change-me"


def _build_code_service_from_env(demo_mode: bool) -> EmailCodeService:
    from_address = os.getenv("MEMBERSHIP_CODE_EMAIL_FROM", "").strip()
    if from_address:
        sender: EmailCodeSender = SesEmailCodeSender(from_address)
        logger.info("Membership login codes will be emailed via SES from %s", from_address)
    else:
        sender = LoggingEmailCodeSender()
    # Out of demo mode the on-screen code is hidden, so delivery must be reliable.
    return EmailCodeService(sender, strict_delivery=not demo_mode)


def build_membership_service_from_env() -> MembershipService:
    demo_mode = _bool_env("MEMBERSHIP_DEMO_MODE", True)
    return MembershipService(
        member_directory=_build_member_directory_from_env(),
        subscriber_auth=CognitoSubscriberAuth(),
        code_service=_build_code_service_from_env(demo_mode),
        token_service=MembershipTokenService(_membership_secret()),
        billing=PlaceholderBillingProvider(),
        demo_mode=demo_mode,
    )
