"""Who may manage the document library, and how that list is changed.

Access has two layers, deliberately:

* **Root admins** come from the ``GSI_ADMIN_EMAILS`` environment variable. They
  cannot be removed through the UI, so a mistaken revoke - or an empty table -
  can never lock everyone out of the library. Changing them is a deploy.
* **Granted admins** are rows an existing admin adds from inside the app, stored
  in the same DynamoDB table as conversations and projects (a fixed partition,
  the way projects reuse a prefixed sort key). These can be added and removed at
  runtime, with no redeploy.

Someone is an admin if they are in *either* layer.

Granting is deliberately not account creation: an email is only accepted once we
can confirm it already belongs to a subscriber (Cognito) or a GSI member (the
member directory). Otherwise a typo would silently create a dead entry that
looks like access was given.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# A fixed partition key for the shared conversations table. Real user ids are
# "member:<email>" / "subscriber:<email>" / a Cognito sub, so this cannot collide.
ADMIN_PARTITION = "__admins__"
ADMIN_SORT_KEY_PREFIX = "ADMIN#"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_email(email: str | None) -> str:
    return (email or "").strip().lower()


class AdminAccessError(Exception):
    """A problem worth showing to the admin doing the granting, in their words."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class AdminRecord:
    email: str
    granted_by: str = ""
    granted_at: str = field(default_factory=_utc_now_iso)

    def to_item(self) -> dict[str, Any]:
        return {
            "user_id": ADMIN_PARTITION,
            "conversation_id": f"{ADMIN_SORT_KEY_PREFIX}{self.email}",
            "record_type": "admin",
            "email": self.email,
            "granted_by": self.granted_by,
            "granted_at": self.granted_at,
        }

    @classmethod
    def from_item(cls, data: dict[str, Any]) -> "AdminRecord":
        return cls(
            email=normalize_email(str(data.get("email", ""))),
            granted_by=str(data.get("granted_by") or ""),
            granted_at=str(data.get("granted_at") or _utc_now_iso()),
        )


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class AdminStore(Protocol):
    def list_admins(self) -> list[AdminRecord]:  # pragma: no cover - protocol
        ...

    def add_admin(self, record: AdminRecord) -> None:  # pragma: no cover - protocol
        ...

    def remove_admin(self, email: str) -> bool:  # pragma: no cover - protocol
        ...


class InMemoryAdminStore:
    """Local/dev backing. Grants live only as long as the process."""

    def __init__(self) -> None:
        self._admins: dict[str, AdminRecord] = {}

    def list_admins(self) -> list[AdminRecord]:
        return sorted(self._admins.values(), key=lambda record: record.email)

    def add_admin(self, record: AdminRecord) -> None:
        self._admins[record.email] = record

    def remove_admin(self, email: str) -> bool:
        return self._admins.pop(normalize_email(email), None) is not None


class DynamoDBAdminStore:
    """Grants in the shared conversations table under a fixed partition.

    Reuses the existing table so no new infrastructure (or IAM resource) is
    needed - the runtime role is already scoped to exactly this table.
    """

    def __init__(self, *, table_name: str, region: str | None = None) -> None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the optional 'aws' dependencies for DynamoDB storage.") from exc

        self.table_name = table_name
        self._resource = boto3.resource("dynamodb", region_name=region or os.getenv("AWS_REGION"))
        self._table = self._resource.Table(table_name)

    def list_admins(self) -> list[AdminRecord]:
        from boto3.dynamodb.conditions import Key

        response = self._table.query(KeyConditionExpression=Key("user_id").eq(ADMIN_PARTITION))
        records = [AdminRecord.from_item(item) for item in response.get("Items", [])]
        return sorted((record for record in records if record.email), key=lambda r: r.email)

    def add_admin(self, record: AdminRecord) -> None:
        self._table.put_item(Item=record.to_item())

    def remove_admin(self, email: str) -> bool:
        response = self._table.delete_item(
            Key={
                "user_id": ADMIN_PARTITION,
                "conversation_id": f"{ADMIN_SORT_KEY_PREFIX}{normalize_email(email)}",
            },
            ReturnValues="ALL_OLD",
        )
        return bool(response.get("Attributes"))


def build_admin_store_from_env() -> AdminStore:
    table_name = os.getenv("DYNAMODB_CONVERSATIONS_TABLE", "").strip()
    if table_name:
        return DynamoDBAdminStore(table_name=table_name, region=os.getenv("AWS_REGION"))
    return InMemoryAdminStore()


# ---------------------------------------------------------------------------
# "Does this person already have an account?"
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AccountLookup:
    """Outcome of checking an email against the identity backends.

    ``found`` and ``checkable`` are separate on purpose: "we looked and there is
    no such account" and "we could not look" must not produce the same message,
    or a misconfiguration would read to the admin as the person not existing.
    """

    found: bool
    account_type: str | None = None
    checkable: bool = True
    detail: str = ""


class AccountDirectory:
    """Answers whether an email belongs to an existing subscriber or GSI member."""

    def __init__(self, member_directory: Any | None = None) -> None:
        self._member_directory = member_directory

    def find(self, email: str) -> AccountLookup:
        address = normalize_email(email)
        if not address or "@" not in address:
            raise AdminAccessError("That does not look like an email address.")

        member = self._member_lookup(address)
        if member.found:
            return member

        subscriber = self._cognito_lookup(address)
        if subscriber.found:
            return subscriber

        # Neither backend found them. Only report "no account" if we could
        # actually check; otherwise say what stopped us.
        if not member.checkable and not subscriber.checkable:
            return AccountLookup(
                found=False,
                checkable=False,
                detail=subscriber.detail or member.detail,
            )
        return AccountLookup(found=False)

    def _member_lookup(self, email: str) -> AccountLookup:
        directory = self._member_directory
        exists = getattr(directory, "exists", None) if directory is not None else None
        if not callable(exists):
            # The real GSI member directory is not wired up yet (see HANDOFF §6.8a),
            # so member accounts simply cannot be confirmed from here.
            return AccountLookup(found=False, checkable=False, detail="member directory unavailable")
        try:
            return AccountLookup(found=bool(exists(email)), account_type="member")
        except Exception as exc:  # noqa: BLE001 - a directory outage is not "no account"
            logger.warning("Member directory lookup failed for %s: %s", email, exc)
            return AccountLookup(found=False, checkable=False, detail=str(exc))

    def _cognito_lookup(self, email: str) -> AccountLookup:
        from standards_rag.auth import load_auth_config_from_env

        config = load_auth_config_from_env()
        if not config.user_pool_id or not config.region:
            return AccountLookup(found=False, checkable=False, detail="Cognito is not configured")

        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:  # pragma: no cover - only when the aws extra is missing
            return AccountLookup(found=False, checkable=False, detail="AWS SDK unavailable")

        client = boto3.client("cognito-idp", region_name=config.region)
        try:
            client.admin_get_user(UserPoolId=config.user_pool_id, Username=email)
            return AccountLookup(found=True, account_type="subscriber")
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code == "UserNotFoundException":
                return AccountLookup(found=False, account_type="subscriber")
            # AccessDenied, throttling, pool misconfig - we did not get an answer.
            logger.warning("Cognito lookup failed for %s: %s", email, code or exc)
            return AccountLookup(found=False, checkable=False, detail=code or str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cognito lookup errored for %s: %s", email, exc)
            return AccountLookup(found=False, checkable=False, detail=str(exc))


# ---------------------------------------------------------------------------
# The registry the API talks to
# ---------------------------------------------------------------------------


def root_admin_emails() -> set[str]:
    raw = os.getenv("GSI_ADMIN_EMAILS", "")
    return {normalize_email(part) for part in raw.split(",") if part.strip()}


class AdminRegistry:
    """Root allowlist + runtime grants, and the rules for changing the latter."""

    def __init__(self, store: AdminStore, *, accounts: AccountDirectory | None = None) -> None:
        self._store = store
        self._accounts = accounts or AccountDirectory()

    # -- reading ----------------------------------------------------------

    def is_admin(self, email: str | None) -> bool:
        address = normalize_email(email)
        if not address:
            return False
        if address in root_admin_emails():
            return True
        try:
            return any(record.email == address for record in self._store.list_admins())
        except Exception as exc:  # noqa: BLE001 - a storage outage must not grant access
            logger.warning("Could not read granted admins: %s", exc)
            return False

    def people(self) -> list[dict[str, Any]]:
        """Everyone who can manage the library, roots first."""
        roots = sorted(root_admin_emails())
        rows: list[dict[str, Any]] = [
            {"email": email, "source": "root", "removable": False, "granted_by": "", "granted_at": ""}
            for email in roots
        ]
        for record in self._store.list_admins():
            if record.email in root_admin_emails():
                continue  # promoted to root since; the root row already covers them
            rows.append(
                {
                    "email": record.email,
                    "source": "granted",
                    "removable": True,
                    "granted_by": record.granted_by,
                    "granted_at": record.granted_at,
                }
            )
        return rows

    # -- writing ----------------------------------------------------------

    def grant(self, email: str, *, granted_by: str) -> dict[str, Any]:
        address = normalize_email(email)
        if not address or "@" not in address:
            raise AdminAccessError("That does not look like an email address.")
        if address in root_admin_emails():
            raise AdminAccessError(f"{address} already manages the library.")
        if any(record.email == address for record in self._store.list_admins()):
            raise AdminAccessError(f"{address} already manages the library.")

        lookup = self._accounts.find(address)
        if not lookup.found:
            if not lookup.checkable:
                raise AdminAccessError(
                    "We could not check whether that email has an account, so we have not "
                    "given anyone access. Please try again in a moment.",
                    status_code=503,
                )
            raise AdminAccessError(
                f"We could not find an account for {address}. They need to sign in to the "
                "chatbot at least once before you can give them access."
            )

        record = AdminRecord(email=address, granted_by=normalize_email(granted_by))
        self._store.add_admin(record)
        return {
            "email": record.email,
            "account_type": lookup.account_type,
            "granted_by": record.granted_by,
            "granted_at": record.granted_at,
        }

    def revoke(self, email: str, *, revoked_by: str) -> None:
        address = normalize_email(email)
        if address in root_admin_emails():
            raise AdminAccessError(
                f"{address} is set in the service configuration and cannot be removed here."
            )
        if address == normalize_email(revoked_by):
            raise AdminAccessError("You cannot remove your own access.")
        if not self._store.remove_admin(address):
            raise AdminAccessError(f"{address} does not manage the library.", status_code=404)


def build_admin_registry_from_env(member_directory: Any | None = None) -> AdminRegistry:
    return AdminRegistry(
        build_admin_store_from_env(),
        accounts=AccountDirectory(member_directory=member_directory),
    )
