from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from standards_rag.admin_access import (
    AccountDirectory,
    AccountLookup,
    AdminAccessError,
    AdminRecord,
    AdminRegistry,
    InMemoryAdminStore,
)
from standards_rag.membership import PlaceholderMemberDirectory


class FakeAccounts(AccountDirectory):
    """Stands in for Cognito + the member directory."""

    def __init__(self, known: set[str], *, checkable: bool = True) -> None:
        super().__init__()
        self._known = known
        self._checkable = checkable

    def find(self, email: str) -> AccountLookup:
        address = email.strip().lower()
        if not self._checkable:
            return AccountLookup(found=False, checkable=False, detail="Cognito is not configured")
        if address in self._known:
            return AccountLookup(found=True, account_type="subscriber")
        return AccountLookup(found=False)


def _registry(known: set[str] | None = None, *, checkable: bool = True) -> AdminRegistry:
    return AdminRegistry(
        InMemoryAdminStore(),
        accounts=FakeAccounts(known or set(), checkable=checkable),
    )


class RootAdminTests(unittest.TestCase):
    @mock.patch.dict("os.environ", {"GSI_ADMIN_EMAILS": "Root@GSI.org, second@gsi.org"})
    def test_root_admins_are_matched_case_insensitively(self) -> None:
        registry = _registry()
        self.assertTrue(registry.is_admin("root@gsi.org"))
        self.assertTrue(registry.is_admin("  SECOND@gsi.org "))
        self.assertFalse(registry.is_admin("nobody@gsi.org"))

    @mock.patch.dict("os.environ", {"GSI_ADMIN_EMAILS": ""})
    def test_no_configured_admins_means_nobody_is_one(self) -> None:
        registry = _registry()
        self.assertFalse(registry.is_admin("anyone@gsi.org"))
        self.assertFalse(registry.is_admin(""))
        self.assertFalse(registry.is_admin(None))

    @mock.patch.dict("os.environ", {"GSI_ADMIN_EMAILS": "root@gsi.org"})
    def test_a_root_admin_cannot_be_removed_through_the_api(self) -> None:
        registry = _registry()
        with self.assertRaises(AdminAccessError) as caught:
            registry.revoke("root@gsi.org", revoked_by="other@gsi.org")
        self.assertIn("service configuration", str(caught.exception))
        self.assertTrue(registry.is_admin("root@gsi.org"))


class GrantTests(unittest.TestCase):
    @mock.patch.dict("os.environ", {"GSI_ADMIN_EMAILS": "root@gsi.org"})
    def test_granting_an_existing_account_gives_access(self) -> None:
        registry = _registry({"colleague@gsi.org"})
        self.assertFalse(registry.is_admin("colleague@gsi.org"))

        granted = registry.grant("Colleague@GSI.org", granted_by="root@gsi.org")
        self.assertEqual(granted["email"], "colleague@gsi.org")
        self.assertEqual(granted["granted_by"], "root@gsi.org")
        self.assertTrue(registry.is_admin("colleague@gsi.org"))

    @mock.patch.dict("os.environ", {"GSI_ADMIN_EMAILS": "root@gsi.org"})
    def test_an_email_without_an_account_is_refused(self) -> None:
        registry = _registry({"colleague@gsi.org"})
        with self.assertRaises(AdminAccessError) as caught:
            registry.grant("stranger@nowhere.com", granted_by="root@gsi.org")
        self.assertIn("could not find an account", str(caught.exception))
        self.assertFalse(registry.is_admin("stranger@nowhere.com"))

    @mock.patch.dict("os.environ", {"GSI_ADMIN_EMAILS": "root@gsi.org"})
    def test_a_failed_lookup_is_not_reported_as_a_missing_account(self) -> None:
        """"We could not check" must never read as "they do not exist"."""
        registry = _registry({"colleague@gsi.org"}, checkable=False)
        with self.assertRaises(AdminAccessError) as caught:
            registry.grant("colleague@gsi.org", granted_by="root@gsi.org")
        self.assertEqual(caught.exception.status_code, 503)
        self.assertIn("could not check", str(caught.exception))

    @mock.patch.dict("os.environ", {"GSI_ADMIN_EMAILS": "root@gsi.org"})
    def test_junk_input_is_refused(self) -> None:
        registry = _registry({"colleague@gsi.org"})
        for value in ("", "   ", "not-an-email"):
            with self.assertRaises(AdminAccessError):
                registry.grant(value, granted_by="root@gsi.org")

    @mock.patch.dict("os.environ", {"GSI_ADMIN_EMAILS": "root@gsi.org"})
    def test_granting_twice_is_refused_clearly(self) -> None:
        registry = _registry({"colleague@gsi.org"})
        registry.grant("colleague@gsi.org", granted_by="root@gsi.org")
        with self.assertRaises(AdminAccessError) as caught:
            registry.grant("colleague@gsi.org", granted_by="root@gsi.org")
        self.assertIn("already manages", str(caught.exception))


class RevokeTests(unittest.TestCase):
    @mock.patch.dict("os.environ", {"GSI_ADMIN_EMAILS": "root@gsi.org"})
    def test_revoking_removes_access(self) -> None:
        registry = _registry({"colleague@gsi.org"})
        registry.grant("colleague@gsi.org", granted_by="root@gsi.org")
        registry.revoke("colleague@gsi.org", revoked_by="root@gsi.org")
        self.assertFalse(registry.is_admin("colleague@gsi.org"))

    @mock.patch.dict("os.environ", {"GSI_ADMIN_EMAILS": "root@gsi.org"})
    def test_you_cannot_revoke_yourself(self) -> None:
        """Guards the one-granted-admin case from locking themselves out."""
        registry = _registry({"colleague@gsi.org"})
        registry.grant("colleague@gsi.org", granted_by="root@gsi.org")
        with self.assertRaises(AdminAccessError) as caught:
            registry.revoke("colleague@gsi.org", revoked_by="Colleague@GSI.org")
        self.assertIn("your own access", str(caught.exception))
        self.assertTrue(registry.is_admin("colleague@gsi.org"))

    @mock.patch.dict("os.environ", {"GSI_ADMIN_EMAILS": "root@gsi.org"})
    def test_revoking_someone_who_has_no_access_is_a_404(self) -> None:
        registry = _registry({"colleague@gsi.org"})
        with self.assertRaises(AdminAccessError) as caught:
            registry.revoke("colleague@gsi.org", revoked_by="root@gsi.org")
        self.assertEqual(caught.exception.status_code, 404)


class ListingTests(unittest.TestCase):
    @mock.patch.dict("os.environ", {"GSI_ADMIN_EMAILS": "root@gsi.org"})
    def test_people_marks_which_rows_can_be_removed(self) -> None:
        registry = _registry({"colleague@gsi.org"})
        registry.grant("colleague@gsi.org", granted_by="root@gsi.org")
        rows = {row["email"]: row for row in registry.people()}

        self.assertEqual(rows["root@gsi.org"]["source"], "root")
        self.assertFalse(rows["root@gsi.org"]["removable"])
        self.assertEqual(rows["colleague@gsi.org"]["source"], "granted")
        self.assertTrue(rows["colleague@gsi.org"]["removable"])

    @mock.patch.dict("os.environ", {"GSI_ADMIN_EMAILS": "colleague@gsi.org"})
    def test_someone_promoted_to_root_is_listed_once(self) -> None:
        store = InMemoryAdminStore()
        store.add_admin(AdminRecord(email="colleague@gsi.org", granted_by="root@gsi.org"))
        registry = AdminRegistry(store, accounts=FakeAccounts(set()))
        rows = registry.people()
        self.assertEqual([row["email"] for row in rows], ["colleague@gsi.org"])
        self.assertEqual(rows[0]["source"], "root")

    @mock.patch.dict("os.environ", {"GSI_ADMIN_EMAILS": "root@gsi.org"})
    def test_a_storage_outage_denies_rather_than_grants(self) -> None:
        class BrokenStore(InMemoryAdminStore):
            def list_admins(self):
                raise RuntimeError("DynamoDB unavailable")

        registry = AdminRegistry(BrokenStore(), accounts=FakeAccounts(set()))
        self.assertFalse(registry.is_admin("colleague@gsi.org"))
        self.assertTrue(registry.is_admin("root@gsi.org"))  # root still works


class MemberDirectoryLookupTests(unittest.TestCase):
    def test_the_placeholder_directory_can_confirm_a_member_exists(self) -> None:
        directory = PlaceholderMemberDirectory.with_demo_seed()
        self.assertTrue(directory.exists("member@gsi.org"))
        self.assertTrue(directory.exists("  MEMBER@GSI.ORG "))
        self.assertFalse(directory.exists("someone@else.com"))

    def test_a_directory_without_exists_is_reported_as_uncheckable(self) -> None:
        """The real GSI directory is not wired up yet; that is not "no account"."""

        class LegacyDirectory:
            def verify(self, email, password):
                return None

        result = AccountDirectory(member_directory=LegacyDirectory())._member_lookup("a@b.com")
        self.assertFalse(result.found)
        self.assertFalse(result.checkable)


if __name__ == "__main__":
    unittest.main()
