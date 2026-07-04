import unittest

from standards_rag.membership import (
    EmailCodeService,
    LoggingEmailCodeSender,
    MembershipError,
    MembershipService,
    MembershipTokenService,
    PlaceholderBillingProvider,
    PlaceholderMemberDirectory,
    _jwt_decode,
    _jwt_encode,
)


class FakeSubscriberAuth:
    """Stands in for Cognito in tests: an in-memory account book."""

    def __init__(self) -> None:
        self.accounts: dict[str, str] = {"subscriber@gsi.org": "subscriber1234"}
        self.confirmed: set[str] = {"subscriber@gsi.org"}

    def verify_password(self, email: str, password: str) -> None:
        email = email.lower()
        if self.accounts.get(email) != password:
            raise MembershipError("That email or password is incorrect.", status_code=401)
        if email not in self.confirmed:
            raise MembershipError("Confirm your email first.", status_code=400)

    def sign_up(self, email: str, password: str) -> dict:
        email = email.lower()
        if email in self.accounts:
            raise MembershipError("An account with this email already exists.", status_code=409)
        self.accounts[email] = password
        return {"user_confirmed": False, "destination": email}

    def confirm(self, email: str, code: str) -> None:
        self.confirmed.add(email.lower())

    def resend_signup_code(self, email: str) -> dict:
        return {"destination": email}


def _build_service(demo_mode: bool = True) -> MembershipService:
    return MembershipService(
        member_directory=PlaceholderMemberDirectory.with_demo_seed(),
        subscriber_auth=FakeSubscriberAuth(),
        code_service=EmailCodeService(LoggingEmailCodeSender()),
        token_service=MembershipTokenService("test-secret"),
        billing=PlaceholderBillingProvider(),
        demo_mode=demo_mode,
    )


class JwtTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        token = _jwt_encode({"sub": "x", "exp": 9_999_999_999}, "s")
        self.assertEqual(_jwt_decode(token, "s")["sub"], "x")

    def test_bad_signature_returns_none(self) -> None:
        token = _jwt_encode({"sub": "x", "exp": 9_999_999_999}, "s")
        self.assertIsNone(_jwt_decode(token, "other-secret"))

    def test_expired_returns_none(self) -> None:
        token = _jwt_encode({"sub": "x", "exp": 1}, "s")
        self.assertIsNone(_jwt_decode(token, "s"))

    def test_garbage_returns_none(self) -> None:
        self.assertIsNone(_jwt_decode("not-a-token", "s"))


class MemberLoginTests(unittest.TestCase):
    def test_valid_member_gets_token_without_code(self) -> None:
        service = _build_service()
        result = service.member_login("member@gsi.org", "member1234")
        self.assertEqual(result["account_type"], "member")
        principal = service.validate_token(result["access_token"])
        self.assertIsNotNone(principal)
        self.assertEqual(principal.account_type, "member")
        self.assertEqual(principal.user_id, "member:member@gsi.org")

    def test_bad_member_credentials_rejected(self) -> None:
        service = _build_service()
        with self.assertRaises(MembershipError) as ctx:
            service.member_login("member@gsi.org", "wrong")
        self.assertEqual(ctx.exception.status_code, 401)


class SubscriberFlowTests(unittest.TestCase):
    def test_login_requires_code_then_issues_token(self) -> None:
        service = _build_service()
        challenge = service.subscriber_login_start("subscriber@gsi.org", "subscriber1234")
        self.assertEqual(challenge["challenge"], "email_code")
        self.assertIn("demo_code", challenge)  # demo_mode on
        result = service.subscriber_verify("subscriber@gsi.org", challenge["demo_code"])
        self.assertEqual(result["account_type"], "subscriber")
        principal = service.validate_token(result["access_token"])
        self.assertEqual(principal.account_type, "subscriber")
        self.assertEqual(principal.user_id, "subscriber:subscriber@gsi.org")

    def test_wrong_password_rejected(self) -> None:
        service = _build_service()
        with self.assertRaises(MembershipError):
            service.subscriber_login_start("subscriber@gsi.org", "nope")

    def test_wrong_code_rejected(self) -> None:
        service = _build_service()
        service.subscriber_login_start("subscriber@gsi.org", "subscriber1234")
        with self.assertRaises(MembershipError):
            service.subscriber_verify("subscriber@gsi.org", "000000")

    def test_demo_mode_off_hides_code(self) -> None:
        service = _build_service(demo_mode=False)
        challenge = service.subscriber_login_start("subscriber@gsi.org", "subscriber1234")
        self.assertNotIn("demo_code", challenge)


class SubscribeCheckoutTests(unittest.TestCase):
    def test_subscribe_registers_cognito_user_and_needs_confirm(self) -> None:
        service = _build_service()
        result = service.subscribe("new@user.com", "password123", {"card": "4242"})
        self.assertEqual(result["subscription"]["status"], "trialing")
        self.assertFalse(result["signup"]["user_confirmed"])
        self.assertNotIn("access_token", result)  # must confirm email first
        # Confirming the emailed signup code starts the session.
        confirmed = service.subscriber_confirm_signup("new@user.com", "123456")
        self.assertEqual(confirmed["account_type"], "subscriber")
        principal = service.validate_token(confirmed["access_token"])
        self.assertEqual(principal.account_type, "subscriber")

    def test_duplicate_email_rejected(self) -> None:
        service = _build_service()
        with self.assertRaises(MembershipError) as ctx:
            service.subscribe("subscriber@gsi.org", "password123", {})
        self.assertEqual(ctx.exception.status_code, 409)

    def test_short_password_rejected(self) -> None:
        service = _build_service()
        with self.assertRaises(MembershipError):
            service.subscribe("brand@new.com", "short", {})


if __name__ == "__main__":
    unittest.main()
