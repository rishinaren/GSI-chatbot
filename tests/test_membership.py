import json
import unittest
from unittest import mock

from standards_rag.membership import (
    EmailCodeService,
    LoggingEmailCodeSender,
    MembershipError,
    MembershipService,
    MembershipTokenService,
    PlaceholderBillingProvider,
    PlaceholderMemberDirectory,
    ResendEmailCodeSender,
    SesEmailCodeSender,
    SmtpEmailCodeSender,
    _build_email_code_sender_from_env,
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


def _env(**overrides: str) -> mock._patch_dict:
    """Clears every email-transport var, then applies the given overrides."""
    base = dict.fromkeys(
        (
            "MEMBERSHIP_EMAIL_PROVIDER",
            "MEMBERSHIP_CODE_EMAIL_FROM",
            "RESEND_API_KEY",
            "SMTP_HOST",
            "SMTP_PORT",
            "SMTP_USERNAME",
            "SMTP_PASSWORD",
            "SMTP_STARTTLS",
        ),
        "",
    )
    return mock.patch.dict("os.environ", {**base, **overrides}, clear=False)


class EmailSenderSelectionTests(unittest.TestCase):
    def test_auto_prefers_resend(self) -> None:
        with _env(RESEND_API_KEY="re_test", SMTP_HOST="smtp.example.com", MEMBERSHIP_CODE_EMAIL_FROM="a@b.org"):
            self.assertIsInstance(_build_email_code_sender_from_env(), ResendEmailCodeSender)

    def test_auto_falls_back_to_smtp_then_ses(self) -> None:
        with _env(SMTP_HOST="smtp.example.com", MEMBERSHIP_CODE_EMAIL_FROM="a@b.org"):
            self.assertIsInstance(_build_email_code_sender_from_env(), SmtpEmailCodeSender)
        with _env(MEMBERSHIP_CODE_EMAIL_FROM="a@b.org"):
            self.assertIsInstance(_build_email_code_sender_from_env(), SesEmailCodeSender)

    def test_explicit_provider_wins(self) -> None:
        with _env(
            MEMBERSHIP_EMAIL_PROVIDER="smtp",
            RESEND_API_KEY="re_test",
            SMTP_HOST="smtp.example.com",
            MEMBERSHIP_CODE_EMAIL_FROM="a@b.org",
        ):
            self.assertIsInstance(_build_email_code_sender_from_env(), SmtpEmailCodeSender)

    def test_missing_credentials_fall_back_to_logging(self) -> None:
        # No transport configured at all.
        with _env():
            self.assertIsInstance(_build_email_code_sender_from_env(), LoggingEmailCodeSender)
        # Provider chosen but no sender address to send from.
        with _env(MEMBERSHIP_EMAIL_PROVIDER="resend", RESEND_API_KEY="re_test"):
            self.assertIsInstance(_build_email_code_sender_from_env(), LoggingEmailCodeSender)
        # Sender address present but the provider's own credential is missing.
        with _env(MEMBERSHIP_EMAIL_PROVIDER="resend", MEMBERSHIP_CODE_EMAIL_FROM="a@b.org"):
            self.assertIsInstance(_build_email_code_sender_from_env(), LoggingEmailCodeSender)
        # Typo'd provider name must not silently send nothing.
        with _env(MEMBERSHIP_EMAIL_PROVIDER="mailgunn", MEMBERSHIP_CODE_EMAIL_FROM="a@b.org"):
            self.assertIsInstance(_build_email_code_sender_from_env(), LoggingEmailCodeSender)


class ResendSenderTests(unittest.TestCase):
    def test_posts_expected_payload(self) -> None:
        sender = ResendEmailCodeSender("re_test_key", "GSI <login@gsi.org>")
        captured: dict = {}

        def fake_urlopen(request, timeout=None):  # noqa: ANN001 - test double
            captured["url"] = request.full_url
            captured["headers"] = request.headers
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return mock.MagicMock(__enter__=lambda s: s, __exit__=lambda *a: False, read=lambda: b"{}")

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            sender.send("user@example.com", "123456")

        self.assertEqual(captured["url"], "https://api.resend.com/emails")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer re_test_key")
        # Cloudflare 403s urllib's default agent, so a custom one is mandatory.
        self.assertEqual(captured["headers"]["User-agent"], ResendEmailCodeSender.USER_AGENT)
        self.assertEqual(captured["body"]["from"], "GSI <login@gsi.org>")
        self.assertEqual(captured["body"]["to"], ["user@example.com"])
        self.assertIn("123456", captured["body"]["text"])
        self.assertIn("123456", captured["body"]["html"])


class SmtpSenderTests(unittest.TestCase):
    def test_starttls_and_login_on_587(self) -> None:
        client = mock.MagicMock()
        client.__enter__.return_value = client
        with mock.patch("smtplib.SMTP", return_value=client) as smtp:
            SmtpEmailCodeSender(
                "smtp-relay.example.com",
                port=587,
                username="user",
                password="pass",
                from_address="login@gsi.org",
            ).send("user@example.com", "654321")

        smtp.assert_called_once_with("smtp-relay.example.com", 587, timeout=15)
        client.starttls.assert_called_once()
        client.login.assert_called_once_with("user", "pass")
        message = client.send_message.call_args.args[0]
        self.assertEqual(message["To"], "user@example.com")
        self.assertIn("654321", message.get_body(("plain",)).get_content())

    def test_implicit_tls_on_465(self) -> None:
        client = mock.MagicMock()
        client.__enter__.return_value = client
        with mock.patch("smtplib.SMTP_SSL", return_value=client) as smtp_ssl:
            SmtpEmailCodeSender(
                "smtp-relay.example.com", port=465, from_address="login@gsi.org"
            ).send("user@example.com", "111222")

        smtp_ssl.assert_called_once_with("smtp-relay.example.com", 465, timeout=15)
        client.starttls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
