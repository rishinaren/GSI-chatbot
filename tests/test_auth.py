import unittest

from standards_rag.auth import _map_cognito_error_message, auth_public_config, load_auth_config_from_env


class CognitoErrorMappingTests(unittest.TestCase):
    def test_username_exists(self) -> None:
        message = _map_cognito_error_message(
            "User already exists",
            error_type="UsernameExistsException",
        )
        self.assertIn("already exists", message)

    def test_invalid_password(self) -> None:
        message = _map_cognito_error_message(
            "Password did not conform with policy",
            error_type="InvalidPasswordException",
        )
        self.assertIn("8 characters", message)

    def test_code_mismatch(self) -> None:
        message = _map_cognito_error_message(
            "Invalid verification code provided",
            error_type="CodeMismatchException",
        )
        self.assertIn("verification code", message.lower())


class AuthPublicConfigTests(unittest.TestCase):
    def test_includes_signup_enabled(self) -> None:
        public = auth_public_config(load_auth_config_from_env())
        self.assertIn("signup_enabled", public)


if __name__ == "__main__":
    unittest.main()
