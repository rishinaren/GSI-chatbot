#!/usr/bin/env python3
"""Send a real login-code email through the configured provider.

Verifies the whole delivery path (credentials, From address, DNS/DKIM, and
whether the message lands in the inbox or in spam) without going through the
login UI. Reads the same env vars the API does, so a pass here means the API
will work too.

    PYTHONPATH=src python3 scripts/send_test_login_email.py you@example.com

Add --dry-run to print the resolved provider and stop before sending.
"""

from __future__ import annotations

import argparse
import logging
import sys

from standards_rag.env_bootstrap import load_dotenv_files
from standards_rag.membership import _build_email_code_sender_from_env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recipient", help="address to send the test code to")
    parser.add_argument("--code", default="123456", help="code to send (default: 123456)")
    parser.add_argument("--dry-run", action="store_true", help="resolve the provider but do not send")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv_files()

    sender = _build_email_code_sender_from_env()
    provider = getattr(sender, "PROVIDER_NAME", "custom")
    print(f"provider: {provider}")

    if provider == "log":
        print(
            "\nNo email transport is configured, so the code would only be logged.\n"
            "Set RESEND_API_KEY (+ MEMBERSHIP_CODE_EMAIL_FROM), or SMTP_HOST and friends.",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        print("dry run - nothing sent")
        return 0

    print(f"sending {args.code} to {args.recipient} ...")
    try:
        sender.send(args.recipient, args.code)
    except Exception as exc:  # noqa: BLE001 - this script exists to surface transport errors
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1

    print(
        "sent - check the inbox.\n"
        "If it landed in spam, the From domain is not DKIM-aligned: send from a domain "
        "you own and have verified with the provider, not a consumer gmail.com address."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
