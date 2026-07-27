#!/usr/bin/env bash
# Point the running API at an email provider for membership login codes.
#
# Stores the credential in Secrets Manager (never in the App Runner env, never in
# shell history) and wires App Runner to reference it, preserving every other
# setting. App Runner redeploys automatically; takes ~3 minutes.
#
#   ./scripts/set_email_provider.sh resend "GSI Chatbot <login@yourdomain.com>"
#   ./scripts/set_email_provider.sh smtp   "GSI Chatbot <login@yourdomain.com>" smtp-relay.brevo.com 587 your-smtp-login
#
# The secret itself is read from a prompt, not an argument, so it never lands in
# `ps` output or ~/.zsh_history.
set -euo pipefail

SERVICE_ARN="arn:aws:apprunner:us-east-1:833239388618:service/gsi-standards-rag-api/0bb1f04e23324f0abaf0d17ce01c643b"
REGION="us-east-1"
ACCOUNT="833239388618"

PROVIDER="${1:-}"
FROM_ADDRESS="${2:-}"

if [[ -z "$PROVIDER" || -z "$FROM_ADDRESS" ]]; then
  echo "usage: $0 <resend|smtp> \"Name <addr@domain>\" [smtp_host smtp_port smtp_username]" >&2
  exit 64
fi

case "$PROVIDER" in
  resend) SECRET_NAME="gsi/resend-api-key"; SECRET_ENV="RESEND_API_KEY"; PROMPT="Resend API key (re_...)" ;;
  smtp)   SECRET_NAME="gsi/smtp-password";  SECRET_ENV="SMTP_PASSWORD";  PROMPT="SMTP password" ;;
  *) echo "unknown provider: $PROVIDER (expected 'resend' or 'smtp')" >&2; exit 64 ;;
esac

# --- read the credential without echoing it -------------------------------
read -rsp "$PROMPT: " CREDENTIAL
echo
[[ -n "$CREDENTIAL" ]] || { echo "empty credential, aborting" >&2; exit 1; }

TMP="$(mktemp)"
chmod 600 "$TMP"
trap 'rm -f "$TMP" "$TMP.cfg"' EXIT
printf '%s' "$CREDENTIAL" > "$TMP"
unset CREDENTIAL

# --- store it (create or rotate) ------------------------------------------
if aws secretsmanager describe-secret --secret-id "$SECRET_NAME" >/dev/null 2>&1; then
  aws secretsmanager put-secret-value --secret-id "$SECRET_NAME" \
    --secret-string "file://$TMP" --query 'Name' --output text
  echo "rotated existing secret $SECRET_NAME"
else
  aws secretsmanager create-secret --name "$SECRET_NAME" \
    --description "GSI chatbot membership login-code email credential" \
    --secret-string "file://$TMP" --query 'Name' --output text
  echo "created secret $SECRET_NAME"
fi

# --- merge into the live App Runner config, changing nothing else ---------
aws apprunner describe-service --service-arn "$SERVICE_ARN" \
  --query 'Service.SourceConfiguration' --output json > "$TMP.cfg"

PROVIDER="$PROVIDER" FROM_ADDRESS="$FROM_ADDRESS" SECRET_ENV="$SECRET_ENV" \
SECRET_ARN="arn:aws:secretsmanager:${REGION}:${ACCOUNT}:secret:${SECRET_NAME}" \
SMTP_HOST="${3:-}" SMTP_PORT="${4:-587}" SMTP_USERNAME="${5:-}" \
python3 - "$TMP.cfg" <<'PY'
import json, os, sys

path = sys.argv[1]
cfg = json.load(open(path))
image = cfg["ImageRepository"]["ImageConfiguration"]
env = image.setdefault("RuntimeEnvironmentVariables", {})
secrets = image.setdefault("RuntimeEnvironmentSecrets", {})

env["MEMBERSHIP_EMAIL_PROVIDER"] = os.environ["PROVIDER"]
env["MEMBERSHIP_CODE_EMAIL_FROM"] = os.environ["FROM_ADDRESS"]
secrets[os.environ["SECRET_ENV"]] = os.environ["SECRET_ARN"]

if os.environ["PROVIDER"] == "smtp":
    host = os.environ.get("SMTP_HOST", "")
    if not host:
        sys.exit("smtp provider needs a host: $0 smtp \"<from>\" <host> <port> <username>")
    env["SMTP_HOST"] = host
    env["SMTP_PORT"] = os.environ.get("SMTP_PORT") or "587"
    env["SMTP_USERNAME"] = os.environ.get("SMTP_USERNAME", "")

json.dump(cfg, open(path, "w"), indent=2)
print(f"provider={env['MEMBERSHIP_EMAIL_PROVIDER']} from={env['MEMBERSHIP_CODE_EMAIL_FROM']}")
PY

aws apprunner update-service --service-arn "$SERVICE_ARN" \
  --source-configuration "file://$TMP.cfg" --query 'Service.Status' --output text

cat <<EOF

App Runner is redeploying (~3 min). Verify with:

  curl -s https://37g2f3huxn.us-east-1.awsapprunner.com/auth/membership/config | python3 -m json.tool

Expect "email_provider": "$PROVIDER". Then send yourself a real code:

  PYTHONPATH=src python3 scripts/send_test_login_email.py you@example.com
EOF
