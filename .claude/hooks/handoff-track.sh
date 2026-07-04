#!/usr/bin/env bash
# PostToolUse hook (matcher: Edit|Write|MultiEdit|NotebookEdit).
# Records every file the assistant edits this session into a per-session
# "pending" file. Editing HANDOFF.md clears it (the handoff is now current).
# The companion Stop hook (handoff-guard.sh) blocks the turn from ending while
# the pending file is non-empty, so HANDOFF.md is always kept in sync.
set -euo pipefail

input="$(cat)"
session="$(printf '%s' "$input" | jq -r '.session_id // "nosession"')"
path="$(printf '%s' "$input" | jq -r '.tool_input.file_path // .tool_input.notebook_path // empty')"
[ -z "$path" ] && exit 0

pending="${TMPDIR:-/tmp}/gsi-handoff-pending-${session}"

# Updating the handoff clears the pending flag.
if [ "$(basename "$path")" = "HANDOFF.md" ]; then
  rm -f "$pending"
  exit 0
fi

# Don't demand a handoff update for edits to the hook system itself or scratch files.
case "$path" in
  */.claude/hooks/*|*/.claude/settings*|*/scratchpad/*|*/__pycache__/*) exit 0 ;;
esac

printf '%s\n' "$path" >> "$pending"
exit 0
