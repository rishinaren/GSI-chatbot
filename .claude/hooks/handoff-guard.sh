#!/usr/bin/env bash
# Stop hook. Blocks the turn from ending if the assistant edited project files
# this session but has not yet updated HANDOFF.md to reflect them. Once HANDOFF.md
# is edited, handoff-track.sh clears the pending flag and this hook allows the stop.
set -euo pipefail

input="$(cat)"
active="$(printf '%s' "$input" | jq -r '.stop_hook_active // false')"
session="$(printf '%s' "$input" | jq -r '.session_id // "nosession"')"
pending="${TMPDIR:-/tmp}/gsi-handoff-pending-${session}"

# Loop guard: if we already blocked once this stop-cycle, allow the stop.
[ "$active" = "true" ] && exit 0

if [ -s "$pending" ]; then
  files="$(sort -u "$pending" | sed 's#.*/##' | paste -sd ', ' -)"
  reason="You edited files this session (${files}) but HANDOFF.md has not been updated to reflect them. Per the project working agreement, update HANDOFF.md now — strip stale entries and record what changed — then finish. (Enforced by .claude/hooks/handoff-guard.sh.)"
  jq -cn --arg r "$reason" '{decision:"block", reason:$r}'
  exit 0
fi

exit 0
