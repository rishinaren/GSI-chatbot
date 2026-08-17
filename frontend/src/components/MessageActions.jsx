import { useState } from "react";
import { submitFeedback } from "../api";

// What you can do with an answer once you have read it. The row is invisible
// until the message is hovered or something in it takes focus, so a finished
// answer stays clean text and the controls appear where the eye already is.
//
// Thumbs-down is two steps that are one opinion: the click records straight away
// (so a dismissed box still leaves the signal) and the comment, if it comes, is
// sent back against the same id rather than as a second rating.

function CopyIcon() {
  return (
    <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="9" y="9" width="11" height="11" rx="2" />
      <path d="M5 15V6a2 2 0 0 1 2-2h9" />
    </svg>
  );
}

function TickIcon() {
  return (
    <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M20 6L9 17l-5-5" />
    </svg>
  );
}

function ThumbUpIcon({ filled }) {
  return (
    <svg viewBox="0 0 24 24" width="15" height="15" fill={filled ? "currentColor" : "none"} stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M7 10.5v9H4.8A1.8 1.8 0 0 1 3 17.7v-5.4a1.8 1.8 0 0 1 1.8-1.8z" />
      <path d="M7 10.5l4.2-7a2 2 0 0 1 3.7 1.2l-.7 4.1h4.4a2.2 2.2 0 0 1 2.1 2.8l-1.6 5.8a2.5 2.5 0 0 1-2.4 1.8H7z" />
    </svg>
  );
}

function ThumbDownIcon({ filled }) {
  return (
    <svg viewBox="0 0 24 24" width="15" height="15" fill={filled ? "currentColor" : "none"} stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M17 13.5v-9h2.2A1.8 1.8 0 0 1 21 6.3v5.4a1.8 1.8 0 0 1-1.8 1.8z" />
      <path d="M17 13.5l-4.2 7a2 2 0 0 1-3.7-1.2l.7-4.1H5.4a2.2 2.2 0 0 1-2.1-2.8l1.6-5.8A2.5 2.5 0 0 1 7.3 4.5H17z" />
    </svg>
  );
}

function RetryIcon() {
  return (
    <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M20 11a8 8 0 1 0-1.1 5.3" />
      <path d="M20 5v6h-6" />
    </svg>
  );
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // clipboard.writeText needs a secure context; fall back to a scratch node.
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    // execCommand only copies from a focused selection, so focus before selecting.
    area.focus();
    area.select();
    let ok = false;
    try {
      ok = document.execCommand("copy");
    } catch {
      ok = false;
    }
    document.body.removeChild(area);
    return ok;
  }
}

export default function MessageActions({ answer, conversationId, onRetry, canRetry }) {
  const [copied, setCopied] = useState(false);
  const [rating, setRating] = useState("");
  const [ticket, setTicket] = useState(null);
  const [boxOpen, setBoxOpen] = useState(false);
  const [comment, setComment] = useState("");
  const [sending, setSending] = useState(false);
  const [thanks, setThanks] = useState(false);
  const [error, setError] = useState("");

  async function handleCopy() {
    const ok = await copyText(answer ?? "");
    if (!ok) {
      setError("We could not copy that. Please select the text and copy it.");
      return;
    }
    setError("");
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  async function rate(next) {
    if (!conversationId) return;
    setError("");
    // Show the new state immediately; a rating is not worth making anyone wait.
    setRating(next);
    if (next === "down") {
      setBoxOpen(true);
    } else {
      setBoxOpen(false);
      setThanks(false);
    }
    try {
      const saved = await submitFeedback({ conversationId, rating: next, answer });
      setTicket({ feedbackId: saved.feedback_id, createdAt: saved.created_at });
    } catch {
      setRating("");
      setBoxOpen(false);
      setError("We could not save that just now. Please try again.");
    }
  }

  async function sendComment(event) {
    event.preventDefault();
    if (!comment.trim() || sending) return;
    setSending(true);
    setError("");
    try {
      await submitFeedback({
        conversationId,
        rating: "down",
        comment,
        answer,
        feedbackId: ticket?.feedbackId,
        createdAt: ticket?.createdAt,
      });
      setBoxOpen(false);
      setComment("");
      setThanks(true);
    } catch {
      setError("We could not send that just now. Please try again.");
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="msg-actions-wrap">
      <div className="msg-actions">
        <button
          type="button"
          className="msg-action"
          onClick={handleCopy}
          aria-label={copied ? "Copied" : "Copy answer"}
          title={copied ? "Copied" : "Copy"}
        >
          {copied ? <TickIcon /> : <CopyIcon />}
        </button>
        <button
          type="button"
          className={`msg-action ${rating === "up" ? "on" : ""}`}
          onClick={() => rate("up")}
          aria-pressed={rating === "up"}
          aria-label="Good answer"
          title="Good answer"
        >
          <ThumbUpIcon filled={rating === "up"} />
        </button>
        <button
          type="button"
          className={`msg-action ${rating === "down" ? "on" : ""}`}
          onClick={() => rate("down")}
          aria-pressed={rating === "down"}
          aria-label="Bad answer"
          title="Bad answer"
        >
          <ThumbDownIcon filled={rating === "down"} />
        </button>
        {canRetry ? (
          <button
            type="button"
            className="msg-action"
            onClick={onRetry}
            aria-label="Ask again"
            title="Ask again"
          >
            <RetryIcon />
          </button>
        ) : null}
      </div>

      {boxOpen ? (
        <form className="feedback-box" onSubmit={sendComment}>
          <label className="feedback-label" htmlFor="feedback-comment">
            Give us any feedback about the response you received.
          </label>
          <textarea
            id="feedback-comment"
            className="feedback-input"
            value={comment}
            onChange={(event) => setComment(event.target.value)}
            placeholder="What was wrong with it?"
            rows={3}
            maxLength={2000}
            autoFocus
          />
          <div className="feedback-actions">
            <button
              type="button"
              className="feedback-skip"
              onClick={() => {
                setBoxOpen(false);
                setComment("");
              }}
            >
              Not now
            </button>
            <button type="submit" className="feedback-send" disabled={!comment.trim() || sending}>
              {sending ? "Sending…" : "Send feedback"}
            </button>
          </div>
        </form>
      ) : null}

      {thanks ? <p className="feedback-thanks">Thanks - we have passed that on.</p> : null}
      {error ? <p className="feedback-error">{error}</p> : null}
    </div>
  );
}
