import { useEffect, useId, useRef, useState } from "react";
import { submitFeedback } from "../api";
import { saveReportPdf } from "../reportPdf";

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

function ReportIcon() {
  return (
    <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M6.5 3.5h7l4 4V20a1 1 0 0 1-1 1h-10a1 1 0 0 1-1-1V4.5a1 1 0 0 1 1-1z" />
      <path d="M13.5 3.5V8h4" />
      <path d="M12 10.5v6M9.5 14l2.5 2.5 2.5-2.5" />
    </svg>
  );
}

function ConversationIcon() {
  return (
    <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" strokeWidth="1.55" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M8 6.5h10a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-1.5l-2.5 2v-2H8a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2z" />
      <path d="M6 9.5H5a2 2 0 0 0-2 2v4a2 2 0 0 0 2 2h1v2l2.5-2H11" />
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

export default function MessageActions({
  question,
  answer,
  citations,
  conversationExchanges,
  conversationId,
  onRetry,
  canRetry,
}) {
  const wrapRef = useRef(null);
  const reportPickerId = useId();
  const [copied, setCopied] = useState(false);
  const [reportPickerOpen, setReportPickerOpen] = useState(false);
  const [reportState, setReportState] = useState("idle");
  const [rating, setRating] = useState("");
  const [ticket, setTicket] = useState(null);
  const [boxOpen, setBoxOpen] = useState(false);
  const [comment, setComment] = useState("");
  const [sending, setSending] = useState(false);
  const [thanks, setThanks] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!reportPickerOpen) return undefined;

    const closeOnOutsideClick = (event) => {
      if (!wrapRef.current?.contains(event.target)) {
        setReportPickerOpen(false);
      }
    };
    const closeOnEscape = (event) => {
      if (event.key === "Escape") {
        setReportPickerOpen(false);
      }
    };
    document.addEventListener("pointerdown", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [reportPickerOpen]);

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

  async function handleSaveReport(scope) {
    if (reportState === "saving") return;
    setError("");
    setReportState("saving");
    try {
      const report =
        scope === "conversation"
          ? { scope, exchanges: conversationExchanges }
          : { scope: "answer", question, answer, citations };
      const result = await saveReportPdf(report);
      if (result.canceled) {
        setReportState("idle");
        return;
      }
      setReportPickerOpen(false);
      setReportState("saved");
      window.setTimeout(() => setReportState("idle"), 1800);
    } catch {
      setReportState("idle");
      setError("We could not create that PDF. Please try again.");
    }
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
    <div className="msg-actions-wrap" ref={wrapRef}>
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
        <button
          type="button"
          className={`msg-action msg-report-action ${reportState === "saved" ? "on" : ""}`}
          onClick={() => {
            setError("");
            setReportPickerOpen((open) => !open);
          }}
          disabled={reportState === "saving"}
          aria-expanded={reportPickerOpen}
          aria-controls={reportPickerId}
          aria-label={reportState === "saved" ? "Report saved" : "Export PDF report"}
          title="Export PDF report"
        >
          {reportState === "saved" ? <TickIcon /> : <ReportIcon />}
          <span>{reportState === "saving" ? "Preparing…" : reportState === "saved" ? "Saved" : "Export PDF"}</span>
        </button>
      </div>

      {reportPickerOpen ? (
        <div
          className="report-scope-picker"
          id={reportPickerId}
          role="group"
          aria-label="Choose what to include in the PDF"
        >
          <button
            type="button"
            className="report-scope-option"
            onClick={() => handleSaveReport("answer")}
            disabled={reportState === "saving"}
          >
            <span className="report-scope-icon"><ReportIcon /></span>
            <span>
              <strong>This question &amp; answer</strong>
              <small>Include its citations</small>
            </span>
          </button>
          <button
            type="button"
            className="report-scope-option"
            onClick={() => handleSaveReport("conversation")}
            disabled={reportState === "saving"}
          >
            <span className="report-scope-icon"><ConversationIcon /></span>
            <span>
              <strong>Entire conversation</strong>
              <small>Every question, answer &amp; citation</small>
            </span>
          </button>
        </div>
      ) : null}

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
