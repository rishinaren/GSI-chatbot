import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { getAdminFeedback } from "../api";
import { PageHead, useAdminData } from "./AdminLayout";
import { formatDate } from "./AdminIcons";

// A rating on its own tells an administrator nothing actionable. What they need
// is the exchange: the question that was asked, the answer it got, and what the
// person said was wrong with it. So the list is one tab per conversation, and
// opening one shows those three things in that order.

function ratingLabel(rating) {
  return rating === "up" ? "Liked" : "Disliked";
}

export default function AdminFeedback({ head }) {
  const { data, loading, error } = useAdminData(getAdminFeedback, "Could not load the feedback.");
  const [openId, setOpenId] = useState("");

  const conversations = data?.conversations ?? [];
  const active =
    conversations.find((row) => row.conversation_id === openId) ?? conversations[0] ?? null;

  useEffect(() => {
    if (!openId && conversations.length) {
      setOpenId(conversations[0].conversation_id);
    }
  }, [openId, conversations]);

  if (loading) {
    return (
      <>
        <PageHead head={head} />
        <p className="library-empty">Loading…</p>
      </>
    );
  }

  if (error) {
    return (
      <>
        <PageHead head={head} />
        <div className="library-error">{error}</div>
      </>
    );
  }

  if (!conversations.length) {
    return (
      <>
        <PageHead head={head} />
        <p className="library-empty">Nobody has rated an answer yet.</p>
      </>
    );
  }

  return (
    <>
      <PageHead
        head={head}
        sub={`${data.down_count} of ${data.total} rated answers were marked unhelpful.`}
      />

      <div className="feedback-split">
        <div className="feedback-tabs" role="tablist" aria-orientation="vertical">
          {conversations.map((row) => (
            <button
              key={row.conversation_id}
              type="button"
              role="tab"
              aria-selected={active?.conversation_id === row.conversation_id}
              className={`feedback-tab ${
                active?.conversation_id === row.conversation_id ? "active" : ""
              }`}
              onClick={() => setOpenId(row.conversation_id)}
            >
              <span className="feedback-tab-title">Conversation about {row.title}</span>
              <span className="feedback-tab-meta">
                {row.down_count > 0 ? `${row.down_count} disliked` : `${row.count} rated`}
                {row.last_at ? ` · ${formatDate(row.last_at)}` : ""}
              </span>
            </button>
          ))}
        </div>

        <div className="feedback-detail">
          {active?.items.map((item) => (
            <article key={item.feedback_id} className="feedback-entry">
              <header className="feedback-entry-head">
                <span className={`feedback-rating ${item.rating}`}>{ratingLabel(item.rating)}</span>
                <span className="feedback-entry-meta">
                  {item.user_email}
                  {item.created_at ? ` · ${formatDate(item.created_at)}` : ""}
                </span>
              </header>

              {item.question ? (
                <div className="feedback-field">
                  <div className="feedback-field-label">Question</div>
                  <p className="feedback-field-text">{item.question}</p>
                </div>
              ) : null}

              {item.answer ? (
                <div className="feedback-field">
                  <div className="feedback-field-label">Answer</div>
                  {/* Rendered, not raw: the administrator should be looking at
                      the answer the person actually read, headings and all. */}
                  <div className="feedback-answer markdown-body">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm, remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                    >
                      {item.answer}
                    </ReactMarkdown>
                  </div>
                </div>
              ) : null}

              <div className="feedback-field">
                <div className="feedback-field-label">What they told us</div>
                <p className="feedback-field-text">
                  {item.comment || <span className="feedback-none">No comment left.</span>}
                </p>
              </div>
            </article>
          ))}
        </div>
      </div>
    </>
  );
}
