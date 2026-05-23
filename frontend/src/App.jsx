import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";

const conversationId = crypto.randomUUID();

/**
 * Models often emit math as plain parentheses or \(...\) instead of $...$.
 * remark-math + KaTeX need explicit math delimiters.
 *
 * Important: do not wrap (...) in $...$ when the "(" belongs to \left( ... \right)
 * — that used to produce \left$...\right$ and break KaTeX.
 */
const PAREN_NOT_AFTER_LATEX_OPENER = "(?<!\\\\(?:left|bigl|Bigl|biggl|Biggl|mleft))";

function normalizeAssistantContent(text) {
  let out = text;

  out = out.replace(/\\\(([\s\S]*?)\\\)/g, (_, inner) => `$${inner.trim()}$`);
  out = out.replace(/\\\[([\s\S]*?)\\\]/g, (_, inner) => `$$\n${inner.trim()}\n$$`);

  // Repair invalid \left$ ... \right$ (model typo or legacy normalization bug).
  out = out.replace(/\\left\s*\$([\s\S]*?)\\right\s*\$/g, (_, inner) => `\\left(${inner.trim()}\\right)`);
  out = out.replace(/\\left\s*\$([\s\S]*?)\\right\s*\)/g, (_, inner) => `\\left(${inner.trim()}\\right)`);
  out = out.replace(/\\right\s*\$(?=\s*\^)/g, "\\right)");

  const latexHint = /\\[a-zA-Z]+|\\\(|\\\)|\\\[|\\\]|\^[_\{]|\^[_0-9A-Za-z]|_[\{0-9A-Za-z]/;

  const singleSymbolParen = new RegExp(
    `${PAREN_NOT_AFTER_LATEX_OPENER}\\(\\s*([A-Za-z](?:_\\{[^}]+\\}|_[A-Za-z0-9]+)?)\\s*\\)`,
    "g",
  );
  out = out.replace(singleSymbolParen, (_, symbol) => `$${symbol}$`);

  const genericParen = new RegExp(
    `${PAREN_NOT_AFTER_LATEX_OPENER}\\(\\s*([^()]{0,240}?)\\s*\\)`,
    "g",
  );
  out = out.replace(genericParen, (match, inner) => {
    if (!latexHint.test(inner)) {
      return match;
    }
    if (inner.includes("$")) {
      return match;
    }
    return `$${inner.trim()}$`;
  });

  return out;
}

function SendIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
    </svg>
  );
}

function App() {
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      text: "Ask a standards question. I will answer from the ingested documents and cite the source pages I used.",
      citations: [],
    },
  ]);
  const [question, setQuestion] = useState("");
  const [unitPreference, setUnitPreference] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const canSubmit = useMemo(() => question.trim().length > 0 && !isLoading, [question, isLoading]);

  async function sendQuestion(text) {
    const trimmed = text.trim();
    if (!trimmed) return;

    setError("");
    setIsLoading(true);
    setMessages((current) => [...current, { id: crypto.randomUUID(), role: "user", text: trimmed }]);
    setQuestion("");

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: trimmed,
          conversation_id: conversationId,
          unit_preference: unitPreference || null,
        }),
      });

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      const data = await response.json();
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: data.answer,
          citations: data.citations ?? [],
        },
      ]);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Something went wrong.");
    } finally {
      setIsLoading(false);
    }
  }

  function onSubmit(event) {
    event.preventDefault();
    void sendQuestion(question);
  }

  function onKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendQuestion(question);
    }
  }

  return (
    <div className="app-outer">
      <div className="phone">
        <header className="chat-header">
          <div className="chat-header-inner">
            <div className="chat-header-titles">
              <h1>GSI Chatbot</h1>
              <p className="chat-header-sub">Standards Q&amp;A — answers from your ingested index</p>
            </div>
            <div className="header-tools">
              <label htmlFor="units">Units</label>
              <select
                id="units"
                value={unitPreference}
                onChange={(event) => setUnitPreference(event.target.value)}
              >
                <option value="">As in standard</option>
                <option value="si">SI / metric</option>
                <option value="imperial">US / imperial</option>
              </select>
            </div>
          </div>
        </header>

        <div className="chat-body">
          <div className="chat-content">
            <div className="message-scroll">
              {messages.map((message) => (
                <div key={message.id} className={`msg-row ${message.role}`}>
                  <div className={`avatar ${message.role === "user" ? "user" : ""}`} aria-hidden>
                    {message.role === "user" ? "You" : "AI"}
                  </div>
                  <div className={`bubble ${message.role === "user" ? "user" : "bot"}`}>
                    {message.role === "assistant" ? (
                      <div className="markdown-body">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm, remarkMath]}
                          rehypePlugins={[rehypeKatex]}
                        >
                          {normalizeAssistantContent(message.text)}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <p className="message-text-plain">{message.text}</p>
                    )}

                    {message.citations?.length > 0 && (
                      <div className="meta-block">
                        <div className="meta-title">Citations</div>
                        <ul>
                          {message.citations.map((citation) => {
                            const pageSuffix =
                              citation.page_start != null
                                ? citation.page_end != null &&
                                  citation.page_end !== citation.page_start
                                  ? `, pages ${citation.page_start}-${citation.page_end}`
                                  : `, page ${citation.page_start}`
                                : "";
                            const docLine = (
                              <>
                                <strong>{citation.standard_id}</strong>
                                {", "}
                                {citation.title}
                              </>
                            );
                            return (
                              <li key={citation.chunk_id}>
                                {citation.pdf_url ? (
                                  <a
                                    className="doc-pdf-link"
                                    href={citation.pdf_url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                  >
                                    {docLine}
                                  </a>
                                ) : (
                                  docLine
                                )}
                                {citation.section ? `, Section ${citation.section}` : ""}
                                {pageSuffix}
                              </li>
                            );
                          })}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="typing-row" aria-live="polite" aria-busy="true">
                  <div className="avatar" aria-hidden>
                    AI
                  </div>
                  <div className="typing-dots">
                    <span />
                    <span />
                    <span />
                  </div>
                </div>
              )}
            </div>

            <div className="composer-wrap">
              {error ? <div className="composer-error">{error}</div> : null}
              <form className="composer-inner" onSubmit={onSubmit}>
                <span className="composer-icon" aria-hidden>
                  &#9786;
                </span>
                <textarea
                  className="composer-input"
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  onKeyDown={onKeyDown}
                  placeholder="Type here..."
                  rows={1}
                />
                <button type="submit" className="send-btn" disabled={!canSubmit} aria-label="Send">
                  <SendIcon />
                </button>
              </form>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
