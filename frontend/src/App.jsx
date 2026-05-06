import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";

const conversationId = crypto.randomUUID();

const starterQuestions = [
  "What is ASTM D5321 about?",
  "Compare D5321 and D6241.",
  "Which standard is relevant for transmissivity?",
];

/**
 * Models often emit math as plain parentheses or \(...\) instead of $...$.
 * remark-math + KaTeX need explicit math delimiters.
 */
function normalizeAssistantContent(text) {
  let out = text;

  out = out.replace(/\\\(([\s\S]*?)\\\)/g, (_, inner) => `$${inner.trim()}$`);
  out = out.replace(/\\\[([\s\S]*?)\\\]/g, (_, inner) => `$$\n${inner.trim()}\n$$`);

  const latexHint = /\\[a-zA-Z]+|\\\(|\\\)|\\\[|\\\]|\^[_\{]|\^[_0-9A-Za-z]|_[\{0-9A-Za-z]/;

  out = out.replace(/\(\s*([A-Za-z](?:_\{[^}]+\}|_[A-Za-z0-9]+)?)\s*\)/g, (_, symbol) => `$${symbol}$`);

  out = out.replace(/\(\s*([^()]{0,240}?)\s*\)/g, (match, inner) => {
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

function App() {
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      text: "Ask a standards question. I will answer from the ingested documents and cite the source pages I used.",
      citations: [],
      retrievedDocuments: [],
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
    setMessages((current) => [
      ...current,
      { id: crypto.randomUUID(), role: "user", text: trimmed },
    ]);
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
          retrievedDocuments: data.retrieved_documents ?? [],
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

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <h1>GSI Chatbot</h1>
        <p className="sidebar-copy">
          Minimal local UI for testing the standards RAG pipeline.
        </p>

        <label className="control-group">
          <span>Units</span>
          <select value={unitPreference} onChange={(event) => setUnitPreference(event.target.value)}>
            <option value="">Original standard units</option>
            <option value="si">SI / Metric</option>
            <option value="imperial">US / Imperial</option>
          </select>
        </label>

        <div className="starter-list">
          <span>Try one:</span>
          {starterQuestions.map((starter) => (
            <button
              key={starter}
              type="button"
              className="starter-button"
              onClick={() => void sendQuestion(starter)}
              disabled={isLoading}
            >
              {starter}
            </button>
          ))}
        </div>
      </aside>

      <main className="chat-panel">
        <div className="chat-header">
          <div>
            <h2>Standards Q&A</h2>
            <p>Answers come from your ingested standards index.</p>
          </div>
          <span className={`status-pill ${isLoading ? "busy" : "ready"}`}>
            {isLoading ? "Thinking..." : "Ready"}
          </span>
        </div>

        <div className="message-list">
          {messages.map((message) => (
            <article key={message.id} className={`message ${message.role}`}>
              <div className="message-role">{message.role === "assistant" ? "Assistant" : "You"}</div>
              {message.role === "assistant" ? (
                <div className="message-body markdown-body">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm, remarkMath]}
                    rehypePlugins={[rehypeKatex]}
                  >
                    {normalizeAssistantContent(message.text)}
                  </ReactMarkdown>
                </div>
              ) : (
                <pre className="message-text">{message.text}</pre>
              )}

              {message.retrievedDocuments?.length > 0 && (
                <div className="meta-block">
                  <div className="meta-title">Retrieved documents</div>
                  <ul>
                    {message.retrievedDocuments.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}

              {message.citations?.length > 0 && (
                <div className="meta-block">
                  <div className="meta-title">Citations</div>
                  <ul>
                    {message.citations.map((citation) => (
                      <li key={citation.chunk_id}>
                        <strong>{citation.standard_id}</strong>
                        {citation.section ? `, Section ${citation.section}` : ""}
                        {citation.page_start ? `, page ${citation.page_start}` : ""}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </article>
          ))}
        </div>

        <form className="composer" onSubmit={onSubmit}>
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask about a standard, compare two methods, or ask a follow-up..."
            rows={4}
          />
          <div className="composer-footer">
            {error ? <span className="error-text">{error}</span> : <span className="hint-text">FastAPI backend should be running on port 8000.</span>}
            <button type="submit" disabled={!canSubmit}>
              Send
            </button>
          </div>
        </form>
      </main>
    </div>
  );
}

export default App;
