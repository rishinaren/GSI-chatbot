import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import ChatSidebar from "./components/ChatSidebar";
import LoginPanel from "./components/LoginPanel";
import {
  createConversation,
  deleteConversation,
  getConversation,
  listConversations,
  sendChat,
  withApiBase,
} from "./api";
import { ApiError } from "./api";
import { clearSession, loadAuthState, signOut } from "./auth";

const PAREN_NOT_AFTER_LATEX_OPENER = "(?<!\\\\(?:left|bigl|Bigl|biggl|Biggl|mleft))";

function normalizeAssistantContent(text) {
  let out = text;
  out = out.replace(/\\\(([\s\S]*?)\\\)/g, (_, inner) => `$${inner.trim()}$`);
  out = out.replace(/\\\[([\s\S]*?)\\\]/g, (_, inner) => `$$\n${inner.trim()}\n$$`);
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
    if (!latexHint.test(inner) || inner.includes("$")) {
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

function welcomeMessage() {
  return {
    id: "welcome",
    role: "assistant",
    text: "Ask a standards question. I will answer from the ingested documents and cite the source pages I used.",
    citations: [],
  };
}

function messagesFromConversation(record) {
  if (!record?.messages?.length) {
    return [welcomeMessage()];
  }
  return record.messages.map((message, index) => ({
    id: `${record.conversation_id}-${index}`,
    role: message.role,
    text: message.text,
    citations: message.citations ?? [],
  }));
}

function ChatApp() {
  const [authState, setAuthState] = useState({ loading: true, authRequired: false, isLoggedIn: true });
  const [conversations, setConversations] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [messages, setMessages] = useState([welcomeMessage()]);
  const [question, setQuestion] = useState("");
  const [unitPreference, setUnitPreference] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [followUpSuggestions, setFollowUpSuggestions] = useState([]);

  const canSubmit = useMemo(() => question.trim().length > 0 && !isLoading, [question, isLoading]);

  useEffect(() => {
    void bootstrap();
  }, []);

  async function bootstrap() {
    try {
      let state = await loadAuthState();
      if (state.isLoggedIn) {
        try {
          await refreshConversations();
        } catch (refreshError) {
          if (refreshError instanceof ApiError && refreshError.status === 401) {
            clearSession();
            state = { ...state, isLoggedIn: false };
          } else {
            throw refreshError;
          }
        }
      }
      setAuthState({ loading: false, ...state });
    } catch (bootstrapError) {
      setAuthState({
        loading: false,
        authRequired: true,
        isLoggedIn: false,
        configured: false,
      });
      setError(bootstrapError instanceof Error ? bootstrapError.message : "Failed to initialize app.");
    }
  }

  async function refreshConversations() {
    const data = await listConversations();
    setConversations(data.conversations ?? []);
  }

  async function handleSignedIn() {
    const state = await loadAuthState();
    setAuthState({ loading: false, ...state });
    await refreshConversations();
  }

  async function handleNewChat() {
    setError("");
    const record = await createConversation({ unit_preference: unitPreference || null });
    setActiveConversationId(record.conversation_id);
    setMessages([welcomeMessage()]);
    setFollowUpSuggestions([]);
    await refreshConversations();
  }

  async function handleSelectConversation(conversationId) {
    setError("");
    setActiveConversationId(conversationId);
    const record = await getConversation(conversationId);
    setMessages(messagesFromConversation(record));
    setUnitPreference(record.unit_preference || "");
    setFollowUpSuggestions([]);
  }

  async function handleDeleteConversation(conversationId) {
    await deleteConversation(conversationId);
    if (conversationId === activeConversationId) {
      setActiveConversationId(null);
      setMessages([welcomeMessage()]);
    }
    await refreshConversations();
  }

  async function ensureConversationId() {
    if (activeConversationId) {
      return activeConversationId;
    }
    const record = await createConversation({ unit_preference: unitPreference || null });
    setActiveConversationId(record.conversation_id);
    await refreshConversations();
    return record.conversation_id;
  }

  async function sendQuestion(text) {
    const trimmed = text.trim();
    if (!trimmed) return;

    setError("");
    setIsLoading(true);
    setFollowUpSuggestions([]);
    setMessages((current) => [...current, { id: crypto.randomUUID(), role: "user", text: trimmed }]);
    setQuestion("");

    try {
      const conversationId = await ensureConversationId();
      const data = await sendChat({
        question: trimmed,
        conversation_id: conversationId,
        unit_preference: unitPreference || null,
      });
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: data.answer,
          citations: data.citations ?? [],
          needsClarification: data.needs_clarification,
        },
      ]);
      setFollowUpSuggestions(data.follow_up_suggestions ?? []);
      await refreshConversations();
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

  if (authState.loading) {
    return <div className="app-loading">Loading...</div>;
  }

  if (authState.authRequired && !authState.isLoggedIn) {
    return <LoginPanel onSignedIn={handleSignedIn} connectionError={error} />;
  }

  return (
    <div className="app-shell">
      <ChatSidebar
        conversations={conversations}
        activeConversationId={activeConversationId}
        onSelect={handleSelectConversation}
        onNewChat={handleNewChat}
        onDelete={handleDeleteConversation}
      />

      <div className="app-main">
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
              {authState.configured ? (
                <button type="button" className="sign-out-btn" onClick={() => { signOut(); window.location.reload(); }}>
                  Sign out
                </button>
              ) : null}
            </div>
          </div>
        </header>

        <ChatBody
          messages={messages}
          isLoading={isLoading}
          error={error}
          followUpSuggestions={followUpSuggestions}
          question={question}
          setQuestion={setQuestion}
          canSubmit={canSubmit}
          onSubmit={onSubmit}
          onKeyDown={onKeyDown}
          sendQuestion={sendQuestion}
        />
      </div>
    </div>
  );
}

function ChatBody({
  messages,
  isLoading,
  error,
  followUpSuggestions,
  question,
  setQuestion,
  canSubmit,
  onSubmit,
  onKeyDown,
  sendQuestion,
}) {
  return (
    <div className="chat-body">
      <div className="chat-content">
        <div className="message-scroll">
          {messages.map((message) => (
            <div key={message.id} className={`msg-row ${message.role}`}>
              <ChatAvatar role={message.role} />
              <div className={`bubble ${message.role === "user" ? "user" : "bot"}`}>
                {message.role === "assistant" ? (
                  <div className="markdown-body">
                    <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
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
                            ? citation.page_end != null && citation.page_end !== citation.page_start
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
                        const pdfUrl = withApiBase(citation.pdf_url);
                        return (
                          <li key={citation.chunk_id}>
                            {pdfUrl ? (
                              <a className="doc-pdf-link" href={pdfUrl} target="_blank" rel="noopener noreferrer">
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
              <ChatAvatar role="assistant" />
              <div className="typing-dots">
                <span />
                <span />
                <span />
              </div>
            </div>
          )}
        </div>

        {followUpSuggestions.length > 0 && (
          <div className="follow-up-row">
            {followUpSuggestions.map((suggestion) => (
              <button key={suggestion} type="button" className="follow-up-chip" onClick={() => sendQuestion(suggestion)}>
                {suggestion}
              </button>
            ))}
          </div>
        )}

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
  );
}

function ChatAvatar({ role }) {
  return (
    <div className={`avatar ${role === "user" ? "user" : ""}`} aria-hidden>
      {role === "user" ? "You" : "AI"}
    </div>
  );
}

export default ChatApp;
