import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import ChatSidebar from "./components/ChatSidebar";
import LoginPanel from "./components/LoginPanel";
import {
  ApiError,
  assignConversationToProject,
  createConversation,
  createProject,
  deleteConversation,
  deleteProject,
  getConversation,
  listConversations,
  listProjects,
  pinConversation,
  renameProject,
  sendChat,
  withAuthedFileUrl,
} from "./api";
import { clearSession, getUserEmail, loadAuthState, signOut } from "./auth";

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

function messagesFromConversation(record) {
  if (!record?.messages?.length) {
    return [];
  }
  return record.messages.map((message, index) => ({
    id: `${record.conversation_id}-${index}`,
    role: message.role,
    text: message.text,
    citations: message.citations ?? [],
    videos: message.videos ?? [],
  }));
}

function ChatApp() {
  const [authState, setAuthState] = useState({ loading: true, authRequired: false, isLoggedIn: true });
  const [conversations, setConversations] = useState([]);
  const [projects, setProjects] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [unitPreference, setUnitPreference] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [followUpSuggestions, setFollowUpSuggestions] = useState([]);

  const canSubmit = useMemo(() => question.trim().length > 0 && !isLoading, [question, isLoading]);
  const hasStarted = messages.length > 0;
  const showAuthModal = authState.authRequired && !authState.isLoggedIn;

  useEffect(() => {
    void bootstrap();
  }, []);

  async function bootstrap() {
    try {
      let state = await loadAuthState();
      if (state.isLoggedIn) {
        try {
          await refreshSidebar();
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

  async function refreshProjects() {
    const data = await listProjects();
    setProjects(data.projects ?? []);
  }

  async function refreshSidebar() {
    await Promise.all([refreshConversations(), refreshProjects()]);
  }

  async function handleSignedIn() {
    const state = await loadAuthState();
    setAuthState({ loading: false, ...state });
    await refreshSidebar();
  }

  function startNewChat() {
    setError("");
    setActiveConversationId(null);
    setMessages([]);
    setFollowUpSuggestions([]);
    setQuestion("");
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
      startNewChat();
    }
    await refreshSidebar();
  }

  async function handleTogglePin(conversationId, pinned) {
    setConversations((current) =>
      current.map((conversation) =>
        conversation.conversation_id === conversationId ? { ...conversation, pinned } : conversation,
      ),
    );
    try {
      await pinConversation(conversationId, pinned);
      await refreshConversations();
    } catch (pinError) {
      setError(pinError instanceof Error ? pinError.message : "Could not update pin.");
      await refreshConversations();
    }
  }

  async function handleCreateProject(name) {
    try {
      const project = await createProject(name);
      await refreshProjects();
      return project;
    } catch (projectError) {
      setError(projectError instanceof Error ? projectError.message : "Could not create project.");
      return null;
    }
  }

  async function handleRenameProject(projectId, name) {
    try {
      await renameProject(projectId, name);
      await refreshProjects();
    } catch (projectError) {
      setError(projectError instanceof Error ? projectError.message : "Could not rename project.");
    }
  }

  async function handleDeleteProject(projectId) {
    try {
      await deleteProject(projectId);
      await refreshSidebar();
    } catch (projectError) {
      setError(projectError instanceof Error ? projectError.message : "Could not delete project.");
    }
  }

  async function handleAssignToProject(conversationId, projectId) {
    setConversations((current) =>
      current.map((conversation) =>
        conversation.conversation_id === conversationId
          ? { ...conversation, project_id: projectId }
          : conversation,
      ),
    );
    try {
      await assignConversationToProject(conversationId, projectId);
      await refreshSidebar();
    } catch (assignError) {
      setError(assignError instanceof Error ? assignError.message : "Could not update project.");
      await refreshSidebar();
    }
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
          videos: data.videos ?? [],
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
    return <div className="app-loading">Loading…</div>;
  }

  const composerProps = { question, setQuestion, canSubmit, onSubmit, onKeyDown };
  const unitControl = (
    <div className="unit-control">
      <label htmlFor="units">Units</label>
      <div className="select-wrap">
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
  );

  return (
    <div className="app-shell">
      <ChatSidebar
        conversations={conversations}
        projects={projects}
        activeConversationId={activeConversationId}
        onSelect={handleSelectConversation}
        onNewChat={startNewChat}
        onDelete={handleDeleteConversation}
        onTogglePin={handleTogglePin}
        onCreateProject={handleCreateProject}
        onRenameProject={handleRenameProject}
        onDeleteProject={handleDeleteProject}
        onAssignToProject={handleAssignToProject}
        userEmail={authState.configured ? getUserEmail() : ""}
        canSignOut={Boolean(authState.configured)}
        onSignOut={() => {
          signOut();
          window.location.reload();
        }}
      />

      <div className="app-main">
        <header className="chat-header">
          <div className="chat-header-inner">
            <div className="chat-header-titles">
              <h1>GSI Chatbot</h1>
              <p className="chat-header-sub">Standards Q&amp;A — grounded in your ASTM &amp; ISO index</p>
            </div>
            <div className="header-tools">{unitControl}</div>
          </div>
          <a
            className="header-logo-link"
            href="https://geosynthetic-institute.org/"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="Geosynthetic Institute website"
            title="Geosynthetic Institute"
          >
            <img className="header-logo" src="/gsi-logo.png" alt="Geosynthetic Institute logo" />
          </a>
        </header>

        {hasStarted ? (
          <ChatThread
            messages={messages}
            isLoading={isLoading}
            error={error}
            followUpSuggestions={followUpSuggestions}
            sendQuestion={sendQuestion}
            composerProps={composerProps}
          />
        ) : (
          <EmptyState error={error} composerProps={composerProps} />
        )}
      </div>

      {showAuthModal ? <LoginPanel onSignedIn={handleSignedIn} connectionError={error} /> : null}
    </div>
  );
}

function EmptyState({ error, composerProps }) {
  return (
    <div className="empty-state">
      <div className="empty-inner">
        <h2 className="empty-title">Ready when you are.</h2>
        <p className="empty-sub">
          Ask about an ASTM or ISO standard, a test method, or request a video walkthrough.
        </p>
        {error ? <div className="composer-error centered">{error}</div> : null}
        <Composer {...composerProps} variant="hero" />
      </div>
    </div>
  );
}

function ChatThread({ messages, isLoading, error, followUpSuggestions, sendQuestion, composerProps }) {
  return (
    <div className="chat-body">
      <div className="chat-content">
        <div className="message-scroll">
          {messages.map((message) => (
            <div key={message.id} className={`msg-row ${message.role}`}>
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

                {message.videos?.length > 0 && (
                  <div className="video-block">
                    <div className="meta-title">Related videos</div>
                    {message.videos.map((video) => (
                      <VideoEmbed key={video.video_id || video.youtube_id} video={video} />
                    ))}
                  </div>
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
                        const pdfUrl = withAuthedFileUrl(citation.pdf_url);
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
          <Composer {...composerProps} variant="docked" />
        </div>
      </div>
    </div>
  );
}

function Composer({ question, setQuestion, canSubmit, onSubmit, onKeyDown, variant }) {
  return (
    <form className={`composer-inner ${variant}`} onSubmit={onSubmit}>
      <textarea
        className="composer-input"
        value={question}
        onChange={(event) => setQuestion(event.target.value)}
        onKeyDown={onKeyDown}
        placeholder="Ask anything about standards…"
        rows={1}
      />
      <button type="submit" className="send-btn" disabled={!canSubmit} aria-label="Send">
        <SendIcon />
      </button>
    </form>
  );
}

function VideoEmbed({ video }) {
  return (
    <div className="video-embed">
      <div className="video-frame">
        <iframe
          src={video.embed_url}
          title={video.title}
          loading="lazy"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
        />
      </div>
      <a className="video-caption" href={video.youtube_url} target="_blank" rel="noopener noreferrer">
        {video.title}
      </a>
    </div>
  );
}

export default ChatApp;
