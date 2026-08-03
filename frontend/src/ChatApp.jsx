import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import ChatSidebar from "./components/ChatSidebar";
import AuthExperience from "./components/AuthExperience";
import AdminLibrary from "./components/AdminLibrary";
import {
  ApiError,
  assignConversationToProject,
  createConversation,
  createProject,
  deleteConversation,
  deleteProject,
  getAdminConfig,
  getConversation,
  listConversations,
  listProjects,
  pinConversation,
  renameProject,
  sendChat,
  withAuthedFileUrl,
} from "./api";
import { clearSession, getUserEmail, isAuthenticated, signOut } from "./auth";

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
    videoSuggestions: message.video_suggestions ?? [],
  }));
}

function ChatApp() {
  const [loading, setLoading] = useState(true);
  const [authed, setAuthed] = useState(isAuthenticated());
  const [conversations, setConversations] = useState([]);
  const [projects, setProjects] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [unitPreference, setUnitPreference] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [followUpSuggestions, setFollowUpSuggestions] = useState([]);
  const [canManageLibrary, setCanManageLibrary] = useState(false);
  const [libraryOpen, setLibraryOpen] = useState(false);

  const canSubmit = useMemo(() => question.trim().length > 0 && !isLoading, [question, isLoading]);
  const hasStarted = messages.length > 0;
  const showAuthModal = !authed;

  useEffect(() => {
    void bootstrap();
  }, []);

  async function bootstrap() {
    if (!isAuthenticated()) {
      setLoading(false);
      return;
    }
    try {
      await refreshSidebar();
      void refreshAdminAccess();
      setAuthed(true);
    } catch (refreshError) {
      if (refreshError instanceof ApiError && refreshError.status === 401) {
        clearSession();
        setAuthed(false);
      } else {
        setError(refreshError instanceof Error ? refreshError.message : "Failed to initialize app.");
      }
    } finally {
      setLoading(false);
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

  // The library entry point only appears for accounts on the admin allowlist;
  // the API enforces the same rule on every library call.
  async function refreshAdminAccess() {
    try {
      const config = await getAdminConfig();
      setCanManageLibrary(Boolean(config.is_admin));
    } catch {
      setCanManageLibrary(false);
    }
  }

  async function handleSignedIn() {
    setError("");
    setAuthed(true);
    void refreshAdminAccess();
    try {
      await refreshSidebar();
    } catch (refreshError) {
      setError(refreshError instanceof Error ? refreshError.message : "Failed to load your chats.");
    }
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
          videoSuggestions: data.video_suggestions ?? [],
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

  if (loading) {
    return <div className="app-loading">Loading…</div>;
  }

  const composerProps = { question, setQuestion, canSubmit, onSubmit, onKeyDown };
  // Units selector hidden: in prod the OpenAI rewriter strips the appended unit-conversion
  // note (it forbids numbers not in the source), so the control had no visible effect.
  // unitPreference is retained (sent as-is) so a real unit feature can be re-added later.

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
        userEmail={authed ? getUserEmail() : ""}
        canSignOut={authed}
        canManageLibrary={authed && canManageLibrary}
        onOpenLibrary={() => setLibraryOpen(true)}
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
              <p className="chat-header-sub">Standards Q&amp;A — grounded in your ASTM, ISO, and GRI index</p>
            </div>
          </div>
          <div className="header-right">
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
          </div>
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

      {libraryOpen && canManageLibrary ? (
        <AdminLibrary onClose={() => setLibraryOpen(false)} />
      ) : null}

      {showAuthModal ? <AuthExperience onSignedIn={handleSignedIn} connectionError={error} /> : null}
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

                {message.videoSuggestions?.length > 0 && (
                  <div className="video-block">
                    {message.videoSuggestions.map((video) => (
                      <VideoSuggestion key={video.video_id || video.youtube_id} video={video} />
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
                        // GRI/ASTM citations link out (member portal / ASTM Compass);
                        // everything else falls back to the inline authed PDF.
                        const href = citation.source_url || withAuthedFileUrl(citation.pdf_url);
                        return (
                          <li key={citation.chunk_id}>
                            {href ? (
                              <a className="doc-pdf-link" href={href} target="_blank" rel="noopener noreferrer">
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

// Tier-2 (semantic-only) match: the video isn't directly about a cited standard,
// so we ask before showing it rather than auto-embedding.
function VideoSuggestion({ video }) {
  const [shown, setShown] = useState(false);
  if (shown) {
    return <VideoEmbed video={video} />;
  }
  return (
    <div className="video-suggestion">
      <span className="video-suggestion-text">
        {video.reason ? (
          video.reason
        ) : (
          <>A related video may help: <strong>{video.title}</strong></>
        )}
      </span>
      <button type="button" className="video-suggestion-btn" onClick={() => setShown(true)}>
        Show video
      </button>
    </div>
  );
}

export default ChatApp;
