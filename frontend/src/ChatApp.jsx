import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import ChatSidebar from "./components/ChatSidebar";
import AuthExperience from "./components/AuthExperience";
import AdminPortal from "./components/AdminPortal";
import MessageActions from "./components/MessageActions";
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
  uploadChatAttachment,
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

function AttachIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M12 5v14M5 12h14" />
    </svg>
  );
}

function PdfGlyph() {
  return (
    <svg viewBox="0 0 24 24" width="22" height="22" fill="none" aria-hidden>
      <path
        d="M14 2.5H7.5A2.5 2.5 0 0 0 5 5v14a2.5 2.5 0 0 0 2.5 2.5h9A2.5 2.5 0 0 0 19 19V7.5z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      <path d="M14 2.5V7a.5.5 0 0 0 .5.5H19" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
      <text x="12" y="17.4" textAnchor="middle" fontSize="6.4" fontWeight="700" fill="currentColor">
        PDF
      </text>
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
    // Names only: the file itself was never kept, so a reopened chat shows what
    // was asked about without pretending the document is still readable.
    attachments: message.attachments ?? [],
  }));
}

/** One attached file, from the moment it is picked to the moment it is read.
 *
 * The chip appears the instant a file is chosen, showing a spinner in place of
 * the icon, so choosing a large PDF never looks like nothing happened.
 */
function AttachmentChip({ item, onRemove }) {
  const failed = item.status === "error";
  return (
    <div className={`attach-chip ${item.status}`} title={failed ? item.error : item.name}>
      <span className="attach-chip-icon" aria-hidden="true">
        {item.status === "loading" ? <span className="attach-spinner" /> : <PdfGlyph />}
      </span>
      <span className="attach-chip-text">
        <span className="attach-chip-name">{item.name}</span>
        <span className="attach-chip-kind">
          {item.status === "loading" ? "Reading…" : failed ? item.error : "PDF"}
        </span>
      </span>
      {onRemove ? (
        <button
          type="button"
          className="attach-chip-remove"
          onClick={() => onRemove(item.localId)}
          aria-label={`Remove ${item.name}`}
        >
          <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden>
            <path d="M6 6l12 12M18 6L6 18" />
          </svg>
        </button>
      ) : null}
    </div>
  );
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
  const [adminOpen, setAdminOpen] = useState(false);
  const [attachments, setAttachments] = useState([]);

  // Nothing may be sent while a file is still being read - the answer would be
  // written without the document the question is about.
  const uploading = attachments.some((item) => item.status === "loading");
  const canSubmit = useMemo(
    () => question.trim().length > 0 && !isLoading && !uploading,
    [question, isLoading, uploading],
  );
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
    setAttachments([]);
  }

  // The file is read on the server, which keeps it for this session only - it is
  // never added to the library everyone queries.
  async function handleAttach(files) {
    const picked = Array.from(files ?? []);
    if (!picked.length) return;
    setError("");

    const pending = picked.map((file) => ({
      localId: crypto.randomUUID(),
      name: file.name,
      status: "loading",
      attachmentId: null,
      error: "",
    }));
    setAttachments((current) => [...current, ...pending]);

    await Promise.all(
      picked.map(async (file, index) => {
        const { localId } = pending[index];
        try {
          const saved = await uploadChatAttachment(file);
          setAttachments((current) =>
            current.map((item) =>
              item.localId === localId
                ? {
                    ...item,
                    status: "ready",
                    attachmentId: saved.attachment_id,
                    name: saved.file_name || item.name,
                    pageCount: saved.page_count,
                  }
                : item,
            ),
          );
        } catch (uploadError) {
          const message =
            uploadError instanceof Error ? uploadError.message : "We could not read that file.";
          setAttachments((current) =>
            current.map((item) =>
              item.localId === localId ? { ...item, status: "error", error: message } : item,
            ),
          );
        }
      }),
    );
  }

  function removeAttachment(localId) {
    setAttachments((current) => current.filter((item) => item.localId !== localId));
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

  async function sendQuestion(text, { attach = attachments } = {}) {
    const trimmed = text.trim();
    if (!trimmed) return;

    const ready = attach.filter((item) => item.status === "ready");
    setError("");
    setIsLoading(true);
    setFollowUpSuggestions([]);
    setMessages((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        role: "user",
        text: trimmed,
        attachments: ready.map((item) => ({ file_name: item.name, page_count: item.pageCount })),
        // Kept so "Ask again" re-asks with the same documents rather than
        // silently dropping them.
        attachmentIds: ready.map((item) => item.attachmentId),
      },
    ]);
    setQuestion("");
    setAttachments([]);

    try {
      const conversationId = await ensureConversationId();
      const data = await sendChat({
        question: trimmed,
        conversation_id: conversationId,
        unit_preference: unitPreference || null,
        attachment_ids: ready.map((item) => item.attachmentId),
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
    if (!canSubmit) return;
    void sendQuestion(question);
  }

  function onKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (!canSubmit) return;
      void sendQuestion(question);
    }
  }

  // "Ask again" re-sends the question that produced this answer, with whatever
  // was attached to it, as a new turn - the old exchange stays readable above.
  function retryFrom(index) {
    for (let cursor = index - 1; cursor >= 0; cursor -= 1) {
      const candidate = messages[cursor];
      if (candidate.role === "user") {
        const ids = candidate.attachmentIds ?? [];
        void sendQuestion(candidate.text, {
          attach: ids.map((attachmentId, position) => ({
            status: "ready",
            attachmentId,
            name: candidate.attachments?.[position]?.file_name ?? "Attachment",
            pageCount: candidate.attachments?.[position]?.page_count,
          })),
        });
        return;
      }
    }
  }

  if (loading) {
    return <div className="app-loading">Loading…</div>;
  }

  // The admin portal takes over the whole window rather than floating above the
  // chat: it is a different job, and Chats/Projects belong to the person asking
  // questions, not the person curating what gets answered.
  if (adminOpen && authed && canManageLibrary) {
    return (
      <AdminPortal
        email={getUserEmail()}
        onExit={() => setAdminOpen(false)}
        onSignOut={() => {
          signOut();
          window.location.reload();
        }}
      />
    );
  }

  const composerProps = {
    question,
    setQuestion,
    canSubmit,
    onSubmit,
    onKeyDown,
    attachments,
    onAttach: handleAttach,
    onRemoveAttachment: removeAttachment,
  };
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
        onOpenLibrary={() => setAdminOpen(true)}
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
            conversationId={activeConversationId}
            onRetry={retryFrom}
          />
        ) : (
          <EmptyState error={error} composerProps={composerProps} />
        )}
      </div>

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

function ChatThread({
  messages,
  isLoading,
  error,
  followUpSuggestions,
  sendQuestion,
  composerProps,
  conversationId,
  onRetry,
}) {
  return (
    <div className="chat-body">
      <div className="chat-content">
        <div className="message-scroll">
          {messages.map((message, index) => (
            <div key={message.id} className={`msg-row ${message.role}`}>
              <div className={`bubble ${message.role === "user" ? "user" : "bot"}`}>
                {message.role === "assistant" ? (
                  <div className="markdown-body">
                    <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
                      {normalizeAssistantContent(message.text)}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <>
                    {message.attachments?.length > 0 && (
                      <div className="msg-attachments">
                        {message.attachments.map((file) => (
                          <span className="msg-attachment" key={file.file_name}>
                            <PdfGlyph />
                            {file.file_name}
                          </span>
                        ))}
                      </div>
                    )}
                    <p className="message-text-plain">{message.text}</p>
                  </>
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
                        // An attachment belongs to the person who asked, not to the
                        // library, so it is labelled as theirs and never linked -
                        // the file was read for the question and not kept.
                        const attached = citation.source_kind === "attachment";
                        const docLine = attached ? (
                          <>
                            <strong>Your document</strong>
                            {", "}
                            {citation.title}
                          </>
                        ) : (
                          <>
                            <strong>{citation.standard_id}</strong>
                            {", "}
                            {citation.title}
                          </>
                        );
                        // GRI/ASTM citations link out (member portal / ASTM Compass);
                        // everything else falls back to the inline authed PDF.
                        const href = attached
                          ? null
                          : citation.source_url || withAuthedFileUrl(citation.pdf_url);
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

                {message.role === "assistant" && !message.needsClarification ? (
                  <MessageActions
                    answer={message.text}
                    conversationId={conversationId}
                    canRetry={!isLoading}
                    onRetry={() => onRetry(index)}
                  />
                ) : null}
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

function Composer({
  question,
  setQuestion,
  canSubmit,
  onSubmit,
  onKeyDown,
  variant,
  attachments = [],
  onAttach,
  onRemoveAttachment,
}) {
  const fileInput = useRef(null);

  return (
    <form className={`composer-inner ${variant}`} onSubmit={onSubmit}>
      {attachments.length > 0 ? (
        <div className="composer-attachments">
          {attachments.map((item) => (
            <AttachmentChip key={item.localId} item={item} onRemove={onRemoveAttachment} />
          ))}
        </div>
      ) : null}

      <div className="composer-row">
        <input
          ref={fileInput}
          type="file"
          accept="application/pdf,.pdf"
          multiple
          className="composer-file-input"
          onChange={(event) => {
            void onAttach?.(event.target.files);
            // Reset so picking the same file twice still fires a change.
            event.target.value = "";
          }}
        />
        <button
          type="button"
          className="attach-btn"
          onClick={() => fileInput.current?.click()}
          aria-label="Attach a PDF"
          title="Attach a PDF"
        >
          <AttachIcon />
        </button>
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
      </div>
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
