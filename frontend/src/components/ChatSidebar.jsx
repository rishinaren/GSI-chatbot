import { useEffect, useRef, useState } from "react";

function PinIcon({ filled }) {
  return (
    <svg
      viewBox="0 0 24 24"
      width="15"
      height="15"
      aria-hidden="true"
      fill={filled ? "currentColor" : "none"}
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M14.7 2.6a1 1 0 0 1 1.41 0l5.28 5.29a1 1 0 0 1 0 1.41l-.32.32a2.5 2.5 0 0 1-2.62.58l-1.64-.6-3.86 3.86.5 3.2a1 1 0 0 1-1.7.86L8.1 14.49l-4.39 4.4a1 1 0 0 1-1.42-1.42l4.4-4.39-3.43-3.16a1 1 0 0 1 .86-1.7l3.2.5L8.18 4.9a2.5 2.5 0 0 1 .58-2.62z" />
    </svg>
  );
}

function DotsIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true">
      <circle cx="5" cy="12" r="1.6" />
      <circle cx="12" cy="12" r="1.6" />
      <circle cx="19" cy="12" r="1.6" />
    </svg>
  );
}

function FolderIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="15"
      height="15"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M3 7a2 2 0 0 1 2-2h3.5l2 2H19a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
    </svg>
  );
}

function ChevronLeft() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M15 18l-6-6 6-6" />
    </svg>
  );
}

const MENU_WIDTH = 196;

export default function ChatSidebar({
  conversations,
  projects,
  activeConversationId,
  onSelect,
  onNewChat,
  onDelete,
  onTogglePin,
  onCreateProject,
  onRenameProject,
  onDeleteProject,
  onAssignToProject,
  userEmail,
  onSignOut,
  canSignOut,
}) {
  const [activeTab, setActiveTab] = useState("chats");
  const [openProjectId, setOpenProjectId] = useState(null);
  const [menu, setMenu] = useState(null);
  const [prompt, setPrompt] = useState(null);

  const closeMenu = () => setMenu(null);

  useEffect(() => {
    if (!menu) return undefined;
    const onDocClick = () => closeMenu();
    const onKey = (event) => event.key === "Escape" && closeMenu();
    window.addEventListener("click", onDocClick);
    window.addEventListener("keydown", onKey);
    window.addEventListener("resize", closeMenu);
    return () => {
      window.removeEventListener("click", onDocClick);
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("resize", closeMenu);
    };
  }, [menu]);

  function openMenu(event, kind, id) {
    event.preventDefault();
    event.stopPropagation();
    const rect = event.currentTarget.getBoundingClientRect();
    const x = Math.min(rect.right, window.innerWidth - MENU_WIDTH - 12);
    setMenu({ kind, id, view: "root", x: Math.max(12, x), y: rect.bottom + 6 });
  }

  function askName({ title, confirmLabel, initialValue = "", onConfirm }) {
    setPrompt({ title, confirmLabel, value: initialValue, onConfirm });
  }

  function newProjectFor(conversationId) {
    askName({
      title: "Name your project",
      confirmLabel: "Create & add",
      onConfirm: async (name) => {
        const project = await onCreateProject(name);
        if (project && conversationId) {
          await onAssignToProject(conversationId, project.project_id);
        }
      },
    });
  }

  const openProject = projects.find((project) => project.project_id === openProjectId) || null;

  function renderChatRow(conversation) {
    const isActive = conversation.conversation_id === activeConversationId;
    return (
      <div
        key={conversation.conversation_id}
        className={`sidebar-item ${isActive ? "active" : ""} ${conversation.pinned ? "pinned" : ""}`}
      >
        <button
          type="button"
          className="sidebar-item-button"
          onClick={() => onSelect(conversation.conversation_id)}
        >
          <span className="sidebar-item-title">{conversation.title}</span>
          <span className="sidebar-item-meta">
            {conversation.pinned ? "Pinned · " : ""}
            {conversation.message_count || 0} messages
          </span>
        </button>
        <div className="sidebar-item-actions">
          <button
            type="button"
            className={`icon-btn pin-btn ${conversation.pinned ? "active" : ""}`}
            aria-label={conversation.pinned ? "Unpin chat" : "Pin chat"}
            title={conversation.pinned ? "Unpin" : "Pin"}
            onClick={(event) => {
              event.stopPropagation();
              onTogglePin(conversation.conversation_id, !conversation.pinned);
            }}
          >
            <PinIcon filled={conversation.pinned} />
          </button>
          <button
            type="button"
            className="icon-btn dots-btn"
            aria-label="Chat options"
            title="More"
            onClick={(event) => openMenu(event, "chat", conversation.conversation_id)}
          >
            <DotsIcon />
          </button>
        </div>
      </div>
    );
  }

  const menuConversation =
    menu?.kind === "chat"
      ? conversations.find((conversation) => conversation.conversation_id === menu.id)
      : null;

  return (
    <aside className="chat-sidebar">
      <div className="sidebar-brand">
        <span className="sidebar-brand-dot" aria-hidden="true" />
        <span>GSI Chatbot</span>
      </div>

      <button type="button" className="sidebar-new-chat" onClick={onNewChat}>
        <span className="plus" aria-hidden="true">+</span>
        New chat
      </button>

      <div className="sidebar-tabs" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === "chats"}
          className={`sidebar-tab ${activeTab === "chats" ? "active" : ""}`}
          onClick={() => setActiveTab("chats")}
        >
          Chats
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === "projects"}
          className={`sidebar-tab ${activeTab === "projects" ? "active" : ""}`}
          onClick={() => {
            setActiveTab("projects");
            setOpenProjectId(null);
          }}
        >
          Projects
        </button>
      </div>

      <div className="sidebar-scroll">
        {activeTab === "chats" && (
          <>
            <p className="sidebar-section-label">Saved research threads</p>
            {!conversations.length ? (
              <div className="sidebar-empty">No saved chats yet.</div>
            ) : (
              <div className="sidebar-list">{conversations.map(renderChatRow)}</div>
            )}
          </>
        )}

        {activeTab === "projects" && !openProject && (
          <>
            <div className="sidebar-section-row">
              <p className="sidebar-section-label">Your projects</p>
              <button
                type="button"
                className="text-btn"
                onClick={() =>
                  askName({
                    title: "Name your project",
                    confirmLabel: "Create project",
                    onConfirm: (name) => onCreateProject(name),
                  })
                }
              >
                + New
              </button>
            </div>
            {!projects.length ? (
              <div className="sidebar-empty">
                No projects yet. Group related chats into a project from a chat's ··· menu.
              </div>
            ) : (
              <div className="sidebar-list">
                {projects.map((project) => (
                  <div key={project.project_id} className="sidebar-item">
                    <button
                      type="button"
                      className="sidebar-item-button project-row"
                      onClick={() => setOpenProjectId(project.project_id)}
                    >
                      <span className="project-row-icon">
                        <FolderIcon />
                      </span>
                      <span className="project-row-text">
                        <span className="sidebar-item-title">{project.name}</span>
                        <span className="sidebar-item-meta">
                          {project.conversation_count || 0}{" "}
                          {project.conversation_count === 1 ? "chat" : "chats"}
                        </span>
                      </span>
                    </button>
                    <div className="sidebar-item-actions">
                      <button
                        type="button"
                        className="icon-btn dots-btn"
                        aria-label="Project options"
                        title="More"
                        onClick={(event) => openMenu(event, "project", project.project_id)}
                      >
                        <DotsIcon />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        {activeTab === "projects" && openProject && (
          <>
            <button type="button" className="project-back" onClick={() => setOpenProjectId(null)}>
              <ChevronLeft />
              All projects
            </button>
            <div className="project-detail-head">
              <h3 className="project-detail-title">{openProject.name}</h3>
              <button
                type="button"
                className="text-btn"
                onClick={() =>
                  askName({
                    title: "Rename project",
                    confirmLabel: "Save",
                    initialValue: openProject.name,
                    onConfirm: (name) => onRenameProject(openProject.project_id, name),
                  })
                }
              >
                Rename
              </button>
            </div>
            {(() => {
              const inProject = conversations.filter(
                (conversation) => conversation.project_id === openProject.project_id,
              );
              if (!inProject.length) {
                return <div className="sidebar-empty">No chats in this project yet.</div>;
              }
              return <div className="sidebar-list">{inProject.map(renderChatRow)}</div>;
            })()}
          </>
        )}
      </div>

      <div className="sidebar-footer">
        {userEmail ? <span className="sidebar-user" title={userEmail}>{userEmail}</span> : <span />}
        {canSignOut ? (
          <button type="button" className="text-btn" onClick={onSignOut}>
            Sign out
          </button>
        ) : null}
      </div>

      {menu ? (
        <div
          className="menu-pop"
          style={{ left: menu.x, top: menu.y, width: MENU_WIDTH }}
          onClick={(event) => event.stopPropagation()}
          role="menu"
        >
          {menu.kind === "chat" && menu.view === "root" && (
            <>
              <button
                type="button"
                className="menu-item"
                onClick={() => {
                  onTogglePin(menu.id, !menuConversation?.pinned);
                  closeMenu();
                }}
              >
                <PinIcon filled={menuConversation?.pinned} />
                {menuConversation?.pinned ? "Unpin" : "Pin"}
              </button>
              <button
                type="button"
                className="menu-item"
                onClick={() => setMenu((current) => ({ ...current, view: "projects" }))}
              >
                <FolderIcon />
                Add to project
                <span className="menu-chevron">›</span>
              </button>
              {menuConversation?.project_id ? (
                <button
                  type="button"
                  className="menu-item"
                  onClick={() => {
                    onAssignToProject(menu.id, null);
                    closeMenu();
                  }}
                >
                  Remove from project
                </button>
              ) : null}
              <div className="menu-divider" />
              <button
                type="button"
                className="menu-item danger"
                onClick={() => {
                  onDelete(menu.id);
                  closeMenu();
                }}
              >
                Delete chat
              </button>
            </>
          )}

          {menu.kind === "chat" && menu.view === "projects" && (
            <>
              <button
                type="button"
                className="menu-item menu-back"
                onClick={() => setMenu((current) => ({ ...current, view: "root" }))}
              >
                <ChevronLeft />
                Add to project
              </button>
              <div className="menu-divider" />
              <div className="menu-scroll">
                {projects.length ? (
                  projects.map((project) => (
                    <button
                      type="button"
                      key={project.project_id}
                      className="menu-item"
                      onClick={() => {
                        onAssignToProject(menu.id, project.project_id);
                        closeMenu();
                      }}
                    >
                      <FolderIcon />
                      <span className="menu-item-label">{project.name}</span>
                    </button>
                  ))
                ) : (
                  <div className="menu-note">No projects yet.</div>
                )}
              </div>
              <div className="menu-divider" />
              <button
                type="button"
                className="menu-item"
                onClick={() => {
                  const conversationId = menu.id;
                  closeMenu();
                  newProjectFor(conversationId);
                }}
              >
                <span className="plus" aria-hidden="true">+</span>
                New project…
              </button>
            </>
          )}

          {menu.kind === "project" && (
            <>
              <button
                type="button"
                className="menu-item"
                onClick={() => {
                  const project = projects.find((item) => item.project_id === menu.id);
                  closeMenu();
                  askName({
                    title: "Rename project",
                    confirmLabel: "Save",
                    initialValue: project?.name || "",
                    onConfirm: (name) => onRenameProject(menu.id, name),
                  });
                }}
              >
                Rename project
              </button>
              <div className="menu-divider" />
              <button
                type="button"
                className="menu-item danger"
                onClick={() => {
                  const projectId = menu.id;
                  closeMenu();
                  if (openProjectId === projectId) setOpenProjectId(null);
                  onDeleteProject(projectId);
                }}
              >
                Delete project
              </button>
            </>
          )}
        </div>
      ) : null}

      {prompt ? (
        <NamePrompt
          title={prompt.title}
          confirmLabel={prompt.confirmLabel}
          initialValue={prompt.value}
          onCancel={() => setPrompt(null)}
          onSubmit={async (name) => {
            const trimmed = name.trim();
            if (!trimmed) return;
            setPrompt(null);
            await prompt.onConfirm(trimmed);
          }}
        />
      ) : null}
    </aside>
  );
}

function NamePrompt({ title, confirmLabel, initialValue, onCancel, onSubmit }) {
  const [value, setValue] = useState(initialValue || "");
  const inputRef = useRef(null);

  useEffect(() => {
    inputRef.current?.focus();
    inputRef.current?.select();
  }, []);

  return (
    <div className="prompt-overlay" onClick={onCancel}>
      <form
        className="prompt-card"
        onClick={(event) => event.stopPropagation()}
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit(value);
        }}
      >
        <h3 className="prompt-title">{title}</h3>
        <input
          ref={inputRef}
          className="prompt-input"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          placeholder="Project name"
          maxLength={80}
        />
        <div className="prompt-actions">
          <button type="button" className="btn-ghost" onClick={onCancel}>
            Cancel
          </button>
          <button type="submit" className="btn-primary" disabled={!value.trim()}>
            {confirmLabel}
          </button>
        </div>
      </form>
    </div>
  );
}
