function PinIcon({ filled }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" width="16" height="16">
      <path
        d="M14 3l7 7-3 .5-3.5 3.5L13 19l-2-2-3.5 3.5L6 22l1.5-1.5L11 17l-2-2 2-1.5L14.5 10 15 6.5 14 3z"
        fill={filled ? "currentColor" : "none"}
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function ChatSidebar({
  conversations,
  activeConversationId,
  onSelect,
  onNewChat,
  onDelete,
  onTogglePin,
}) {
  return (
    <aside className="chat-sidebar">
      <div className="sidebar-header">
        <h2>Chats</h2>
        <p>Saved research threads</p>
      </div>
      <button type="button" className="sidebar-new-chat" onClick={onNewChat}>
        + New chat
      </button>
      {!conversations.length ? (
        <div className="sidebar-empty">No saved chats yet.</div>
      ) : (
        <div className="sidebar-list">
          {conversations.map((conversation) => (
            <div
              key={conversation.conversation_id}
              className={`sidebar-item ${
                conversation.conversation_id === activeConversationId ? "active" : ""
              } ${conversation.pinned ? "pinned" : ""}`}
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
                  className={`sidebar-pin ${conversation.pinned ? "active" : ""}`}
                  aria-label={conversation.pinned ? "Unpin chat" : "Pin chat"}
                  title={conversation.pinned ? "Unpin chat" : "Pin chat"}
                  onClick={() => onTogglePin(conversation.conversation_id, !conversation.pinned)}
                >
                  <PinIcon filled={conversation.pinned} />
                </button>
                <button
                  type="button"
                  className="sidebar-delete"
                  aria-label="Delete chat"
                  onClick={() => onDelete(conversation.conversation_id)}
                >
                  ×
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </aside>
  );
}
