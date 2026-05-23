export default function ChatSidebar({
  conversations,
  activeConversationId,
  onSelect,
  onNewChat,
  onDelete,
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
              className={`sidebar-item ${conversation.conversation_id === activeConversationId ? "active" : ""}`}
            >
              <button
                type="button"
                className="sidebar-item-button"
                onClick={() => onSelect(conversation.conversation_id)}
              >
                <span className="sidebar-item-title">{conversation.title}</span>
                <span className="sidebar-item-meta">{conversation.message_count || 0} messages</span>
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
          ))}
        </div>
      )}
    </aside>
  );
}
