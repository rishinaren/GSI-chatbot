import { useEffect, useId, useState } from "react";

const OPTIONS = [
  { value: "testing", label: "Testing with standards", icon: "testing" },
  { value: "design", label: "Designing with geosynthetics", icon: "design" },
  { value: "both", label: "Both testing and design", icon: "both" },
];

function FocusIcon({ name }) {
  if (name === "testing") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M9 3h6M10 3v5l-4.4 8.2A3.2 3.2 0 0 0 8.4 21h7.2a3.2 3.2 0 0 0 2.8-4.8L14 8V3" />
        <path d="M7.3 14h9.4" />
      </svg>
    );
  }
  if (name === "design") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="12" cy="5" r="2" />
        <path d="M12 7v4M12 11L5 20M12 11l7 9M7.7 16h8.6" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3l8 4-8 4-8-4zM4 12l8 4 8-4M4 17l8 4 8-4" />
    </svg>
  );
}

export default function ChatPreferencesModal({ initialFocus, onboarding, onSave, onClose }) {
  const titleId = useId();
  const [focus, setFocus] = useState(initialFocus || "both");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setFocus(initialFocus || "both");
  }, [initialFocus]);

  async function handleSubmit(event) {
    event.preventDefault();
    if (saving) return;
    setSaving(true);
    setError("");
    try {
      await onSave(focus);
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Could not save your preference.");
    } finally {
      setSaving(false);
    }
  }

  const selectedLabel = focus === "testing" ? "Testing" : focus === "design" ? "Design" : "Both";

  return (
    <div className="preference-overlay" role="dialog" aria-modal="true" aria-labelledby={titleId}>
      <form className="preference-card" onSubmit={handleSubmit}>
        {!onboarding ? (
          <button type="button" className="preference-close" onClick={onClose} aria-label="Close settings">
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M6 6l12 12M18 6L6 18" />
            </svg>
          </button>
        ) : null}

        <div className="preference-brand">
          <span className="sidebar-brand-dot" aria-hidden="true" />
          GSI Chatbot
        </div>
        <p className="preference-eyebrow">{onboarding ? "Set up your chat" : "Chat preferences"}</p>
        <h1 id={titleId}>{onboarding ? "What brings you to GSI?" : "Choose your default focus"}</h1>
        <p className="preference-subtitle">
          Choose how you plan to use the chatbot. You can change this anytime in Settings.
        </p>

        <fieldset className="preference-options">
          <legend className="sr-only">Chat focus</legend>
          {OPTIONS.map((option) => (
            <label
              key={option.value}
              className={`preference-option ${focus === option.value ? "selected" : ""}`}
            >
              <span className="preference-option-icon"><FocusIcon name={option.icon} /></span>
              <span>{option.label}</span>
              <input
                type="radio"
                name="chat-focus"
                value={option.value}
                checked={focus === option.value}
                onChange={() => setFocus(option.value)}
              />
            </label>
          ))}
        </fieldset>

        <p className="preference-selected" aria-live="polite">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 6L9 17l-5-5" /></svg>
          {selectedLabel} selected.
        </p>

        {error ? <p className="preference-error" role="alert">{error}</p> : null}
        <div className="preference-actions">
          {!onboarding ? (
            <button type="button" className="preference-cancel" onClick={onClose}>Cancel</button>
          ) : null}
          <button type="submit" className="preference-save" disabled={saving}>
            {saving ? "Saving…" : onboarding ? "Continue to chat" : "Save changes"}
          </button>
        </div>
      </form>
    </div>
  );
}
