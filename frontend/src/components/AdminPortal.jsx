import { useState } from "react";
import AdminDashboard from "./AdminDashboard";
import AdminDocuments from "./AdminDocuments";
import AdminVideos from "./AdminVideos";
import AdminAccess from "./AdminAccess";
import { BackIcon, DashboardIcon, LibraryIcon, PeopleIcon, VideoIcon } from "./AdminIcons";

// The admin portal replaces the chat shell rather than floating over it: an
// administrator here is looking after the knowledge base, so Chats and Projects
// - which belong to the person asking questions, not the person curating - are
// deliberately not on screen. "Back to chat" is the one way across.

const SECTIONS = [
  {
    key: "dashboard",
    label: "Dashboard",
    icon: DashboardIcon,
    title: "Dashboard",
    subtitle: "What the assistant knows, and whether anything needs looking at.",
  },
  {
    key: "documents",
    label: "Documents",
    icon: LibraryIcon,
    title: "Documents",
    subtitle: "The standards the assistant can read and quote from.",
  },
  {
    key: "videos",
    label: "Videos",
    icon: VideoIcon,
    title: "Videos",
    subtitle: "The walkthroughs offered alongside answers, and the standards they cover.",
  },
  {
    key: "access",
    label: "Access",
    icon: PeopleIcon,
    title: "Access",
    subtitle: "The people who can add and replace what the assistant reads.",
  },
];

export default function AdminPortal({ email, onExit, onSignOut }) {
  const [section, setSection] = useState("dashboard");
  // Set by the dashboard's shortcuts so "Add a document" lands on the upload
  // step instead of the list the person would then have to click through.
  const [documentsView, setDocumentsView] = useState("browse");

  function go(key, view = "browse") {
    setDocumentsView(view);
    setSection(key);
  }

  const active = SECTIONS.find((entry) => entry.key === section) ?? SECTIONS[0];

  return (
    <div className="admin-shell">
      <nav className="admin-nav" aria-label="Admin sections">
        <div className="admin-nav-brand">
          <span className="sidebar-brand-dot" aria-hidden="true" />
          <span>
            GSI <span className="admin-nav-brand-tag">Admin</span>
          </span>
        </div>

        <div className="admin-nav-list" role="tablist" aria-orientation="vertical">
          {SECTIONS.map((entry) => {
            const Icon = entry.icon;
            return (
              <button
                key={entry.key}
                type="button"
                role="tab"
                aria-selected={section === entry.key}
                className={`admin-nav-item ${section === entry.key ? "active" : ""}`}
                onClick={() => go(entry.key)}
              >
                <Icon />
                {entry.label}
              </button>
            );
          })}
        </div>

        <div className="admin-nav-foot">
          <button type="button" className="admin-nav-back" onClick={onExit}>
            <BackIcon />
            Back to chat
          </button>
          <div className="admin-nav-user">
            <span className="admin-nav-email" title={email}>
              {email}
            </span>
            {onSignOut ? (
              <button type="button" className="library-link" onClick={onSignOut}>
                Sign out
              </button>
            ) : null}
          </div>
        </div>
      </nav>

      <main className="admin-main">
        <div className="admin-page">
          {section === "dashboard" ? (
            <AdminDashboard head={active} onNavigate={go} />
          ) : section === "documents" ? (
            <AdminDocuments
              head={active}
              view={documentsView}
              onViewChange={setDocumentsView}
            />
          ) : section === "videos" ? (
            <AdminVideos head={active} />
          ) : (
            <AdminAccess head={active} />
          )}
        </div>
      </main>
    </div>
  );
}
