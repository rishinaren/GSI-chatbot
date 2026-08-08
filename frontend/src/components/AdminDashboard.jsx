import { getAdminOverview } from "../api";
import { Card, PageHead, useAdminData } from "./AdminLayout";
import { DocIcon } from "./AdminIcons";

// The landing page answers two questions and nothing else: how much does the
// assistant know, and is anything broken. Every number here is a count of
// something a person can go and look at in another section, so each block links
// to the section that explains it.

// Matches the wording used for chat recency in the sidebar.
function relativeDays(iso) {
  if (!iso) return "";
  const then = new Date(iso);
  if (Number.isNaN(then.getTime())) return "";
  const startOfDay = (d) => new Date(d.getFullYear(), d.getMonth(), d.getDate());
  const days = Math.round((startOfDay(new Date()) - startOfDay(then)) / 86400000);
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 30) return `${days} days ago`;
  return then.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

const STATE_WORD = { ok: "Working", warn: "Check this", off: "Off" };

function Stat({ value, label, note }) {
  return (
    <div className="admin-stat">
      <span className="admin-stat-value">{value}</span>
      <span className="admin-stat-label">{label}</span>
      {note ? <span className="admin-stat-note">{note}</span> : null}
    </div>
  );
}

export default function AdminDashboard({ head, onNavigate }) {
  const { data, loading, error } = useAdminData(
    getAdminOverview,
    "Could not load the dashboard.",
  );

  if (loading) {
    return (
      <>
        <PageHead head={head} />
        <p className="library-empty">Loading…</p>
      </>
    );
  }

  if (error || !data) {
    return (
      <>
        <PageHead head={head} />
        <div className="library-error">{error || "Could not load the dashboard."}</div>
      </>
    );
  }

  const totals = data.totals ?? {};
  const publishers = data.by_publisher ?? [];
  const biggest = Math.max(1, ...publishers.map((entry) => entry.count));
  const uploads = data.recent_uploads ?? [];
  const count = (value) => (value ?? 0).toLocaleString();

  return (
    <>
      <PageHead
        head={head}
        action={
          <button
            type="button"
            className="library-primary"
            onClick={() => onNavigate("documents", "add")}
          >
            <span aria-hidden="true">+</span> Add a document
          </button>
        }
      />

      <div className="admin-stats">
        <Stat
          value={count(totals.documents)}
          label="Documents ingested"
          note={`${count(totals.pages)} pages in total`}
        />
        <Stat
          value={count(totals.sections)}
          label="Indexed sections"
          note="Searchable chunks the assistant reads"
        />
        <Stat
          value={count(totals.linked_videos)}
          label="Linked videos"
          note={
            totals.videos === totals.linked_videos
              ? `All ${count(totals.videos)} videos`
              : `of ${count(totals.videos)} videos`
          }
        />
        <Stat
          value={count(totals.uploads)}
          label="Added here"
          note="Uploaded through this portal"
        />
      </div>

      <Card
        title="Where they come from"
        note="Every document in the library, by the body that publishes it."
        action={
          <button type="button" className="library-link" onClick={() => onNavigate("documents")}>
            See all documents
          </button>
        }
      >
        <div className="admin-bars">
          {publishers.map((entry) => (
            <div key={entry.name} className="admin-bar-row">
              <span className="admin-bar-name">{entry.name}</span>
              <span className="admin-bar-track">
                <span
                  className="admin-bar-fill"
                  style={{ width: `${Math.max(2, (entry.count / biggest) * 100)}%` }}
                />
              </span>
              <span className="admin-bar-value">{count(entry.count)}</span>
            </div>
          ))}
          {publishers.length === 0 ? (
            <p className="library-empty">There are no documents yet.</p>
          ) : null}
        </div>
      </Card>

      <Card
        title="Recently added"
        note="Documents uploaded through this portal, newest first."
        action={
          uploads.length ? (
            <button type="button" className="library-link" onClick={() => onNavigate("documents")}>
              See all documents
            </button>
          ) : null
        }
      >
        {uploads.length === 0 ? (
          <p className="admin-empty">
            Nothing has been uploaded here yet. The {count(totals.documents)} standards in the
            library were loaded when the assistant was set up - add one and it will appear here.
          </p>
        ) : (
          <div className="admin-recent">
            {uploads.map((row) => (
              <div key={row.document_id} className="library-row">
                <span className="library-row-icon" aria-hidden="true">
                  <DocIcon />
                </span>
                <div className="library-row-main">
                  <div className="library-row-top">
                    <span className="library-row-id">{row.standard_id}</span>
                    <span className={`library-pill ${(row.issuing_body || "").toLowerCase()}`}>
                      {row.issuing_body}
                    </span>
                  </div>
                  <p className="library-row-title">{row.title}</p>
                </div>
                <div className="library-row-meta">
                  <span>{relativeDays(row.added_at)}</span>
                  {row.added_by ? <span>{row.added_by}</span> : null}
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>

      <Card title="System status" note="What is running behind the assistant right now.">
        <div className="admin-status">
          {(data.status ?? []).map((row) => (
            <div key={row.key} className="admin-status-row">
              {/* The dot is never the only signal - the state is also spelled out
                  next to the label, so it reads without relying on colour. */}
              <span className={`admin-status-dot ${row.state}`} aria-hidden="true" />
              <div className="admin-status-main">
                <div className="admin-status-top">
                  <span className="admin-status-label">{row.label}</span>
                  <span className={`admin-status-state ${row.state}`}>
                    {STATE_WORD[row.state] ?? row.state}
                  </span>
                </div>
                <p className="admin-status-detail">{row.detail}</p>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </>
  );
}
