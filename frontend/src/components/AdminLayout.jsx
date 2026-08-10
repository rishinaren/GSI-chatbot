import { useEffect, useState } from "react";
import { BackIcon } from "./AdminIcons";

// The pieces every admin panel is built from. They live here rather than in
// AdminPortal so the panels do not have to import the shell that renders them,
// which would be a cycle.

/** Page header: title, one explanatory line, and an optional action.
 *
 * `onBack` is only passed when the reader arrived from somewhere they can be
 * returned to (a dashboard shortcut), so a section opened straight from the nav
 * does not grow a Back button pointing nowhere.
 */
export function PageHead({ head, sub, action, onBack }) {
  return (
    <header className="admin-page-head">
      <div>
        {onBack ? (
          <button type="button" className="admin-back" onClick={onBack}>
            <BackIcon />
            Back
          </button>
        ) : null}
        <h1 className="admin-page-title">{head.title}</h1>
        <p className="admin-page-sub">{sub ?? head.subtitle}</p>
      </div>
      {action ? <div className="admin-page-action">{action}</div> : null}
    </header>
  );
}

/** A titled block of content. `note` sits under the title, in muted text. */
export function Card({ title, note, action, children, className = "" }) {
  return (
    <section className={`admin-card ${className}`}>
      {title ? (
        <div className="admin-card-head">
          <div>
            <h2 className="admin-card-title">{title}</h2>
            {note ? <p className="admin-card-note">{note}</p> : null}
          </div>
          {action ? <div className="admin-card-action">{action}</div> : null}
        </div>
      ) : null}
      {children}
    </section>
  );
}

/** Load-once fetch with the loading/error states every panel needs. */
export function useAdminData(loader, fallbackMessage) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    loader()
      .then((result) => {
        if (cancelled) return;
        setData(result);
        setError("");
      })
      .catch((loadError) => {
        if (cancelled) return;
        setError(loadError instanceof Error ? loadError.message : fallbackMessage);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // Panels load once when they mount; switching sections remounts them.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { data, setData, loading, error, setError };
}
