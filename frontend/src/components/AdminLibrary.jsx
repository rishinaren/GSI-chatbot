import { useEffect, useMemo, useRef, useState } from "react";
import { addLibraryDocument, analyzeLibraryDocument, listLibraryDocuments } from "../api";

// The people using this screen manage standards, not software. Every label here
// is deliberately in their words: "documents" and "sections", never chunks,
// embeddings or indexes; and nothing is written until they have seen what we
// read off the PDF and confirmed it.

const PUBLISHERS = ["ASTM", "GRI", "ISO", "Other"];
const MAX_MB = 40;

function CloseIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
      <path d="M6 6l12 12M18 6L6 18" />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="11" cy="11" r="7" />
      <path d="M20 20l-3.2-3.2" />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 16V4M8 8l4-4 4 4" />
      <path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M20 6L9 17l-5-5" />
    </svg>
  );
}

function DocIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
      <path d="M14 3v5h5" />
    </svg>
  );
}

function formatDate(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

function formatSize(bytes) {
  if (!bytes) return "";
  const mb = bytes / (1024 * 1024);
  return mb >= 1 ? `${mb.toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

export default function AdminLibrary({ onClose }) {
  const [view, setView] = useState("browse");
  const [documents, setDocuments] = useState([]);
  const [summary, setSummary] = useState({ document_count: 0, section_count: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [publisher, setPublisher] = useState("All");

  useEffect(() => {
    void refresh();
  }, []);

  useEffect(() => {
    const onKey = (event) => {
      if (event.key === "Escape" && view === "browse") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, view]);

  async function refresh() {
    setLoading(true);
    try {
      const data = await listLibraryDocuments();
      setDocuments(data.documents ?? []);
      setSummary({
        document_count: data.document_count ?? 0,
        section_count: data.section_count ?? 0,
        by_publisher: data.by_publisher ?? {},
      });
      setError("");
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Could not load the library.");
    } finally {
      setLoading(false);
    }
  }

  const publishers = useMemo(() => {
    const found = Object.keys(summary.by_publisher || {});
    return ["All", ...found];
  }, [summary]);

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return documents.filter((doc) => {
      if (publisher !== "All" && doc.issuing_body !== publisher) return false;
      if (!needle) return true;
      return (
        (doc.standard_id || "").toLowerCase().includes(needle) ||
        (doc.title || "").toLowerCase().includes(needle)
      );
    });
  }, [documents, query, publisher]);

  return (
    <div className="library-overlay" onClick={view === "browse" ? onClose : undefined}>
      <div
        className="library-shell"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-label="Document library"
      >
        {view === "browse" ? (
          <>
            <header className="library-head with-action">
              <div className="library-head-text">
                <h2 className="library-title">Document library</h2>
                <p className="library-sub">
                  {loading
                    ? "Loading…"
                    : `The ${summary.document_count} standards the assistant can read and quote from.`}
                </p>
              </div>
              <div className="library-head-actions">
                <button type="button" className="library-primary" onClick={() => setView("add")}>
                  <span aria-hidden="true">+</span> Add a document
                </button>
                <button type="button" className="library-close" onClick={onClose} aria-label="Close">
                  <CloseIcon />
                </button>
              </div>
            </header>

            <div className="library-toolbar">
              <div className="library-search">
                <SearchIcon />
                <input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Search by number or title"
                  aria-label="Search the library"
                />
              </div>
              <div className="library-chips">
                {publishers.map((name) => (
                  <button
                    key={name}
                    type="button"
                    className={`library-chip ${publisher === name ? "active" : ""}`}
                    onClick={() => setPublisher(name)}
                  >
                    {name}
                    {name !== "All" ? (
                      <span className="library-chip-count">{summary.by_publisher?.[name] ?? 0}</span>
                    ) : null}
                  </button>
                ))}
              </div>
            </div>

            {error ? <div className="library-error">{error}</div> : null}

            <div className="library-list">
              {!loading && visible.length === 0 ? (
                <p className="library-empty">
                  {documents.length === 0
                    ? "There are no documents yet. Add the first one."
                    : "No documents match what you typed."}
                </p>
              ) : null}

              {visible.map((doc) => (
                <div key={doc.document_id} className="library-row">
                  <span className="library-row-icon" aria-hidden="true">
                    <DocIcon />
                  </span>
                  <div className="library-row-main">
                    <div className="library-row-top">
                      <span className="library-row-id">{doc.standard_id}</span>
                      <span className={`library-pill ${doc.issuing_body.toLowerCase()}`}>
                        {doc.issuing_body}
                      </span>
                      {doc.uploaded ? <span className="library-pill added">Added here</span> : null}
                    </div>
                    <p className="library-row-title">{doc.title}</p>
                  </div>
                  <div className="library-row-meta">
                    <span>{doc.page_count ? `${doc.page_count} pages` : "-"}</span>
                    {doc.added_at ? <span>{formatDate(doc.added_at)}</span> : null}
                  </div>
                </div>
              ))}
            </div>

            <footer className="library-foot">
              Everything here is searched every time someone asks a question.
            </footer>
          </>
        ) : (
          <AddDocument
            onCancel={() => setView("browse")}
            onFinished={async () => {
              await refresh();
              setView("browse");
            }}
            onAdded={refresh}
          />
        )}
      </div>
    </div>
  );
}

function AddDocument({ onCancel, onFinished, onAdded }) {
  const [step, setStep] = useState("choose");
  const [file, setFile] = useState(null);
  const [details, setDetails] = useState(null);
  const [form, setForm] = useState({ standardId: "", title: "", issuingBody: "ASTM", year: "" });
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  async function handleFile(chosen) {
    if (!chosen) return;
    setError("");
    if (!/\.pdf$/i.test(chosen.name) && chosen.type !== "application/pdf") {
      setError(`${chosen.name} is not a PDF. Please choose the PDF version of the standard.`);
      return;
    }
    setFile(chosen);
    setStep("reading");
    try {
      const preview = await analyzeLibraryDocument(chosen);
      setDetails(preview);
      setForm({
        standardId: preview.standard_id || "",
        title: preview.title || "",
        issuingBody: PUBLISHERS.includes(preview.issuing_body) ? preview.issuing_body : "Other",
        year: preview.year ? String(preview.year) : "",
      });
      setStep("confirm");
    } catch (readError) {
      setError(readError instanceof Error ? readError.message : "We could not read that PDF.");
      setStep("choose");
      setFile(null);
    }
  }

  async function submit(event) {
    event.preventDefault();
    setError("");
    setStep("saving");
    try {
      const added = await addLibraryDocument({ file, ...form });
      setResult(added);
      setStep("done");
      void onAdded();
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "We could not add that document.");
      setStep("confirm");
    }
  }

  function startOver() {
    setFile(null);
    setDetails(null);
    setResult(null);
    setError("");
    setStep("choose");
  }

  return (
    <div className="library-add">
      <header className="library-head">
        <div className="library-head-text">
          <h2 className="library-title">
            {step === "done" ? "That's done" : "Add a document"}
          </h2>
          <p className="library-sub">
            {step === "choose" || step === "reading"
              ? "Step 1 of 2 - choose the PDF of the standard."
              : step === "confirm"
                ? "Step 2 of 2 - check the details we read off the front page."
                : step === "saving"
                  ? "Adding it now."
                  : "The assistant can use it straight away."}
          </p>
        </div>
        <div className="library-head-actions">
          <button
            type="button"
            className="library-close"
            onClick={step === "done" ? onFinished : onCancel}
            aria-label="Close"
          >
            <CloseIcon />
          </button>
        </div>
      </header>

      {error ? <div className="library-error">{error}</div> : null}

      {step === "choose" || step === "reading" ? (
        <div className="library-step">
          <button
            type="button"
            className={`library-drop ${dragging ? "dragging" : ""} ${step === "reading" ? "busy" : ""}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              void handleFile(event.dataTransfer.files?.[0]);
            }}
            disabled={step === "reading"}
          >
            {step === "reading" ? (
              <>
                <span className="library-spinner" aria-hidden="true" />
                <span className="library-drop-title">Reading {file?.name}…</span>
                <span className="library-drop-hint">Just a moment.</span>
              </>
            ) : (
              <>
                <UploadIcon />
                <span className="library-drop-title">Drag the PDF here</span>
                <span className="library-drop-hint">or click to choose a file from your computer</span>
                <span className="library-drop-note">PDF only, up to {MAX_MB} MB</span>
              </>
            )}
          </button>
          <input
            ref={inputRef}
            type="file"
            accept="application/pdf,.pdf"
            hidden
            onChange={(event) => {
              void handleFile(event.target.files?.[0]);
              event.target.value = "";
            }}
          />
          <p className="library-help">
            One standard per file. If the PDF is a scan, the words need to be selectable - otherwise
            there is nothing for the assistant to read.
          </p>
          <div className="library-actions">
            <button type="button" className="library-ghost" onClick={onCancel}>
              Back to the library
            </button>
          </div>
        </div>
      ) : null}

      {step === "confirm" && details ? (
        <form className="library-step" onSubmit={submit}>
          <div className="library-file-strip">
            <DocIcon />
            <span className="library-file-name">{details.file_name}</span>
            <span className="library-file-meta">
              {details.page_count} pages · {formatSize(details.size_bytes)}
            </span>
            <button type="button" className="library-link" onClick={startOver}>
              Choose a different file
            </button>
          </div>

          {details.already_in_library ? (
            <div className="library-warn">
              <strong>{details.already_in_library.standard_id}</strong> is already in the library.
              Adding this file will replace the copy that is there now.
            </div>
          ) : (
            <div className="library-notice">
              We read the PDF and filled these in for you. Change anything that looks wrong.
            </div>
          )}

          <div className="library-field">
            <label htmlFor="lib-standard">Standard number</label>
            <input
              id="lib-standard"
              value={form.standardId}
              onChange={(event) => setForm({ ...form, standardId: event.target.value })}
              placeholder="D4595-24"
              required
            />
            <span className="library-field-help">
              The code people use to refer to it, like D4595-24 or GRI-GM13r.
            </span>
          </div>

          <div className="library-field">
            <label htmlFor="lib-title">Title</label>
            <input
              id="lib-title"
              value={form.title}
              onChange={(event) => setForm({ ...form, title: event.target.value })}
              placeholder="Standard Test Method for…"
              required
            />
            <span className="library-field-help">This is what people will see next to answers.</span>
          </div>

          <div className="library-field-row">
            <div className="library-field">
              <label htmlFor="lib-publisher">Who publishes it?</label>
              <select
                id="lib-publisher"
                value={form.issuingBody}
                onChange={(event) => setForm({ ...form, issuingBody: event.target.value })}
              >
                {PUBLISHERS.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </div>
            <div className="library-field">
              <label htmlFor="lib-year">Year (optional)</label>
              <input
                id="lib-year"
                value={form.year}
                onChange={(event) => setForm({ ...form, year: event.target.value })}
                placeholder="2024"
                inputMode="numeric"
              />
            </div>
          </div>

          {details.text_preview ? (
            <details className="library-preview">
              <summary>See the first few lines we read, to be sure it is the right file</summary>
              <p>{details.text_preview}</p>
            </details>
          ) : null}

          <div className="library-actions">
            <button type="button" className="library-ghost" onClick={startOver}>
              Back
            </button>
            <button type="submit" className="library-primary">
              {details.already_in_library ? "Replace it in the library" : "Add to the library"}
            </button>
          </div>
        </form>
      ) : null}

      {step === "saving" ? (
        <div className="library-step library-centered">
          <span className="library-spinner large" aria-hidden="true" />
          <p className="library-saving-title">Adding {form.standardId} to the library</p>
          <p className="library-help centered">
            We are reading every page, teaching the assistant what is in them, and saving your copy.
            This usually takes less than a minute - please keep this window open.
          </p>
        </div>
      ) : null}

      {step === "done" && result ? (
        <div className="library-step library-centered">
          <span className="library-done-badge" aria-hidden="true">
            <CheckIcon />
          </span>
          <p className="library-saving-title">
            {result.standard_id} is {result.replaced ? "updated" : "in the library"}
          </p>
          <p className="library-help centered">
            The assistant read {result.page_count} pages and can quote them from the next question
            on. Try asking about {result.standard_id} to see it work.
          </p>
          {result.warning ? <div className="library-warn">{result.warning}</div> : null}
          <div className="library-actions centered">
            <button type="button" className="library-ghost" onClick={startOver}>
              Add another
            </button>
            <button type="button" className="library-primary" onClick={onFinished}>
              Done
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
