import { useEffect, useState } from "react";
import { grantLibraryAccess, listLibraryPeople, revokeLibraryAccess } from "../api";
import { Card, PageHead } from "./AdminLayout";
import { PersonIcon, formatDate } from "./AdminIcons";

// Granting access is not account creation: the API only accepts an email that
// already belongs to a subscriber or GSI member, so the copy here sets that
// expectation before someone types a colleague's address and expects an invite.

export default function AdminAccess({ head }) {
  const [people, setPeople] = useState([]);
  const [you, setYou] = useState("");
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    void refresh();
  }, []);

  async function refresh() {
    setLoading(true);
    try {
      const data = await listLibraryPeople();
      setPeople(data.people ?? []);
      setYou(data.you ?? "");
      setError("");
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Could not load the list.");
    } finally {
      setLoading(false);
    }
  }

  async function submit(event) {
    event.preventDefault();
    const address = email.trim();
    if (!address) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const granted = await grantLibraryAccess(address);
      setNotice(`${granted.email} can now manage the library.`);
      setEmail("");
      await refresh();
    } catch (grantError) {
      setError(grantError instanceof Error ? grantError.message : "Could not give access.");
    } finally {
      setBusy(false);
    }
  }

  async function remove(address) {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await revokeLibraryAccess(address);
      setNotice(`${address} no longer has access.`);
      await refresh();
    } catch (revokeError) {
      setError(revokeError instanceof Error ? revokeError.message : "Could not remove access.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageHead head={head} />

      <Card title="Give someone access">
        <form className="library-invite" onSubmit={submit}>
          <div className="library-invite-row">
            <input
              id="lib-invite"
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="their.email@example.com"
              aria-label="Email address"
              disabled={busy}
            />
            <button type="submit" className="library-primary" disabled={busy || !email.trim()}>
              {busy ? "Checking…" : "Give access"}
            </button>
          </div>
          <span className="library-field-help">
            Note: They need to have signed in to the chatbot at least once already.
          </span>
        </form>

        {error ? <div className="library-error inline">{error}</div> : null}
        {notice ? <div className="library-notice">{notice}</div> : null}
      </Card>

      <Card
        title="Who can manage this"
        note="Anyone listed here can add documents the assistant will quote."
      >
        <div className="library-people-list">
          {loading ? <p className="library-empty">Loading…</p> : null}
          {people.map((person) => (
            <div key={person.email} className="library-person">
              <span className="library-row-icon" aria-hidden="true">
                <PersonIcon />
              </span>
              <div className="library-person-main">
                <span className="library-person-email">
                  {person.email}
                  {person.email === you ? <span className="library-pill added">You</span> : null}
                </span>
                {person.source === "root" ? null : (
                  <p className="library-person-note">
                    Added{person.granted_by ? ` by ${person.granted_by}` : ""}
                    {person.granted_at ? ` on ${formatDate(person.granted_at)}` : ""}
                  </p>
                )}
              </div>
              {person.removable && person.email !== you ? (
                <button
                  type="button"
                  className="library-remove"
                  onClick={() => remove(person.email)}
                  disabled={busy}
                >
                  Remove
                </button>
              ) : null}
            </div>
          ))}
        </div>
      </Card>
    </>
  );
}
