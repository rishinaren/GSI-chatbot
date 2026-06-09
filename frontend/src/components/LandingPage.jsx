const GSI_WEBSITE_URL = "https://geosynthetic-institute.org/";

export default function LandingPage() {
  return (
    <div className="landing-page">
      <main className="landing-card">
        <a
          className="landing-logo-link"
          href={GSI_WEBSITE_URL}
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Geosynthetic Institute website"
        >
          <img className="landing-logo" src="/gsi-logo.png" alt="Geosynthetic Institute logo" />
        </a>

        <div className="landing-copy">
          <p className="landing-eyebrow">Geosynthetic Institute</p>
          <h1>GSI Chatbot</h1>
          <p className="landing-lead">
            A members-only research assistant for querying standards, saving chats, and getting
            geosynthetics insight grounded in ASTM and ISO standards and reports.
          </p>
        </div>

        <a className="landing-cta" href="/app">
          Log in / Sign up
        </a>
      </main>
    </div>
  );
}
