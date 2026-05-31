import { useState } from "react";
import { signIn } from "../auth";

function EyeIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M2 12c1.8-3.5 5.4-6 10-6s8.2 2.5 10 6c-1.8 3.5-5.4 6-10 6s-8.2-2.5-10-6Z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="12" cy="12" r="3.1" fill="none" stroke="currentColor" strokeWidth="1.8" />
    </svg>
  );
}

function EyeOffIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M2 12c1.8-3.5 5.4-6 10-6s8.2 2.5 10 6c-1.8 3.5-5.4 6-10 6s-8.2-2.5-10-6Z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="12" cy="12" r="3.1" fill="none" stroke="currentColor" strokeWidth="1.8" />
      <path
        d="M3 3l18 18"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function friendlyLoginError(rawMessage) {
  const text = String(rawMessage || "");
  if (
    text.includes("Login failed. Check email and password.") ||
    text.includes("NotAuthorizedException") ||
    text.includes("Incorrect username or password")
  ) {
    return "That email or password is incorrect. Please try again.";
  }
  return "We couldn't sign you in. Please try again.";
}

export default function LoginPanel({ onSignedIn, connectionError = "" }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setIsSubmitting(true);
    try {
      await signIn(email.trim(), password);
      onSignedIn();
    } catch (submitError) {
      setError(friendlyLoginError(submitError instanceof Error ? submitError.message : ""));
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="login-panel">
      <div className="login-card">
        <h1>GSI Chatbot</h1>
        <p>Sign in to access the standards research assistant.</p>
        {connectionError ? <div className="login-error">{connectionError}</div> : null}
        <form onSubmit={handleSubmit} className="login-form">
          <label htmlFor="email">Email</label>
          <input
            id="email"
            type="email"
            autoComplete="username"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            required
          />
          <label htmlFor="password">Password</label>
          <div className="password-input-wrap">
            <input
              id="password"
              type={showPassword ? "text" : "password"}
              autoComplete="current-password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              required
            />
            <button
              type="button"
              className="password-toggle-btn"
              onClick={() => setShowPassword((current) => !current)}
              aria-label={showPassword ? "Hide password" : "Show password"}
              aria-pressed={showPassword}
            >
              {showPassword ? <EyeIcon /> : <EyeOffIcon />}
            </button>
          </div>
          {error ? <div className="login-error">{error}</div> : null}
          <button type="submit" className="login-submit-btn" disabled={isSubmitting}>
            {isSubmitting ? "Signing in..." : "Sign in"}
          </button>
        </form>
      </div>
    </div>
  );
}
