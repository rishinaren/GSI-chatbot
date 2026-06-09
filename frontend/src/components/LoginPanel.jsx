import { useState } from "react";
import { confirmSignUp, resendConfirmationCode, signIn, signUp } from "../auth";

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

function friendlyAuthError(rawMessage) {
  const text = String(rawMessage || "");
  if (
    text.includes("Login failed. Check email and password.") ||
    text.includes("NotAuthorizedException") ||
    text.includes("Incorrect username or password") ||
    text.includes("That email or password is incorrect")
  ) {
    return "That email or password is incorrect. Please try again.";
  }
  if (text.includes("Confirm your email")) {
    return text;
  }
  return text || "Something went wrong. Please try again.";
}

function PasswordField({ id, label, value, onChange, autoComplete, showPassword, onToggle }) {
  return (
    <>
      <label htmlFor={id}>{label}</label>
      <div className="password-input-wrap">
        <input
          id={id}
          type={showPassword ? "text" : "password"}
          autoComplete={autoComplete}
          value={value}
          onChange={onChange}
          required
        />
        <button
          type="button"
          className="password-toggle-btn"
          onClick={onToggle}
          aria-label={showPassword ? "Hide password" : "Show password"}
          aria-pressed={showPassword}
        >
          {showPassword ? <EyeIcon /> : <EyeOffIcon />}
        </button>
      </div>
    </>
  );
}

export default function LoginPanel({ onSignedIn, connectionError = "" }) {
  const [mode, setMode] = useState("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [confirmationCode, setConfirmationCode] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [verificationDestination, setVerificationDestination] = useState("");

  function resetMessages() {
    setError("");
    setSuccess("");
  }

  function switchMode(nextMode) {
    resetMessages();
    setMode(nextMode);
    if (nextMode === "signin") {
      setConfirmationCode("");
      setConfirmPassword("");
    }
  }

  async function handleSignIn(event) {
    event.preventDefault();
    resetMessages();
    setIsSubmitting(true);
    try {
      await signIn(email.trim(), password);
      onSignedIn();
    } catch (submitError) {
      const message = friendlyAuthError(submitError instanceof Error ? submitError.message : "");
      if (message.includes("Confirm your email")) {
        setVerificationDestination(email.trim());
        setMode("confirm");
      }
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleSignUp(event) {
    event.preventDefault();
    resetMessages();
    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    setIsSubmitting(true);
    try {
      const result = await signUp(email.trim(), password);
      if (result.user_confirmed) {
        setSuccess("Account created. You can sign in now.");
        switchMode("signin");
        return;
      }
      setVerificationDestination(result.destination || email.trim());
      setSuccess(
        result.destination
          ? `We sent a verification code to ${result.destination}.`
          : "We sent a verification code to your email.",
      );
      setMode("confirm");
    } catch (submitError) {
      setError(friendlyAuthError(submitError instanceof Error ? submitError.message : ""));
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleConfirm(event) {
    event.preventDefault();
    resetMessages();
    setIsSubmitting(true);
    try {
      await confirmSignUp(email.trim(), confirmationCode.trim());
      setSuccess("Email confirmed. Sign in with your new account.");
      switchMode("signin");
    } catch (submitError) {
      setError(friendlyAuthError(submitError instanceof Error ? submitError.message : ""));
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleResendCode() {
    resetMessages();
    setIsSubmitting(true);
    try {
      const result = await resendConfirmationCode(email.trim());
      setSuccess(
        result.destination
          ? `We sent a new verification code to ${result.destination}.`
          : "We sent a new verification code to your email.",
      );
    } catch (submitError) {
      setError(friendlyAuthError(submitError instanceof Error ? submitError.message : ""));
    } finally {
      setIsSubmitting(false);
    }
  }

  const title =
    mode === "signup" ? "Create your account" : mode === "confirm" ? "Verify your email" : "GSI Chatbot";
  const subtitle =
    mode === "signup"
      ? "Sign up to access the standards research assistant."
      : mode === "confirm"
        ? `Enter the verification code we sent to ${verificationDestination || "your email"}.`
        : "Sign in to access the standards research assistant.";

  return (
    <div className="login-panel">
      <div className="login-card">
        <h1>{title}</h1>
        <p>{subtitle}</p>
        {connectionError ? <div className="login-error">{connectionError}</div> : null}
        {success ? <div className="login-success">{success}</div> : null}

        {mode === "signin" ? (
          <form onSubmit={handleSignIn} className="login-form">
            <label htmlFor="email">Email</label>
            <input
              id="email"
              type="email"
              autoComplete="username"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              required
            />
            <PasswordField
              id="password"
              label="Password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete="current-password"
              showPassword={showPassword}
              onToggle={() => setShowPassword((current) => !current)}
            />
            {error ? <div className="login-error">{error}</div> : null}
            <button type="submit" className="login-submit-btn" disabled={isSubmitting}>
              {isSubmitting ? "Signing in..." : "Sign in"}
            </button>
          </form>
        ) : null}

        {mode === "signup" ? (
          <form onSubmit={handleSignUp} className="login-form">
            <label htmlFor="signup-email">Email</label>
            <input
              id="signup-email"
              type="email"
              autoComplete="username"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              required
            />
            <PasswordField
              id="signup-password"
              label="Password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete="new-password"
              showPassword={showPassword}
              onToggle={() => setShowPassword((current) => !current)}
            />
            <p className="login-hint">
              Use at least 8 characters with uppercase, lowercase, numbers, and symbols.
            </p>
            <PasswordField
              id="signup-confirm-password"
              label="Confirm password"
              value={confirmPassword}
              onChange={(event) => setConfirmPassword(event.target.value)}
              autoComplete="new-password"
              showPassword={showConfirmPassword}
              onToggle={() => setShowConfirmPassword((current) => !current)}
            />
            {error ? <div className="login-error">{error}</div> : null}
            <button type="submit" className="login-submit-btn" disabled={isSubmitting}>
              {isSubmitting ? "Creating account..." : "Create account"}
            </button>
          </form>
        ) : null}

        {mode === "confirm" ? (
          <form onSubmit={handleConfirm} className="login-form">
            <label htmlFor="confirm-email">Email</label>
            <input
              id="confirm-email"
              type="email"
              autoComplete="username"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              required
            />
            <label htmlFor="confirmation-code">Verification code</label>
            <input
              id="confirmation-code"
              type="text"
              inputMode="numeric"
              autoComplete="one-time-code"
              value={confirmationCode}
              onChange={(event) => setConfirmationCode(event.target.value)}
              required
            />
            {error ? <div className="login-error">{error}</div> : null}
            <button type="submit" className="login-submit-btn" disabled={isSubmitting}>
              {isSubmitting ? "Verifying..." : "Verify email"}
            </button>
            <button
              type="button"
              className="login-link-btn"
              onClick={() => void handleResendCode()}
              disabled={isSubmitting || !email.trim()}
            >
              Resend code
            </button>
          </form>
        ) : null}

        <div className="login-switch">
          {mode === "signin" ? (
            <button type="button" className="login-link-btn" onClick={() => switchMode("signup")}>
              Need an account? Sign up
            </button>
          ) : null}
          {mode === "signup" ? (
            <button type="button" className="login-link-btn" onClick={() => switchMode("signin")}>
              Already have an account? Sign in
            </button>
          ) : null}
          {mode === "confirm" ? (
            <button type="button" className="login-link-btn" onClick={() => switchMode("signin")}>
              Back to sign in
            </button>
          ) : null}
        </div>
      </div>
    </div>
  );
}
