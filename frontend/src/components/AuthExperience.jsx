import { useEffect, useState } from "react";
import {
  getMembershipConfig,
  memberLogin,
  subscribe as apiSubscribe,
  subscriberConfirm,
  subscriberLogin,
  subscriberResendCode,
  subscriberResendSignupCode,
  subscriberVerify,
} from "../api";
import { storeMembershipSession } from "../auth";

// Fallback so the UI renders even if the config request fails.
const FALLBACK_PLAN = {
  name: "GSI Chatbot — Annual",
  amount: 1200,
  currency: "USD",
  interval: "year",
  trial_days: 30,
  monthly_equivalent: 100,
  features: [
    "Unlimited standards Q&A grounded in ASTM, ISO & GRI",
    "Page-bound citations with source links",
    "Video walkthroughs surfaced inline",
    "Saved chats and projects",
  ],
};

function formatMoney(amount, currency) {
  try {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: currency || "USD",
      maximumFractionDigits: 0,
    }).format(amount);
  } catch {
    return `$${amount}`;
  }
}

function errorText(err) {
  return err instanceof Error && err.message ? err.message : "Something went wrong. Please try again.";
}

function BackButton({ onClick, label = "Back" }) {
  return (
    <button type="button" className="auth-back-btn" onClick={onClick}>
      ← {label}
    </button>
  );
}

function Field({ id, label, type = "text", value, onChange, autoComplete, placeholder, inputMode, required = true }) {
  return (
    <div className="auth-field">
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        type={type}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        autoComplete={autoComplete}
        placeholder={placeholder}
        inputMode={inputMode}
        required={required}
      />
    </div>
  );
}

export default function AuthExperience({ onSignedIn, connectionError = "" }) {
  const [step, setStep] = useState("choose");
  const [plan, setPlan] = useState(FALLBACK_PLAN);

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [card, setCard] = useState({ number: "", expiry: "", cvc: "", zip: "" });
  const [code, setCode] = useState("");

  const [challenge, setChallenge] = useState(null); // login code: { email, destination, demo_code }
  const [signupPending, setSignupPending] = useState(null); // signup confirm: { email, destination }
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    let active = true;
    getMembershipConfig()
      .then((config) => {
        if (active && config?.plan) setPlan(config.plan);
      })
      .catch(() => {});
    return () => {
      active = false;
    };
  }, []);

  function goto(nextStep) {
    setError("");
    setNotice("");
    setStep(nextStep);
  }

  function finishSignIn(result) {
    storeMembershipSession(result);
    onSignedIn();
  }

  async function handleMemberLogin(event) {
    event.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      const result = await memberLogin(email.trim(), password);
      finishSignIn(result);
    } catch (err) {
      setError(errorText(err));
    } finally {
      setSubmitting(false);
    }
  }

  async function handleSubscriberLogin(event) {
    event.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      const result = await subscriberLogin(email.trim(), password);
      setChallenge(result);
      setCode("");
      goto("code");
    } catch (err) {
      setError(errorText(err));
    } finally {
      setSubmitting(false);
    }
  }

  async function handleSubscribe(event) {
    event.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      const result = await apiSubscribe({
        email: email.trim(),
        password,
        name: name.trim() || null,
        payment: { ...card },
      });
      if (result.access_token) {
        finishSignIn(result); // account already confirmed
        return;
      }
      setSignupPending(result.signup || { email: email.trim() });
      setCode("");
      goto("signup-confirm");
    } catch (err) {
      setError(errorText(err));
    } finally {
      setSubmitting(false);
    }
  }

  // Subscriber login: verify the per-login email code.
  async function handleVerify(event) {
    event.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      const result = await subscriberVerify(challenge?.email || email.trim(), code.trim());
      finishSignIn(result);
    } catch (err) {
      setError(errorText(err));
    } finally {
      setSubmitting(false);
    }
  }

  async function handleResend() {
    setError("");
    setNotice("");
    setSubmitting(true);
    try {
      const result = await subscriberResendCode(challenge?.email || email.trim());
      setChallenge((current) => ({ ...current, ...result }));
      setNotice("A new code is on its way.");
    } catch (err) {
      setError(errorText(err));
    } finally {
      setSubmitting(false);
    }
  }

  // New subscriber: confirm the Cognito signup email code.
  async function handleConfirmSignup(event) {
    event.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      const result = await subscriberConfirm(signupPending?.email || email.trim(), code.trim());
      finishSignIn(result);
    } catch (err) {
      setError(errorText(err));
    } finally {
      setSubmitting(false);
    }
  }

  async function handleResendSignup() {
    setError("");
    setNotice("");
    setSubmitting(true);
    try {
      await subscriberResendSignupCode(signupPending?.email || email.trim());
      setNotice("A new verification code is on its way.");
    } catch (err) {
      setError(errorText(err));
    } finally {
      setSubmitting(false);
    }
  }

  const monthly = plan.monthly_equivalent ? formatMoney(plan.monthly_equivalent, plan.currency) : null;
  const priceLine = `${formatMoney(plan.amount, plan.currency)} / ${plan.interval}`;

  return (
    <div className="auth-experience" role="dialog" aria-modal="true" aria-label="Sign in">
      <div className={`auth-shell ${step === "subscribe" ? "wide" : ""}`}>
        <div className="auth-brand">
          <span className="auth-brand-dot" />
          GSI Chatbot
        </div>

        {connectionError ? <div className="auth-error">{connectionError}</div> : null}

        {step === "choose" ? (
          <div className="auth-choose">
            <h1>Welcome to the GSI Chatbot</h1>
            <p className="auth-subtitle">
              The standards research assistant for geosynthetics — grounded in ASTM, ISO &amp; GRI.
            </p>
            <div className="auth-options">
              <button type="button" className="auth-option" onClick={() => goto("member")}>
                <span className="auth-option-eyebrow">Already a GSI member</span>
                <span className="auth-option-title">Sign in with GSI credentials</span>
                <span className="auth-option-desc">
                  Your membership already includes access to the chatbot!
                </span>
                <span className="auth-option-cta">Continue →</span>
              </button>

              <button type="button" className="auth-option" onClick={() => goto("subscriber")}>
                <span className="auth-option-eyebrow">Have a subscription</span>
                <span className="auth-option-title">Sign in to your account</span>
                <span className="auth-option-desc">
                  We&apos;ll email you a one-time code to finish signing in.
                </span>
                <span className="auth-option-cta">Continue →</span>
              </button>

              <button type="button" className="auth-option featured" onClick={() => goto("subscribe")}>
                <span className="auth-option-eyebrow">New here</span>
                <span className="auth-option-title">Start your free trial</span>
                <span className="auth-option-desc">
                  {priceLine} · 1-month free trial · cancel anytime.
                </span>
                <span className="auth-option-cta">Subscribe →</span>
              </button>
            </div>
          </div>
        ) : null}

        {step === "member" ? (
          <div className="auth-form-wrap">
            <BackButton onClick={() => goto("choose")} />
            <h1>Sign in as a GSI member</h1>
            <p className="auth-subtitle">
              Use your Geosynthetic Institute credentials. Members are already verified, so there&apos;s
              no email code.
            </p>
            <form className="auth-form" onSubmit={handleMemberLogin}>
              <Field
                id="member-email"
                label="GSI email"
                type="email"
                value={email}
                onChange={setEmail}
                autoComplete="username"
              />
              <Field
                id="member-password"
                label="Password"
                type="password"
                value={password}
                onChange={setPassword}
                autoComplete="current-password"
              />
              {error ? <div className="auth-error">{error}</div> : null}
              <button type="submit" className="auth-submit" disabled={submitting}>
                {submitting ? "Signing in…" : "Sign in"}
              </button>
            </form>
            <p className="auth-fineprint">Verified against the GSI member directory.</p>
          </div>
        ) : null}

        {step === "subscriber" ? (
          <div className="auth-form-wrap">
            <BackButton onClick={() => goto("choose")} />
            <h1>Sign in to your subscription</h1>
            <p className="auth-subtitle">Enter your email and password. We&apos;ll email you a login code.</p>
            <form className="auth-form" onSubmit={handleSubscriberLogin}>
              <Field
                id="sub-email"
                label="Email"
                type="email"
                value={email}
                onChange={setEmail}
                autoComplete="username"
              />
              <Field
                id="sub-password"
                label="Password"
                type="password"
                value={password}
                onChange={setPassword}
                autoComplete="current-password"
              />
              {error ? <div className="auth-error">{error}</div> : null}
              <button type="submit" className="auth-submit" disabled={submitting}>
                {submitting ? "Sending code…" : "Continue"}
              </button>
            </form>
            <p className="auth-fineprint">
              New here?{" "}
              <button type="button" className="auth-link" onClick={() => goto("subscribe")}>
                Start a free trial
              </button>
            </p>
          </div>
        ) : null}

        {step === "code" ? (
          <div className="auth-form-wrap">
            <BackButton onClick={() => goto("subscriber")} />
            <h1>Enter your login code</h1>
            <p className="auth-subtitle">
              We sent a 6-digit code to <strong>{challenge?.destination || "your email"}</strong>.
            </p>
            {challenge?.demo_code ? (
              <div className="auth-demo-hint">
                Demo mode — your code is <strong>{challenge.demo_code}</strong>
              </div>
            ) : null}
            {notice ? <div className="auth-notice">{notice}</div> : null}
            <form className="auth-form" onSubmit={handleVerify}>
              <Field
                id="verify-code"
                label="Verification code"
                value={code}
                onChange={setCode}
                autoComplete="one-time-code"
                inputMode="numeric"
                placeholder="123456"
              />
              {error ? <div className="auth-error">{error}</div> : null}
              <button type="submit" className="auth-submit" disabled={submitting}>
                {submitting ? "Verifying…" : "Verify & sign in"}
              </button>
              <button type="button" className="auth-link centered" onClick={handleResend} disabled={submitting}>
                Resend code
              </button>
            </form>
          </div>
        ) : null}

        {step === "signup-confirm" ? (
          <div className="auth-form-wrap">
            <BackButton onClick={() => goto("subscribe")} />
            <h1>Verify your email</h1>
            <p className="auth-subtitle">
              Your trial is set up. Enter the verification code we emailed to{" "}
              <strong>{signupPending?.destination || "your email"}</strong> to finish.
            </p>
            {notice ? <div className="auth-notice">{notice}</div> : null}
            <form className="auth-form" onSubmit={handleConfirmSignup}>
              <Field
                id="signup-code"
                label="Verification code"
                value={code}
                onChange={setCode}
                autoComplete="one-time-code"
                inputMode="numeric"
                placeholder="123456"
              />
              {error ? <div className="auth-error">{error}</div> : null}
              <button type="submit" className="auth-submit" disabled={submitting}>
                {submitting ? "Verifying…" : "Verify & continue"}
              </button>
              <button
                type="button"
                className="auth-link centered"
                onClick={handleResendSignup}
                disabled={submitting}
              >
                Resend code
              </button>
            </form>
          </div>
        ) : null}

        {step === "subscribe" ? (
          <div className="auth-subscribe">
            <BackButton onClick={() => goto("choose")} />
            <div className="auth-subscribe-grid">
              <aside className="auth-plan">
                <div className="auth-plan-name">{plan.name}</div>
                <div className="auth-plan-price">
                  <span className="auth-plan-amount">{formatMoney(plan.amount, plan.currency)}</span>
                  <span className="auth-plan-interval">/ {plan.interval}</span>
                </div>
                {monthly ? (
                  <div className="auth-plan-note">≈ {monthly}/mo, billed annually</div>
                ) : null}
                <div className="auth-plan-badge">{plan.trial_days ? `${plan.trial_days}-day free trial` : "Free trial"} · cancel anytime</div>
                <ul className="auth-plan-features">
                  {(plan.features || []).map((feature) => (
                    <li key={feature}>{feature}</li>
                  ))}
                </ul>
              </aside>

              <div className="auth-checkout">
                <h1>Start your free trial</h1>
                <p className="auth-subtitle">
                  No charge today — your {plan.trial_days ? `${plan.trial_days}-day ` : ""}free trial starts
                  now. Cancel anytime before it ends.
                </p>
                <form className="auth-form" onSubmit={handleSubscribe}>
                  <Field id="checkout-name" label="Name" value={name} onChange={setName} autoComplete="name" required={false} />
                  <Field
                    id="checkout-email"
                    label="Email"
                    type="email"
                    value={email}
                    onChange={setEmail}
                    autoComplete="email"
                  />
                  <Field
                    id="checkout-password"
                    label="Create a password"
                    type="password"
                    value={password}
                    onChange={setPassword}
                    autoComplete="new-password"
                    placeholder="8+ chars: upper & lower case, number, symbol"
                  />
                  <Field
                    id="checkout-card"
                    label="Card number"
                    value={card.number}
                    onChange={(value) => setCard((c) => ({ ...c, number: value }))}
                    autoComplete="cc-number"
                    inputMode="numeric"
                    placeholder="4242 4242 4242 4242"
                    required={false}
                  />
                  <div className="auth-field-row">
                    <Field
                      id="checkout-exp"
                      label="Expiry"
                      value={card.expiry}
                      onChange={(value) => setCard((c) => ({ ...c, expiry: value }))}
                      autoComplete="cc-exp"
                      placeholder="MM / YY"
                      required={false}
                    />
                    <Field
                      id="checkout-cvc"
                      label="CVC"
                      value={card.cvc}
                      onChange={(value) => setCard((c) => ({ ...c, cvc: value }))}
                      autoComplete="cc-csc"
                      inputMode="numeric"
                      placeholder="123"
                      required={false}
                    />
                    <Field
                      id="checkout-zip"
                      label="ZIP"
                      value={card.zip}
                      onChange={(value) => setCard((c) => ({ ...c, zip: value }))}
                      autoComplete="postal-code"
                      inputMode="numeric"
                      placeholder="00000"
                      required={false}
                    />
                  </div>
                  {error ? <div className="auth-error">{error}</div> : null}
                  <button type="submit" className="auth-submit" disabled={submitting}>
                    {submitting ? "Starting trial…" : "Start free trial"}
                  </button>
                </form>
                <p className="auth-fineprint">
                  Demo checkout — no card is charged and card details are not stored.
                </p>
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
