import {
  clearStoredSession,
  confirmSignUp as apiConfirmSignUp,
  getAuthConfig,
  login as apiLogin,
  resendConfirmationCode as apiResendConfirmationCode,
  signUp as apiSignUp,
} from "./api";

const TOKEN_KEY = "gsi_access_token";
const ID_TOKEN_KEY = "gsi_id_token";
const EMAIL_KEY = "gsi_user_email";
const ACCOUNT_TYPE_KEY = "gsi_account_type";

export function getAccessToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function isAuthenticated() {
  return Boolean(getAccessToken());
}

export function getAccountType() {
  return localStorage.getItem(ACCOUNT_TYPE_KEY) || "";
}

export function getUserEmail() {
  // Membership sessions store the email directly; Cognito sessions carry it in
  // the id token.
  const stored = localStorage.getItem(EMAIL_KEY);
  if (stored) return stored;
  const idToken = localStorage.getItem(ID_TOKEN_KEY);
  if (!idToken) return "";
  try {
    const payload = idToken.split(".")[1];
    const json = JSON.parse(atob(payload.replace(/-/g, "+").replace(/_/g, "/")));
    return json.email || "";
  } catch {
    return "";
  }
}

// Persist a membership (GSI-member / subscriber) session issued by the API.
export function storeMembershipSession({ access_token, email, account_type }) {
  localStorage.setItem(TOKEN_KEY, access_token);
  if (email) localStorage.setItem(EMAIL_KEY, email);
  if (account_type) localStorage.setItem(ACCOUNT_TYPE_KEY, account_type);
  localStorage.removeItem(ID_TOKEN_KEY);
}

export function clearSession() {
  clearStoredSession();
}

export async function loadAuthState() {
  const config = await getAuthConfig();
  return {
    authRequired: Boolean(config.auth_required),
    configured: Boolean(config.cognito_user_pool_id && config.cognito_app_client_id),
    config,
    isLoggedIn: Boolean(getAccessToken()) || !config.auth_required,
  };
}

export async function signIn(email, password) {
  const tokens = await apiLogin(email, password);
  localStorage.setItem(TOKEN_KEY, tokens.access_token);
  localStorage.setItem(ID_TOKEN_KEY, tokens.id_token);
  return tokens;
}

export async function signUp(email, password) {
  return apiSignUp(email, password);
}

export async function confirmSignUp(email, confirmationCode) {
  return apiConfirmSignUp(email, confirmationCode);
}

export async function resendConfirmationCode(email) {
  return apiResendConfirmationCode(email);
}

export function signOut() {
  clearSession();
}
