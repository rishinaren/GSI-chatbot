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

export function getAccessToken() {
  return localStorage.getItem(TOKEN_KEY);
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
