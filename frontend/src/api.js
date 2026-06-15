const API_BASE = import.meta.env.VITE_API_BASE_URL || "";
const TOKEN_KEY = "gsi_access_token";
const ID_TOKEN_KEY = "gsi_id_token";
const PUBLIC_PATHS = new Set([
  "/auth/config",
  "/auth/login",
  "/auth/signup",
  "/auth/confirm",
  "/auth/resend-confirmation",
  "/health",
]);

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function clearStoredSession() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(ID_TOKEN_KEY);
}

function authHeaders(path) {
  if (PUBLIC_PATHS.has(path)) {
    return {};
  }
  const token = localStorage.getItem(TOKEN_KEY);
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function parseApiError(detail) {
  if (!detail) {
    return "";
  }
  try {
    const parsed = JSON.parse(detail);
    if (typeof parsed.detail === "string") {
      return parsed.detail;
    }
  } catch {
    // Keep raw text when the API does not return JSON.
  }
  return detail;
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(path),
      ...(options.headers || {}),
    },
  });
  if (!response.ok) {
    const detail = await response.text();
    if (response.status === 401) {
      clearStoredSession();
    }
    throw new ApiError(
      parseApiError(detail) || `Request failed with status ${response.status}`,
      response.status,
    );
  }
  if (response.status === 204) {
    return null;
  }
  return response.json();
}

export function getHealth() {
  return request("/health");
}

export function getAuthConfig() {
  return request("/auth/config");
}

export function login(email, password) {
  return request("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export function signUp(email, password) {
  return request("/auth/signup", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export function confirmSignUp(email, confirmationCode) {
  return request("/auth/confirm", {
    method: "POST",
    body: JSON.stringify({ email, confirmation_code: confirmationCode }),
  });
}

export function resendConfirmationCode(email) {
  return request("/auth/resend-confirmation", {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

export function listConversations() {
  return request("/conversations");
}

export function createConversation(payload = {}) {
  return request("/conversations", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getConversation(conversationId) {
  return request(`/conversations/${conversationId}`);
}

export function deleteConversation(conversationId) {
  return request(`/conversations/${conversationId}`, { method: "DELETE" });
}

export function pinConversation(conversationId, pinned) {
  return request(`/conversations/${conversationId}`, {
    method: "PATCH",
    body: JSON.stringify({ pinned }),
  });
}

export function searchVideos(query, topK = 3) {
  return request("/videos/search", {
    method: "POST",
    body: JSON.stringify({ query, top_k: topK }),
  });
}

export function sendChat(payload) {
  return request("/chat", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function withApiBase(path) {
  if (!path) return path;
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path}`;
}

// PDFs open via a top-level browser navigation (new tab), which cannot send the
// Authorization header. Append the access token as a query param so the API can
// authenticate the request.
export function withAuthedFileUrl(path) {
  const base = withApiBase(path);
  if (!base) return base;
  const token = localStorage.getItem(TOKEN_KEY);
  if (!token) return base;
  const separator = base.includes("?") ? "&" : "?";
  return `${base}${separator}token=${encodeURIComponent(token)}`;
}
