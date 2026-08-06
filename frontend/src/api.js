const API_BASE = import.meta.env.VITE_API_BASE_URL || "";
const TOKEN_KEY = "gsi_access_token";
const ID_TOKEN_KEY = "gsi_id_token";
const PUBLIC_PATHS = new Set([
  "/auth/config",
  "/auth/login",
  "/auth/signup",
  "/auth/confirm",
  "/auth/resend-confirmation",
  "/auth/membership/config",
  "/auth/member/login",
  "/auth/subscriber/login",
  "/auth/subscriber/verify",
  "/auth/subscriber/resend-code",
  "/auth/subscriber/confirm",
  "/auth/subscriber/resend-signup-code",
  "/billing/subscribe",
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
  localStorage.removeItem("gsi_user_email");
  localStorage.removeItem("gsi_account_type");
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

// File uploads must not carry a JSON Content-Type: the browser sets
// multipart/form-data itself, including the boundary.
async function upload(path, formData) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: authHeaders(path),
      body: formData,
    });
  } catch {
    // fetch only rejects on a network-level failure, which reaches the user as
    // an unhelpful "Failed to fetch" unless we say something they can act on.
    throw new ApiError(
      "We lost the connection while sending that file. Check your internet and try again.",
      0,
    );
  }
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
  return response.json();
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

// ---- Membership + subscription auth ----

export function getMembershipConfig() {
  return request("/auth/membership/config");
}

export function memberLogin(email, password) {
  return request("/auth/member/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export function subscriberLogin(email, password) {
  return request("/auth/subscriber/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export function subscriberVerify(email, code) {
  return request("/auth/subscriber/verify", {
    method: "POST",
    body: JSON.stringify({ email, code }),
  });
}

export function subscriberResendCode(email) {
  return request("/auth/subscriber/resend-code", {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

export function subscriberConfirm(email, code) {
  return request("/auth/subscriber/confirm", {
    method: "POST",
    body: JSON.stringify({ email, code }),
  });
}

export function subscriberResendSignupCode(email) {
  return request("/auth/subscriber/resend-signup-code", {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

export function subscribe({ email, password, name, payment }) {
  return request("/billing/subscribe", {
    method: "POST",
    body: JSON.stringify({ email, password, name, payment }),
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

export function assignConversationToProject(conversationId, projectId) {
  return request(`/conversations/${conversationId}`, {
    method: "PATCH",
    body: JSON.stringify({ project_id: projectId }),
  });
}

export function listProjects() {
  return request("/projects");
}

export function createProject(name) {
  return request("/projects", {
    method: "POST",
    body: JSON.stringify({ name }),
  });
}

export function renameProject(projectId, name) {
  return request(`/projects/${projectId}`, {
    method: "PATCH",
    body: JSON.stringify({ name }),
  });
}

export function deleteProject(projectId) {
  return request(`/projects/${projectId}`, { method: "DELETE" });
}

// ---- Admin document library ----

export function getAdminConfig() {
  return request("/admin/config");
}

export function listLibraryDocuments() {
  return request("/admin/documents");
}

export function analyzeLibraryDocument(file) {
  const form = new FormData();
  form.append("file", file);
  return upload("/admin/documents/analyze", form);
}

export function addLibraryDocument({ file, standardId, title, issuingBody, year }) {
  const form = new FormData();
  form.append("file", file);
  form.append("standard_id", standardId ?? "");
  form.append("title", title ?? "");
  form.append("issuing_body", issuingBody ?? "");
  form.append("year", year ?? "");
  return upload("/admin/documents", form);
}

export function listLibraryPeople() {
  return request("/admin/people");
}

export function grantLibraryAccess(email) {
  return request("/admin/people", {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

export function revokeLibraryAccess(email) {
  return request(`/admin/people/${encodeURIComponent(email)}`, { method: "DELETE" });
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
