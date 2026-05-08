const BASE = "";

function getToken(): string | null {
  return localStorage.getItem("jwt_token");
}

function authHeaders(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request<T>(
  path: string,
  opts: RequestInit = {},
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...opts,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
      ...opts.headers,
    },
  });

  if (res.status === 401) {
    localStorage.removeItem("jwt_token");
    if (!path.includes("/auth/")) {
      window.location.href = "/ui/login";
    }
    throw new Error("Unauthorized");
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }

  return res.json() as Promise<T>;
}

export const api = {
  get: <T>(path: string) => request<T>(path),

  post: <T>(path: string, body?: unknown) =>
    request<T>(path, {
      method: "POST",
      body: body ? JSON.stringify(body) : undefined,
    }),

  setToken: (token: string) => localStorage.setItem("jwt_token", token),

  clearToken: () => localStorage.removeItem("jwt_token"),

  hasToken: () => !!getToken(),
};
