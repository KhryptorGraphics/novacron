// API Client for NovaCron Enhanced API
export class ApiClient {
  private baseURL: string;
  private wsURL: string;
  private token: string | null = null;

  constructor() {
    this.baseURL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8090';
    this.wsURL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8091';

    // Try to get token from localStorage if available
    if (typeof window !== 'undefined') {
      this.token = localStorage.getItem('novacron_token') || null;
    }
  }

  setToken(token: string | null) {
    this.token = token;
    if (typeof window !== 'undefined' && token) {
      localStorage.setItem('novacron_token', token);
    } else if (typeof window !== 'undefined' && !token) {
      localStorage.removeItem('novacron_token');
    }
  }

  clearToken() {
    this.token = null;
    if (typeof window !== 'undefined') {
      localStorage.removeItem('novacron_token');
    }
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(options.headers as Record<string, string> || {}),
    };

    if (this.token) {
      headers.Authorization = `Bearer ${this.token}`;
    }

    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`API Error: ${response.status} - ${error}`);
    }

    if (response.status === 204) {
      return {} as T; // No content
    }

    return response.json();
  }

  async get<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'GET' });
  }

  async post<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async put<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async delete<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'DELETE' });
  }

  // WebSocket connection
  connectWebSocket(path: string): WebSocket | null {
    try {
      const url = `${this.wsURL}${path}`;
      const ws = new WebSocket(url);

      // Add authentication if token is available
      if (this.token) {
        ws.addEventListener('open', () => {
          ws.send(JSON.stringify({
            type: 'auth',
            token: this.token
          }));
        });
      }

      return ws;
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      return null;
    }
  }
}

export const apiClient = new ApiClient();

// ----- Core-mode typed API helpers -----
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8090/api/v1";

export type ApiError = { code: string; message: string };
export type Pagination = { page: number; pageSize: number; total: number; totalPages: number; sortBy?: "name"|"createdAt"|"state"; sortDir?: "asc"|"desc" };
export type ApiEnvelope<T> = { data: T | null; error: ApiError | null; pagination?: Pagination };

function withParams(path: string, params?: Record<string, string | number | undefined>): string {
  const url = new URL(path, API_BASE);
  const sp = new URLSearchParams();
  if (params) {
    for (const [k,v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") sp.set(k, String(v));
  }
  url.search = sp.toString();
  return url.toString();
}

function parsePaginationHeader(res: Response): Pagination | undefined {
  const raw = res.headers.get("X-Pagination");
  if (!raw) return undefined;
  try { return JSON.parse(raw) as Pagination } catch { return undefined }
}

export class ApiHttpError extends Error {
  status: number; code: string; url: string;
  constructor(status: number, code: string, message: string, url: string) { super(message); this.status=status; this.code=code; this.url=url; }
}

/**
 * GET helper for core mode.
 * @param path API path starting with "/" relative to API_BASE
 * @param params Optional query string parameters
 * @param opts Optional options { role } where role defaults to "viewer"
 * @returns ApiEnvelope<T> with pagination populated from X-Pagination header if present
 */
export async function apiGet<T>(path: string, params?: Record<string, string | number | undefined>, opts?: { role?: "viewer" | "operator" }): Promise<ApiEnvelope<T>> {
  try {
    const url = withParams(path, params);
    const role = opts?.role ?? "viewer";
    const res = await fetch(url, { method: "GET", headers: { Accept: "application/json", "X-Role": role }, credentials: "include" });
    
    if (!res.ok) {
      return { data: null, error: { code: `HTTP_${res.status}`, message: res.statusText } };
    }
    
    const env = await res.json() as ApiEnvelope<T>;
    const pg = parsePaginationHeader(res); if (pg) env.pagination = pg;
    if (env.error) throw new ApiHttpError(res.status, env.error.code, env.error.message, url);
    return env;
  } catch (error) {
    return { data: null, error: { code: "FETCH_ERROR", message: error instanceof Error ? error.message : "Unknown error" } };
  }
}

/**
 * POST helper for core mode.
 * @param path API path starting with "/" relative to API_BASE
 * @param body Optional JSON body (will be JSON.stringified if provided)
 * @param opts Optional options { role } where role defaults to "viewer"
 * @returns ApiEnvelope<T>
 */
export async function apiPost<T>(path: string, body?: unknown, opts?: { role?: "viewer" | "operator" }): Promise<ApiEnvelope<T>> {
  const url = new URL(path, API_BASE).toString();
  const role = opts?.role ?? "viewer";
  const res = await fetch(url, { method: "POST", headers: { Accept: "application/json", "Content-Type": "application/json", "X-Role": role }, body: body!==undefined?JSON.stringify(body):undefined, credentials: "include" });
  const env = await res.json() as ApiEnvelope<T>;
  if (env.error) throw new ApiHttpError(res.status, env.error.code, env.error.message, url);
  return env;
}
