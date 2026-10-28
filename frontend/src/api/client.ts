/**
 * Centralized HTTP client for the TDL-HAR backend.
 * - base URL from env (VITE_API_URL) with localStorage override
 * - timeout, JSON parsing, normalized errors
 * - a tiny request log (ring buffer) used by the System/API status page
 * - measured latency per request
 */

const LS_KEY = "thar.apiBase";

export function envBase(): string {
  // injected by Vite at build/dev time
  return (import.meta as { env?: Record<string, string> }).env?.VITE_API_URL ?? "http://127.0.0.1:8000";
}

export function getApiBase(): string {
  try {
    const stored = localStorage.getItem(LS_KEY);
    if (stored) return stored.replace(/\/+$/, "");
  } catch {
    /* storage unavailable */
  }
  return envBase().replace(/\/+$/, "");
}

export function setApiBase(url: string): void {
  try {
    if (url.trim() === "" || url.replace(/\/+$/, "") === envBase().replace(/\/+$/, "")) {
      localStorage.removeItem(LS_KEY);
    } else {
      localStorage.setItem(LS_KEY, url.trim().replace(/\/+$/, ""));
    }
  } catch {
    /* ignore */
  }
}

export interface ApiError extends Error {
  status: number;
  detail?: unknown;
}

export function makeApiError(message: string, status = 0, detail?: unknown): ApiError {
  const err = new Error(message) as ApiError;
  err.status = status;
  err.detail = detail;
  return err;
}

export interface RequestLogEntry {
  ts: number;
  method: string;
  endpoint: string;
  status: number;
  ms: number;
  ok: boolean;
}

const requestLog: RequestLogEntry[] = [];
const MAX_LOG = 80;
let lastSuccessTs = 0;

export function getRequestLog(): readonly RequestLogEntry[] {
  return requestLog;
}

export function getLastSuccessTs(): number {
  return lastSuccessTs;
}

function logRequest(entry: RequestLogEntry): void {
  requestLog.unshift(entry);
  if (requestLog.length > MAX_LOG) requestLog.pop();
  if (entry.ok) lastSuccessTs = entry.ts;
}

export interface RequestOptions {
  timeoutMs?: number;
  parse?: "json" | "void";
}

export async function apiFetch<T>(
  method: "GET" | "POST",
  endpoint: string,
  body?: unknown,
  opts: RequestOptions = {},
): Promise<T> {
  const base = getApiBase();
  const url = `${base}${endpoint}`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), opts.timeoutMs ?? 30000);
  const t0 = performance.now();
  try {
    const res = await fetch(url, {
      method,
      headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
    const ms = performance.now() - t0;
    logRequest({ ts: Date.now(), method, endpoint, status: res.status, ms: Math.round(ms), ok: res.ok });
    if (!res.ok) {
      let detail: unknown = null;
      let message = `HTTP ${res.status}`;
      try {
        const j = (await res.json()) as { detail?: unknown; error?: string };
        if (typeof j.detail === "string") message = j.detail;
        else if (j.detail) detail = j.detail;
        if (j.error) message = `${j.error}: ${message}`;
      } catch {
        /* non-json error body */
      }
      throw makeApiError(message, res.status, detail);
    }
    if (opts.parse === "void") return undefined as T;
    return (await res.json()) as T;
  } catch (e) {
    const ms = performance.now() - t0;
    if ((e as ApiError).status === undefined || (e as ApiError).status === 0) {
      // network / abort / CORS failure
      logRequest({ ts: Date.now(), method, endpoint, status: 0, ms: Math.round(ms), ok: false });
      if (e instanceof DOMException && e.name === "AbortError") {
        throw makeApiError(`Request timed out after ${opts.timeoutMs ?? 30000} ms`, 0);
      }
      throw makeApiError(
        `Cannot reach backend at ${base} — is the API server running?`,
        0,
      );
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

export const api = {
  get: <T>(endpoint: string, timeoutMs?: number) => apiFetch<T>("GET", endpoint, undefined, { timeoutMs }),
  post: <T>(endpoint: string, body?: unknown, timeoutMs?: number) =>
    apiFetch<T>("POST", endpoint, body, { timeoutMs }),
};
