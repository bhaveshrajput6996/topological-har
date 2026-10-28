import { useCallback, useEffect, useRef, useState } from "react";
import { api, type ApiError } from "./client";

export interface ApiState<T> {
  data: T | null;
  loading: boolean;
  error: ApiError | null;
  reload: () => void;
  loadedAt: number | null;
}

/**
 * Fetch-on-mount hook with reload. Keeps previous data while refetching
 * (no blank screens), exposes loading/error for state rendering.
 */
export function useApi<T>(
  fetcher: () => Promise<T>,
  deps: unknown[] = [],
  opts: { enabled?: boolean } = {},
): ApiState<T> {
  const enabled = opts.enabled ?? true;
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(enabled);
  const [error, setError] = useState<ApiError | null>(null);
  const [loadedAt, setLoadedAt] = useState<number | null>(null);
  const [tick, setTick] = useState(0);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetcherRef
      .current()
      .then((d) => {
        if (cancelled) return;
        setData(d);
        setLoadedAt(Date.now());
      })
      .catch((e: ApiError) => {
        if (cancelled) return;
        setError(e);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick, enabled]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { data, loading, error, reload, loadedAt };
}

/** setInterval that clears itself; pass null delay to pause. */
export function useInterval(fn: () => void, delayMs: number | null): void {
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => {
    if (delayMs === null) return;
    const id = setInterval(() => ref.current(), delayMs);
    return () => clearInterval(id);
  }, [delayMs]);
}
