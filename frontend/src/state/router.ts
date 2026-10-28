import { useEffect, useState, useCallback } from "react";

export type RouteInfo = { page: string; params: URLSearchParams };

function parse(): RouteInfo {
  const h = window.location.hash.replace(/^#\/?/, "");
  const [page, query] = h.split("?");
  return { page: page || "overview", params: new URLSearchParams(query ?? "") };
}

export function useRoute(): RouteInfo {
  const [route, setRoute] = useState<RouteInfo>(parse);
  useEffect(() => {
    const on = () => setRoute(parse());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

export function navigate(page: string, params?: Record<string, string>): void {
  const q = params && Object.keys(params).length ? `?${new URLSearchParams(params).toString()}` : "";
  window.location.hash = `/${page}${q}`;
}

export function useNavigate(): (page: string, params?: Record<string, string>) => void {
  return useCallback(navigate, []);
}
