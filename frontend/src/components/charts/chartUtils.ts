import { useEffect, useRef, useState } from "react";

/** Observe the rendered width/height of an element. */
export function useSize<T extends HTMLElement>(): [React.RefObject<T>, number, number] {
  const ref = useRef<T>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        const cr = e.contentRect;
        setSize({ width: Math.round(cr.width), height: Math.round(cr.height) });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, size.width, size.height];
}

export interface Scale {
  (v: number): number;
  domain: [number, number];
  range: [number, number];
  invert: (px: number) => number;
}

export function linearScale(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  const fn = ((v: number) => r0 + ((v - d0) / span) * (r1 - r0)) as Scale;
  fn.domain = domain;
  fn.range = range;
  fn.invert = (px: number) => d0 + ((px - r0) / (r1 - r0 || 1)) * span;
  return fn;
}

/** "Nice" ticks for a numeric axis. */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    if (min === max) return [min];
    return [];
  }
  const span = max - min;
  const step0 = span / Math.max(1, count - 1);
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const norm = step0 / mag;
  let step: number;
  if (norm <= 1) step = 1;
  else if (norm <= 2) step = 2;
  else if (norm <= 2.5) step = 2.5;
  else if (norm <= 5) step = 5;
  else step = 10;
  step *= mag;
  const start = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= max + step * 1e-6; v += step) ticks.push(Math.round(v / step) * step === 0 ? 0 : v);
  return ticks;
}

export function formatTick(v: number): string {
  if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(v % 1000 === 0 ? 0 : 1)}k`;
  if (Math.abs(v) >= 10) return v.toFixed(0);
  if (Math.abs(v) >= 1) return v.toFixed(v % 1 === 0 ? 0 : 1);
  return v.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}

export function downsampleLTTB<T>(data: T[], threshold: number, xOf: (d: T) => number, yOf: (d: T) => number): T[] {
  if (data.length <= threshold) return data;
  const sampled: T[] = [];
  const every = (data.length - 2) / (threshold - 2);
  let a = 0;
  sampled.push(data[0]);
  for (let i = 0; i < threshold - 2; i++) {
    const avgStart = Math.floor((i + 1) * every) + 1;
    const avgEnd = Math.min(Math.floor((i + 2) * every) + 1, data.length);
    let avgX = 0, avgY = 0;
    const rangeLen = Math.max(1, avgEnd - avgStart);
    for (let j = avgStart; j < avgEnd; j++) {
      avgX += xOf(data[j]);
      avgY += yOf(data[j]);
    }
    avgX /= rangeLen;
    avgY /= rangeLen;
    const rangeOffs = Math.floor(i * every) + 1;
    const rangeTo = Math.floor((i + 1) * every) + 1;
    const pointAX = xOf(data[a]);
    const pointAY = yOf(data[a]);
    let maxArea = -1;
    let next = rangeOffs;
    for (let j = rangeOffs; j < rangeTo; j++) {
      const area =
        Math.abs((pointAX - avgX) * (yOf(data[j]) - pointAY) - (pointAX - xOf(data[j])) * (avgY - pointAY)) * 0.5;
      if (area > maxArea) {
        maxArea = area;
        next = j;
      }
    }
    sampled.push(data[next]);
    a = next;
  }
  sampled.push(data[data.length - 1]);
  return sampled;
}
