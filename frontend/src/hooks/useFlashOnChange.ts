import { useEffect, useRef, useState } from "react";

/**
 * Return "flash-profit" / "flash-loss" / "" for ~0.9s after `value` changes,
 * based on whether the new value is greater or less than the previous.
 * First mount is silent (no flash on initial render). NaN/undefined are
 * treated as "no change" so missing data doesn't falsely trigger.
 *
 * CSS classes are defined in index.css (keyframes flash-profit-kf /
 * flash-loss-kf, 0.9s ease-out).
 */
export function useFlashOnChange(value: number | null | undefined): string {
  const [flash, setFlash] = useState<"" | "flash-profit" | "flash-loss">("");
  const prevRef = useRef<number | null>(null);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    if (value === null || value === undefined || Number.isNaN(value)) return;
    const prev = prevRef.current;
    prevRef.current = value;
    if (prev === null || prev === value) return;
    setFlash(value > prev ? "flash-profit" : "flash-loss");
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(() => setFlash(""), 900);
    return () => {
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    };
  }, [value]);

  return flash;
}
