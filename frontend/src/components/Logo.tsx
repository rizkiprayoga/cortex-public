import { useId } from "react";

export type LogoVariant = "mark" | "full";

export interface LogoProps {
  size?: number;
  variant?: LogoVariant;
  title?: string;
  className?: string;
  /** When rendering the full lockup, override the wordmark (default: "Cortex"). */
  wordmark?: string;
}

/**
 * Synapse · neural-network nodes around a central hub.
 * Gradient is hard-coded to the brand palette (cyan → indigo → violet)
 * so the mark reads consistently across every theme.
 */
export function Logo({
  size = 22,
  variant = "mark",
  title = "Cortex",
  className,
  wordmark = "Cortex",
}: LogoProps) {
  const gradientId = useId();

  const mark = (
    <svg
      width={size}
      height={size}
      viewBox="0 0 40 40"
      fill="none"
      role="img"
      aria-label={title}
      className={variant === "mark" ? className : undefined}
    >
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="40" y2="40" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#06b6d4" />
          <stop offset="1" stopColor="#8b5cf6" />
        </linearGradient>
      </defs>
      <line x1="10" y1="12" x2="22" y2="20" stroke={`url(#${gradientId})`} strokeWidth="1.2" opacity="0.7" />
      <line x1="22" y1="20" x2="32" y2="10" stroke={`url(#${gradientId})`} strokeWidth="1.2" opacity="0.7" />
      <line x1="22" y1="20" x2="30" y2="30" stroke={`url(#${gradientId})`} strokeWidth="1.2" opacity="0.7" />
      <line x1="22" y1="20" x2="12" y2="28" stroke={`url(#${gradientId})`} strokeWidth="1.2" opacity="0.7" />
      <circle cx="10" cy="12" r="2.5" fill="#06b6d4" />
      <circle cx="32" cy="10" r="2.5" fill="#6366f1" />
      <circle cx="30" cy="30" r="2.5" fill="#8b5cf6" />
      <circle cx="12" cy="28" r="2.5" fill="#22d3ee" />
      <circle cx="22" cy="20" r="4" fill={`url(#${gradientId})`} />
    </svg>
  );

  if (variant === "mark") return mark;

  return (
    <span className={["inline-flex items-center gap-2", className].filter(Boolean).join(" ")}>
      {mark}
      <span className="text-brand-gradient font-semibold tracking-tight" style={{ fontSize: size * 0.82 }}>
        {wordmark}
      </span>
    </span>
  );
}

export default Logo;
