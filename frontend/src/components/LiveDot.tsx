type LiveStatus = "live" | "stale" | "dead";

interface LiveDotProps {
  status?: LiveStatus;
  size?: number;
  className?: string;
  title?: string;
}

const statusColor: Record<LiveStatus, string> = {
  live: "var(--color-profit)",
  stale: "var(--color-warn)",
  dead: "var(--color-loss)",
};

export function LiveDot({
  status = "live",
  size = 6,
  className = "",
  title,
}: LiveDotProps) {
  return (
    <span
      role="status"
      aria-label={title ?? status}
      title={title ?? status}
      className={`inline-block rounded-full ${status === "live" ? "live-pulse" : ""} ${className}`}
      style={{
        width: size,
        height: size,
        backgroundColor: statusColor[status],
        boxShadow: status === "live" ? `0 0 8px ${statusColor[status]}` : undefined,
      }}
    />
  );
}
