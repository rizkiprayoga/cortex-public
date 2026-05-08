const STATUS_STYLES: Record<string, { dot: string; text: string; bg: string }> = {
  running: {
    dot: "bg-emerald-400",
    text: "text-emerald-400",
    bg: "bg-emerald-400/10",
  },
  paused: {
    dot: "bg-amber-400",
    text: "text-amber-400",
    bg: "bg-amber-400/10",
  },
  stopped: {
    dot: "bg-rose-400",
    text: "text-rose-400",
    bg: "bg-rose-400/10",
  },
};

interface StatusBadgeProps {
  status: string;
  size?: "sm" | "md" | "lg";
}

export function StatusBadge({ status, size = "sm" }: StatusBadgeProps) {
  const s = STATUS_STYLES[status] ?? STATUS_STYLES.stopped;
  const textSize = size === "lg" ? "text-base" : size === "md" ? "text-sm" : "text-xs";

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full ${s.bg} ${s.text} ${textSize} font-medium uppercase`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${s.dot} animate-pulse`} />
      {status}
    </span>
  );
}
