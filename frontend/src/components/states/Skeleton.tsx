export function SkeletonLine({
  width = "60%",
  height = 10,
  className = "",
}: {
  width?: string | number;
  height?: number;
  className?: string;
}) {
  return (
    <div
      className={`skel ${className}`}
      style={{ width, height }}
      aria-hidden
    />
  );
}

export function SkeletonCard({
  title = "60%",
  value = "75%",
  sub = "40%",
  className = "",
}: {
  title?: string;
  value?: string;
  sub?: string;
  className?: string;
}) {
  return (
    <div
      className={`rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5 ${className}`}
    >
      <SkeletonLine width={title} height={10} />
      <SkeletonLine width={value} height={32} className="mt-4" />
      <SkeletonLine width={sub} height={10} className="mt-3" />
    </div>
  );
}

export function SkeletonChart({ height = 220 }: { height?: number }) {
  return (
    <div
      className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5"
    >
      <SkeletonLine width="40%" height={10} />
      <div
        className="mt-4 skel"
        style={{ height, borderRadius: 8 }}
        aria-hidden
      />
    </div>
  );
}

export function SkeletonTableRow({ cols = 6 }: { cols?: number }) {
  return (
    <tr className="border-b border-[var(--color-border)]">
      {Array.from({ length: cols }).map((_, i) => (
        <td key={i} className="px-3 py-3">
          <SkeletonLine width={i === 0 ? "70%" : "60%"} height={12} />
        </td>
      ))}
    </tr>
  );
}
