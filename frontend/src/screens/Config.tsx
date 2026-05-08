import { useEffect, useState } from "react";
import { useRiskConfig, useUpdateRiskConfig } from "@/hooks/useConfig";
import { useTheme, THEMES, type Theme } from "@/hooks/useTheme";
import { useDensity, DENSITIES, type Density } from "@/hooks/useDensity";
import type { RiskConfig } from "@/lib/types";

const CONFIRM_TOKEN = "CONFIRM_HARD_HALT_CHANGE";

const HARD_HALT_FIELDS = new Set<keyof RiskConfig>([
  "max_daily_loss_hard_pct",
  "max_weekly_loss_hard_pct",
  "max_peak_drawdown_pct",
]);

interface FieldDef {
  key: keyof RiskConfig;
  label: string;
  hint?: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  isHard: boolean;
}

const BREAKER_FIELDS: FieldDef[] = [
  {
    key: "max_daily_loss_soft_pct",
    label: "Daily Soft",
    hint: "Reduce position size by 50%",
    unit: "%",
    min: 0.5,
    max: 10,
    step: 0.5,
    isHard: false,
  },
  {
    key: "max_daily_loss_hard_pct",
    label: "Daily Hard",
    hint: "Flatten positions, halt to end-of-day",
    unit: "%",
    min: 1,
    max: 15,
    step: 0.5,
    isHard: true,
  },
  {
    key: "max_weekly_loss_soft_pct",
    label: "Weekly Soft",
    hint: "50% sizing until Monday",
    unit: "%",
    min: 1,
    max: 20,
    step: 0.5,
    isHard: false,
  },
  {
    key: "max_weekly_loss_hard_pct",
    label: "Weekly Hard",
    hint: "Halt until weekly reset",
    unit: "%",
    min: 2,
    max: 25,
    step: 0.5,
    isHard: true,
  },
  {
    key: "max_peak_drawdown_pct",
    label: "Peak Drawdown",
    hint: "Sticky halt — requires manual reset",
    unit: "%",
    min: 3,
    max: 30,
    step: 1,
    isHard: true,
  },
];

const PORTFOLIO_FIELDS: FieldDef[] = [
  { key: "max_total_exposure_pct", label: "Total Margin Cap", unit: "%", min: 5, max: 50, step: 1, isHard: false, hint: "Sum of margin across all positions" },
  { key: "free_margin_reserve_pct", label: "Free Margin Reserve", unit: "%", min: 5, max: 50, step: 1, isHard: false, hint: "Unused buffer required at order time" },
  { key: "max_concurrent_per_symbol", label: "Max per Symbol", unit: "", min: 1, max: 10, step: 1, isHard: false, hint: "Caps pyramiding" },
  { key: "max_concurrent_total", label: "Max Total Positions", unit: "", min: 1, max: 20, step: 1, isHard: false },
  { key: "max_daily_trades", label: "Max Daily Trades", unit: "", min: 1, max: 50, step: 1, isHard: false, hint: "Rolling 24h window (no fixed reset time)" },
];

// ─── Slider row ──────────────────────────────────────────────────────

function SliderRow({
  field,
  value,
  onChange,
  color,
}: {
  field: FieldDef;
  value: number;
  onChange: (key: keyof RiskConfig, val: number) => void;
  color: string;
}) {
  const pct = ((value - field.min) / (field.max - field.min)) * 100;
  return (
    <div className="py-3">
      <div className="flex items-center gap-4">
        <div className="w-48 flex items-center gap-2">
          <div>
            <p className="text-sm font-medium text-[var(--color-text)]">{field.label}</p>
            {field.hint && (
              <p className="text-[10px] text-[var(--color-text-dim)]">{field.hint}</p>
            )}
          </div>
          {field.isHard && (
            <span
              className="inline-flex items-center px-1.5 py-0.5 rounded-full text-[10px] font-bold"
              style={{ background: "rgba(244,63,94,0.18)", color: "var(--chip-loss-fg)" }}
            >
              HARD
            </span>
          )}
        </div>
        <div className="flex-1 relative h-6 flex items-center">
          {/* Track background */}
          <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 h-1.5 rounded-full bg-[var(--color-panel-hi)]" />
          {/* Filled portion */}
          <div
            className="absolute top-1/2 -translate-y-1/2 h-1.5 rounded-full pointer-events-none"
            style={{ width: `${pct.toFixed(1)}%`, background: color }}
          />
          {/* Native range for accessibility */}
          <input
            type="range"
            min={field.min}
            max={field.max}
            step={field.step}
            value={value}
            onChange={(e) => onChange(field.key, Number(e.target.value))}
            className="relative w-full h-6 appearance-none bg-transparent cursor-pointer
              [&::-webkit-slider-thumb]:appearance-none
              [&::-webkit-slider-thumb]:w-4
              [&::-webkit-slider-thumb]:h-4
              [&::-webkit-slider-thumb]:rounded-full
              [&::-webkit-slider-thumb]:bg-[var(--color-text)]
              [&::-webkit-slider-thumb]:border-2
              [&::-moz-range-thumb]:w-4
              [&::-moz-range-thumb]:h-4
              [&::-moz-range-thumb]:rounded-full
              [&::-moz-range-thumb]:bg-[var(--color-text)]
              [&::-moz-range-thumb]:border-2"
            style={
              {
                "--tw-ring-offset-color": "transparent",
              } as React.CSSProperties
            }
          />
          <style>{`
            input[type="range"][data-slider-color="${color}"]::-webkit-slider-thumb { border-color: ${color}; }
          `}</style>
        </div>
        <span className="mono w-16 text-right text-sm font-semibold">
          {value}
          {field.unit}
        </span>
      </div>
    </div>
  );
}

// ─── Theme swatch ────────────────────────────────────────────────────

function ThemeSwitcher() {
  const { theme, setTheme } = useTheme();
  const SWATCHES: Record<Theme, { bg: string; panel: string; accent: string; label: string }> = {
    dark: { bg: "#05070d", panel: "#111827", accent: "#06b6d4", label: "Dark" },
    dim: { bg: "#1c1917", panel: "#2d2a27", accent: "#22d3ee", label: "Dim" },
    light: { bg: "#f1f5f9", panel: "#ffffff", accent: "#0891b2", label: "Light" },
    coffee: { bg: "#f0e4d4", panel: "#fdf8f0", accent: "#6d2932", label: "Coffee" },
  };

  return (
    <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5">
      <p className="section-label mb-3">Appearance</p>
      <p className="text-xs text-[var(--color-text-muted)] mb-4">
        Theme applies instantly and persists in localStorage.
      </p>
      <div className="grid grid-cols-2 gap-2">
        {THEMES.map((t) => {
          const swatch = SWATCHES[t];
          const active = theme === t;
          return (
            <button
              key={t}
              onClick={() => setTheme(t)}
              className={`rounded-lg p-3 text-left transition-all hover:-translate-y-px ${
                active ? "ring-2" : "border border-[var(--color-border)]"
              }`}
              style={{
                background: swatch.panel,
                borderColor: active ? swatch.accent : undefined,
                boxShadow: active ? `0 0 0 2px ${swatch.accent}` : undefined,
              }}
            >
              <div className="flex items-center gap-2 mb-2">
                <span
                  className="w-4 h-4 rounded-full"
                  style={{ background: swatch.accent }}
                />
                <span className="text-sm font-semibold" style={{ color: t === "light" || t === "coffee" ? "#0f172a" : "#f8fafc" }}>
                  {swatch.label}
                </span>
                {active && (
                  <span className="ml-auto text-[10px] mono" style={{ color: swatch.accent }}>
                    ✓
                  </span>
                )}
              </div>
              <div
                className="h-6 rounded"
                style={{
                  background: `linear-gradient(90deg, ${swatch.bg} 0%, ${swatch.panel} 100%)`,
                }}
              />
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ─── Density switcher ────────────────────────────────────────────────

function DensitySwitcher() {
  const { density, setDensity } = useDensity();
  const LABELS: Record<Density, { label: string; sub: string }> = {
    default: { label: "Default", sub: "Comfortable paddings, larger type" },
    compact: { label: "Compact", sub: "Tighter spacing, more rows per screen" },
  };

  return (
    <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5">
      <p className="section-label mb-3">Density</p>
      <p className="text-xs text-[var(--color-text-muted)] mb-4">
        Applies instantly and persists in localStorage.
      </p>
      <div className="grid grid-cols-2 gap-2">
        {DENSITIES.map((d) => {
          const meta = LABELS[d];
          const active = density === d;
          return (
            <button
              key={d}
              onClick={() => setDensity(d)}
              className={`rounded-lg p-3 text-left transition-all hover:-translate-y-px ${
                active ? "ring-2 ring-[var(--color-accent)]" : "border border-[var(--color-border)]"
              }`}
            >
              <div className="flex items-center gap-2 mb-1">
                <span className="text-sm font-semibold">{meta.label}</span>
                {active && (
                  <span className="ml-auto text-[10px] mono text-[var(--color-accent)]">✓</span>
                )}
              </div>
              <p className="text-[11px] text-[var(--color-text-muted)]">{meta.sub}</p>
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ─── Screen ──────────────────────────────────────────────────────────

export function Config() {
  const { data: config, isLoading } = useRiskConfig();
  const mutation = useUpdateRiskConfig();
  const [draft, setDraft] = useState<RiskConfig | null>(null);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [confirmText, setConfirmText] = useState("");

  useEffect(() => {
    if (config && !draft) setDraft({ ...config });
  }, [config, draft]);

  if (isLoading || !draft) {
    return (
      <p className="text-center text-sm text-[var(--color-text-muted)] py-10">
        Loading risk config…
      </p>
    );
  }

  const changes: Partial<RiskConfig> = {};
  if (config) {
    for (const key of Object.keys(draft) as (keyof RiskConfig)[]) {
      if (draft[key] !== config[key]) {
        (changes as Record<string, number>)[key] = draft[key] as number;
      }
    }
  }
  const hasChanges = Object.keys(changes).length > 0;
  const touchesHard = Object.keys(changes).some((k) =>
    HARD_HALT_FIELDS.has(k as keyof RiskConfig),
  );

  const handleChange = (key: keyof RiskConfig, val: number) =>
    setDraft((d) => (d ? { ...d, [key]: val } : d));

  const handleSave = () => {
    if (touchesHard) {
      setConfirmOpen(true);
      return;
    }
    mutation.mutate(changes, {
      onSuccess: (updated) => setDraft({ ...updated }),
    });
  };

  const handleConfirm = () => {
    if (confirmText !== CONFIRM_TOKEN) return;
    mutation.mutate(
      { ...changes, confirmation: CONFIRM_TOKEN },
      {
        onSuccess: (updated) => {
          setDraft({ ...updated });
          setConfirmOpen(false);
          setConfirmText("");
        },
      },
    );
  };

  const handleReset = () => {
    if (config) setDraft({ ...config });
  };

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-xl font-semibold text-[var(--color-text)]">Risk configuration</h1>
        <p className="text-xs text-[var(--color-text-dim)] mt-0.5">
          Hot-reload limits without restarting the bot.
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: sliders */}
        <div className="lg:col-span-2 space-y-6">
          <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-6">
            <div className="mb-4">
              <h3 className="text-base font-semibold text-[var(--color-text)]">
                Circuit breakers
              </h3>
              <p className="text-xs text-[var(--color-text-muted)] mt-0.5">
                Loss caps that trigger protective halts.
              </p>
            </div>
            <div className="divide-y divide-[var(--color-border)]">
              {BREAKER_FIELDS.map((f) => (
                <SliderRow
                  key={f.key}
                  field={f}
                  value={draft[f.key] as number}
                  onChange={handleChange}
                  color={f.isHard ? "var(--color-loss)" : "var(--color-primary)"}
                />
              ))}
            </div>
          </div>

          <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-6">
            <div className="mb-4">
              <h3 className="text-base font-semibold text-[var(--color-text)]">Portfolio caps</h3>
              <p className="text-xs text-[var(--color-text-muted)] mt-0.5">
                Position count + margin + daily throttle.
              </p>
            </div>
            <div className="divide-y divide-[var(--color-border)]">
              {PORTFOLIO_FIELDS.map((f) => (
                <SliderRow
                  key={f.key}
                  field={f}
                  value={draft[f.key] as number}
                  onChange={handleChange}
                  color="var(--indigo)"
                />
              ))}
            </div>
          </div>
        </div>

        {/* Right: save panel + info cards */}
        <div className="space-y-4">
          <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5">
            <p className="section-label mb-2">Pending changes</p>
            {hasChanges ? (
              <>
                <ul className="text-[11px] text-[var(--color-text)] space-y-1 mb-3">
                  {Object.entries(changes).map(([k, v]) => (
                    <li key={k} className="mono">
                      {k}: {(config as Record<string, number>)[k]} →{" "}
                      <b style={{ color: HARD_HALT_FIELDS.has(k as keyof RiskConfig) ? "var(--color-loss)" : "var(--color-profit)" }}>
                        {v as number}
                      </b>
                    </li>
                  ))}
                </ul>
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleReset}
                    className="flex-1 py-2 rounded-lg bg-[var(--color-panel-hi)] text-[var(--color-text-muted)] text-xs hover:text-[var(--color-text)]"
                  >
                    Reset
                  </button>
                  <button
                    onClick={handleSave}
                    disabled={mutation.isPending}
                    className="flex-1 py-2 rounded-lg text-white text-sm font-medium bg-brand-gradient hover:brightness-110 disabled:opacity-50"
                  >
                    {mutation.isPending ? "Saving…" : "Save"}
                  </button>
                </div>
              </>
            ) : (
              <>
                <p className="text-sm text-[var(--color-text-dim)]">
                  None · all values match server state.
                </p>
                <button
                  className="w-full mt-4 py-2.5 rounded-lg bg-[var(--color-panel-hi)] text-[var(--color-text-dim)] text-sm font-medium cursor-not-allowed"
                  disabled
                >
                  Save changes
                </button>
              </>
            )}
            {mutation.isError && (
              <p className="mt-2 text-xs" style={{ color: "var(--color-loss)" }}>
                {mutation.error.message}
              </p>
            )}
          </div>

          <div
            className="rounded-xl p-5"
            style={{
              background: "rgba(234,179,8,0.06)",
              border: "1px solid rgba(234,179,8,0.3)",
            }}
          >
            <p className="text-xs font-semibold mb-1" style={{ color: "var(--color-warn)" }}>
              HARD halt fields
            </p>
            <p className="text-[11px] text-[var(--color-text-muted)]">
              Changes to fields marked HARD require typing{" "}
              <span className="mono text-[var(--color-text)]">{CONFIRM_TOKEN}</span> to apply.
              Live-change guard.
            </p>
          </div>

          <ThemeSwitcher />
          <DensitySwitcher />

          <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5">
            <p className="section-label mb-2">Other settings</p>
            <p className="text-[11px] text-[var(--color-text-muted)]">
              Per-symbol risk %, timeframes, min confidence, and HMM/LSTM weights are in{" "}
              <span className="mono">settings.yaml</span>. Not exposed in UI to prevent
              accidental production changes.
            </p>
          </div>
        </div>
      </div>

      {/* Hard-halt confirmation modal */}
      {confirmOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/60"
            onClick={() => {
              setConfirmOpen(false);
              setConfirmText("");
            }}
          />
          <div className="relative max-w-md w-full mx-4 rounded-xl border border-[var(--color-border-hi)] bg-[var(--color-panel)] p-6">
            <h3
              className="text-lg font-semibold mb-3"
              style={{ color: "var(--color-loss)" }}
            >
              Confirm HARD halt change
            </h3>
            <p className="text-sm text-[var(--color-text)] mb-3">
              You are changing a hard-halt parameter. This can cause the bot to halt trading
              or flatten all positions.
            </p>
            <div className="mono text-xs bg-[var(--color-panel-hi)] rounded p-3 mb-3 space-y-1">
              {Object.entries(changes)
                .filter(([k]) => HARD_HALT_FIELDS.has(k as keyof RiskConfig))
                .map(([k, v]) => (
                  <div key={k}>
                    {k}: {(config as Record<string, number>)[k]} → {v as number}
                  </div>
                ))}
            </div>
            <p className="text-xs text-[var(--color-text-muted)] mb-2">
              Type <span className="mono font-bold">{CONFIRM_TOKEN}</span> to confirm:
            </p>
            <input
              value={confirmText}
              onChange={(e) => setConfirmText(e.target.value)}
              placeholder={CONFIRM_TOKEN}
              className="w-full mb-3 mono text-xs px-3 py-2 rounded bg-[var(--color-panel-hi)] border border-[var(--color-border)] focus:outline-none focus:border-[var(--color-loss)]"
              autoFocus
            />
            <div className="flex gap-2 justify-end">
              <button
                onClick={() => {
                  setConfirmOpen(false);
                  setConfirmText("");
                }}
                className="px-4 py-1.5 text-sm rounded bg-[var(--color-panel-hi)] text-[var(--color-text-muted)]"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirm}
                disabled={confirmText !== CONFIRM_TOKEN || mutation.isPending}
                className="px-4 py-1.5 text-sm rounded text-white font-medium disabled:opacity-40"
                style={{ background: "var(--color-loss)" }}
              >
                {mutation.isPending ? "Saving…" : "Confirm & save"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
