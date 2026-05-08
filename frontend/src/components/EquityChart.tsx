import { useMemo } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { colors } from "@/lib/tokens";

export interface EquityPoint {
  t: string | number; // ISO string, date, or numeric timestamp
  equity: number;
  drawdown?: number | null; // 0..1 (0 = at peak, 0.05 = 5% below peak)
}

interface EquityChartProps {
  data: EquityPoint[];
  height?: number;
  showDrawdown?: boolean;
  showGrid?: boolean;
  className?: string;
  xTickFormatter?: (value: string | number) => string;
  yTickFormatter?: (value: number) => string;
}

function defaultTimeFormatter(v: string | number): string {
  if (typeof v === "number") return new Date(v).toLocaleDateString();
  const d = new Date(v);
  if (Number.isNaN(d.getTime())) return String(v);
  return d.toLocaleDateString();
}

function defaultYFormatter(v: number): string {
  if (Math.abs(v) >= 1000) return `$${(v / 1000).toFixed(1)}k`;
  return `$${v.toFixed(0)}`;
}

export function EquityChart({
  data,
  height = 240,
  showDrawdown = true,
  showGrid = true,
  className = "",
  xTickFormatter = defaultTimeFormatter,
  yTickFormatter = defaultYFormatter,
}: EquityChartProps) {
  // Drawdown plotted as negative $ on its OWN y-axis (yAxisId="dd") so equity
  // axis can zoom tight on its data range — a $50 swing on a $10k account
  // would otherwise be invisible against a 0–10k scale.
  const chartData = useMemo(() => {
    if (data.length === 0) return [];
    let peak = data[0].equity;
    return data.map((p) => {
      peak = Math.max(peak, p.equity);
      const ddFrac = p.drawdown != null ? p.drawdown : Math.max(0, (peak - p.equity) / peak);
      return {
        t: p.t,
        equity: p.equity,
        drawdown: -ddFrac * peak, // negative dollars from peak
      };
    });
  }, [data]);

  // Equity domain: tight zoom around current values + 0.5% padding on each side.
  const equityDomain = useMemo<[number | string, number | string]>(() => {
    if (chartData.length === 0) return ["auto", "auto"];
    const equities = chartData.map((p) => p.equity);
    const lo = Math.min(...equities);
    const hi = Math.max(...equities);
    const span = Math.max(hi - lo, hi * 0.002); // floor span at 0.2% so flat lines aren't a needle
    return [Math.floor(lo - span * 0.5), Math.ceil(hi + span * 0.5)];
  }, [chartData]);

  return (
    <div className={className} style={{ width: "100%", height }}>
      {/* Legend explaining the two lines */}
      <div className="flex items-center gap-4 px-2 pb-1 text-[11px] text-[var(--color-text-muted)]">
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-0.5" style={{ background: colors.primary }} />
          Equity (account value)
        </span>
        {showDrawdown && (
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-0.5" style={{ background: colors.loss }} />
            Drawdown from peak ($, plotted negative)
          </span>
        )}
      </div>
      <ResponsiveContainer width="100%" height={Math.max(height - 22, 120)}>
        <AreaChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={colors.primary} stopOpacity={0.45} />
              <stop offset="100%" stopColor={colors.primary} stopOpacity={0} />
            </linearGradient>
            <linearGradient id="drawdownGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={colors.loss} stopOpacity={0} />
              <stop offset="100%" stopColor={colors.loss} stopOpacity={0.35} />
            </linearGradient>
          </defs>
          {showGrid && (
            <CartesianGrid stroke={colors.border} strokeDasharray="2 4" vertical={false} />
          )}
          <XAxis
            dataKey="t"
            tickFormatter={xTickFormatter}
            stroke={colors.textDim}
            tick={{ fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            minTickGap={32}
          />
          {/* Equity axis — left side, zoomed tight */}
          <YAxis
            yAxisId="equity"
            tickFormatter={yTickFormatter}
            stroke={colors.textDim}
            tick={{ fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={52}
            domain={equityDomain}
            allowDataOverflow={false}
          />
          {/* Drawdown axis — right side, hidden, only used to scale the dd area */}
          {showDrawdown && (
            <YAxis
              yAxisId="dd"
              orientation="right"
              hide
              domain={["dataMin", 0]}
            />
          )}
          <Tooltip
            contentStyle={{
              background: colors.panelHi,
              border: `1px solid ${colors.border}`,
              borderRadius: 8,
              fontSize: 12,
            }}
            labelStyle={{ color: colors.textMuted }}
            itemStyle={{ color: colors.text }}
            labelFormatter={xTickFormatter}
            formatter={(value: number, name: string) => {
              if (name === "drawdown") {
                return [defaultYFormatter(value), "drawdown ($)"];
              }
              return [defaultYFormatter(value), "equity"];
            }}
          />
          {showDrawdown && (
            <Area
              yAxisId="dd"
              type="monotone"
              dataKey="drawdown"
              stroke={colors.loss}
              strokeWidth={1}
              fill="url(#drawdownGradient)"
              isAnimationActive={false}
            />
          )}
          <Area
            yAxisId="equity"
            type="monotone"
            dataKey="equity"
            stroke={colors.primary}
            strokeWidth={2}
            fill="url(#equityGradient)"
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
