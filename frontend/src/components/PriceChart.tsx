import { useEffect, useRef, useState } from "react";
import {
  createChart,
  createSeriesMarkers,
  CandlestickSeries,
  LineSeries,
  AreaSeries,
  LineStyle,
  type IChartApi,
  type IPriceLine,
  type ISeriesApi,
  type SeriesType,
  type UTCTimestamp,
} from "lightweight-charts";
import { readThemeColors, type ThemeColors } from "@/lib/tokens";
import type { OHLCVBar, PositionData } from "@/lib/types";

// Resubscribe to `<html data-theme>` changes so chart options can be re-applied
// when the operator switches themes at runtime.
function useThemeColors(): ThemeColors {
  const [themed, setThemed] = useState<ThemeColors>(() => readThemeColors());
  useEffect(() => {
    const update = () => setThemed(readThemeColors());
    // Initial re-read after mount (CSS vars may resolve late on first paint).
    update();
    const obs = new MutationObserver(update);
    obs.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
    return () => obs.disconnect();
  }, []);
  return themed;
}

export type ChartType = "candles" | "line" | "area";

export interface SignalMarker {
  time: string;          // ISO timestamp
  direction: "buy" | "sell";
  executed: boolean;     // filled vs hollow marker
  score?: number;        // combined_score for tooltip
  /** When ≥ 2, render a cluster marker (amber circle with the count) at this time
   *  instead of a single arrow. Direction/score are kept for tooltip. */
  cluster?: number;
}

interface PriceChartProps {
  bars: OHLCVBar[];
  height?: number | string;
  autoFit?: boolean;
  className?: string;
  positions?: PositionData[];
  /** Signal direction markers — buy/sell arrows on candles */
  markers?: SignalMarker[];
  initialVisibleBars?: number;
  symbol?: string;
  /** Series style: candlestick, line, or filled area of close prices. Default: candles. */
  chartType?: ChartType;
}

/**
 * Derive per-symbol candlestick price format so the Y-axis renders
 * forex-scale moves correctly.
 */
function priceFormatForSymbol(symbol: string | undefined) {
  if (!symbol) return { type: "price" as const, precision: 2, minMove: 0.01 };
  if (symbol.includes("JPY")) {
    return { type: "price" as const, precision: 3, minMove: 0.001 };
  }
  if (symbol === "XAUUSD" || symbol === "ETHUSD" || symbol === "BTCUSD") {
    return { type: "price" as const, precision: 2, minMove: 0.01 };
  }
  // Default forex (EURUSD / USDCAD / GBPUSD / AUDUSD / etc.)
  return { type: "price" as const, precision: 5, minMove: 0.00001 };
}

interface CandleDatum {
  time: UTCTimestamp;
  open: number;
  high: number;
  low: number;
  close: number;
}

function toUtcSeconds(isoString: string): UTCTimestamp {
  // Backend returns naive-UTC ISO strings (e.g. "2026-04-15T05:00:00") with
  // no timezone suffix. Without a 'Z', JS Date.parse treats the value as
  // *local* time, which silently corrupts the underlying timestamp. Force
  // UTC interpretation by appending 'Z' when absent.
  const iso = /[zZ]|[+-]\d{2}:?\d{2}$/.test(isoString) ? isoString : isoString + "Z";
  const ms = Date.parse(iso);
  if (Number.isNaN(ms)) return 0 as UTCTimestamp;
  // Lightweight-charts renders axis labels using UTC parts of the timestamp.
  // To display everything in the user's local TZ we pre-shift the value by
  // the browser's UTC offset — purely cosmetic; underlying OHLCV data keeps
  // its true-UTC semantics. getTimezoneOffset() returns minutes added to
  // local to yield UTC (e.g. WIB = -420), so subtracting it shifts forward.
  const offsetMs = new Date(ms).getTimezoneOffset() * 60_000;
  return Math.floor((ms - offsetMs) / 1000) as UTCTimestamp;
}

function dedupeSortedAscending(bars: OHLCVBar[]): CandleDatum[] {
  const mapped = bars.map((b) => ({
    time: toUtcSeconds(b.time),
    open: b.open,
    high: b.high,
    low: b.low,
    close: b.close,
  }));
  mapped.sort((a, b) => a.time - b.time);
  // Lightweight-charts rejects duplicate timestamps in strict order.
  const out: CandleDatum[] = [];
  let lastT = -1;
  for (const c of mapped) {
    if (c.time === lastT) {
      out[out.length - 1] = c;
    } else {
      out.push(c);
      lastT = c.time;
    }
  }
  return out;
}

export function PriceChart({
  bars,
  height = 320,
  autoFit = true,
  className = "",
  positions,
  markers,
  initialVisibleBars = 300,
  symbol,
  chartType = "candles",
}: PriceChartProps) {
  const themed = useThemeColors();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<SeriesType> | null>(null);
  const chartTypeRef = useRef<ChartType>(chartType);
  // Active price-line handles, keyed by `${ticket}-${kind}` (entry/sl/tp).
  const priceLinesRef = useRef<Map<string, IPriceLine>>(new Map());
  const markersRef = useRef<ReturnType<typeof createSeriesMarkers> | null>(null);
  // Track whether we've already set the initial visible range so a
  // subsequent candle-refresh doesn't snap the user's pan/zoom back to
  // the default window.
  const initialRangeAppliedRef = useRef<boolean>(false);

  // Create chart once
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const c = readThemeColors();
    const chart = createChart(container, {
      autoSize: true,
      layout: {
        background: { color: c.panel },
        textColor: c.textMuted,
        fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif",
      },
      grid: {
        vertLines: { color: c.border },
        horzLines: { color: c.border },
      },
      rightPriceScale: { borderColor: c.border },
      timeScale: {
        borderColor: c.border,
        timeVisible: true,
        secondsVisible: false,
        // "Shift end of the chart from the right border" — keeps a gap of
        // `rightOffset` bars between the last candle and the right edge, so
        // the newest candle isn't covered by entry/SL price-axis labels.
        rightOffset: 50,
      },
      crosshair: { mode: 1 },
    });

    chartRef.current = chart;

    return () => {
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
      priceLinesRef.current.clear();
      markersRef.current = null;
    };
  }, []);

  // (Re)create the series whenever chartType changes. Candles/Line/Area are
  // distinct series types in lightweight-charts, so swapping requires
  // removeSeries + addSeries. Bars/markers/positions effects repopulate on
  // the new series via their own deps.
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    // Tear down old series (drops its markers + price lines with it).
    if (seriesRef.current) {
      try {
        chart.removeSeries(seriesRef.current);
      } catch {
        // chart already disposed
      }
      seriesRef.current = null;
      priceLinesRef.current.clear();
      markersRef.current = null;
    }

    const priceFormat = priceFormatForSymbol(symbol);
    const c = readThemeColors();
    if (chartType === "candles") {
      seriesRef.current = chart.addSeries(CandlestickSeries, {
        upColor: c.profit,
        downColor: c.loss,
        wickUpColor: c.profit,
        wickDownColor: c.loss,
        borderVisible: false,
        priceFormat,
      });
    } else if (chartType === "line") {
      seriesRef.current = chart.addSeries(LineSeries, {
        color: c.primary,
        lineWidth: 2,
        priceFormat,
      });
    } else {
      seriesRef.current = chart.addSeries(AreaSeries, {
        lineColor: c.primary,
        topColor: `${c.primary}55`,
        bottomColor: `${c.primary}00`,
        lineWidth: 2,
        priceFormat,
      });
    }

    chartTypeRef.current = chartType;
    initialRangeAppliedRef.current = false;
  }, [chartType, symbol]);

  // Re-apply chart + series colors whenever the theme swatch switches.
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    chart.applyOptions({
      layout: {
        background: { color: themed.panel },
        textColor: themed.textMuted,
      },
      grid: {
        vertLines: { color: themed.border },
        horzLines: { color: themed.border },
      },
      rightPriceScale: { borderColor: themed.border },
      timeScale: { borderColor: themed.border },
    });
    const series = seriesRef.current;
    if (!series) return;
    const currentType = chartTypeRef.current;
    if (currentType === "candles") {
      series.applyOptions({
        upColor: themed.profit,
        downColor: themed.loss,
        wickUpColor: themed.profit,
        wickDownColor: themed.loss,
      });
    } else if (currentType === "line") {
      series.applyOptions({ color: themed.primary });
    } else {
      series.applyOptions({
        lineColor: themed.primary,
        topColor: `${themed.primary}55`,
        bottomColor: `${themed.primary}00`,
      });
    }
  }, [themed]);

  // Push data whenever bars (or series type) change. For Line/Area series
  // we feed {time, value: close}; for Candles we feed the full OHLC tuple.
  useEffect(() => {
    const series = seriesRef.current;
    const chart = chartRef.current;
    if (!series || !chart) return;
    const candles = dedupeSortedAscending(bars);
    if (chartTypeRef.current === "candles") {
      series.setData(candles);
    } else {
      series.setData(candles.map((c) => ({ time: c.time, value: c.close })));
    }
    const data = candles;
    if (data.length === 0) return;
    if (!initialRangeAppliedRef.current) {
      // Default view: show only the last `initialVisibleBars` candles
      // (300 by default). User can still scroll/zoom to see the full
      // 1000-bar buffer loaded in memory. This matches TradingView/MT5's
      // "recent bars in focus, history just a scroll away" pattern.
      const total = data.length;
      const visible = Math.min(initialVisibleBars, total);
      // Extend the right edge past the last bar to expose the rightOffset
      // gap (MT5 "shift end of chart" behavior).
      const rightGap = 50;
      try {
        chart.timeScale().setVisibleLogicalRange({
          from: total - visible,
          to: total - 1 + rightGap,
        });
      } catch {
        if (autoFit) chart.timeScale().fitContent();
      }
      initialRangeAppliedRef.current = true;
    }
    // Subsequent updates: preserve user's current pan/zoom. Only fit
    // content if explicitly requested AND no data had been shown before.
  }, [bars, autoFit, initialVisibleBars, chartType]);

  // Apply signal direction markers (buy/sell arrows on candles)
  useEffect(() => {
    const series = seriesRef.current;
    if (!series) return;

    const mc = readThemeColors();
    const sorted = (markers ?? [])
      .filter((m) => m.direction === "buy" || m.direction === "sell")
      .map((m) => {
        if (m.cluster && m.cluster >= 2) {
          // Cluster marker — amber circle with count, centered in-bar.
          return {
            time: toUtcSeconds(m.time),
            position: "inBar" as const,
            color: mc.warn,
            shape: "circle" as const,
            text: String(m.cluster),
            size: 1.4,
          };
        }
        return {
          time: toUtcSeconds(m.time),
          position: m.direction === "buy" ? "belowBar" as const : "aboveBar" as const,
          color: m.direction === "buy" ? mc.profit : mc.loss,
          shape: m.direction === "buy" ? "arrowUp" as const : "arrowDown" as const,
          text: m.executed
            ? (m.direction === "buy" ? "BUY" : "SELL")
            : "",
          size: m.executed ? 1 : 0.5,
        };
      })
      .sort((a, b) => a.time - b.time);

    if (markersRef.current) {
      markersRef.current.setMarkers(sorted);
    } else if (sorted.length > 0) {
      markersRef.current = createSeriesMarkers(series, sorted);
    }
  }, [markers, chartType, themed]);

  // Reconcile open-position overlays (entry/SL/TP price lines)
  useEffect(() => {
    const series = seriesRef.current;
    if (!series) return;

    const pc = readThemeColors();
    const wanted = new Map<string, { price: number; color: string; title: string; style: LineStyle }>();
    if (positions && positions.length > 0) {
      for (const pos of positions) {
        const ticket = pos.ticket;
        const sign = pos.direction === "buy" ? "▲" : "▼";
        // Entry — solid neutral
        wanted.set(`${ticket}-entry`, {
          price: pos.entry_price,
          color: pc.textMuted,
          title: `${sign} #${ticket} entry`,
          style: LineStyle.Solid,
        });
        // SL — dashed red
        if (pos.current_stop > 0) {
          wanted.set(`${ticket}-sl`, {
            price: pos.current_stop,
            color: pc.loss,
            title: `SL #${ticket}`,
            style: LineStyle.Dashed,
          });
        }
        // TP — dashed green (only if real)
        if (pos.take_profit && pos.take_profit > 0) {
          wanted.set(`${ticket}-tp`, {
            price: pos.take_profit,
            color: pc.profit,
            title: `TP #${ticket}`,
            style: LineStyle.Dashed,
          });
        }
      }
    }

    const lines = priceLinesRef.current;
    // Remove lines for closed/unwanted positions
    for (const [key, handle] of lines) {
      if (!wanted.has(key)) {
        try {
          series.removePriceLine(handle);
        } catch {
          // already removed
        }
        lines.delete(key);
      }
    }
    // Add or refresh wanted lines
    for (const [key, spec] of wanted) {
      const existing = lines.get(key);
      if (existing) {
        // Update price (for trailing SLs that move)
        try {
          existing.applyOptions({
            price: spec.price,
            color: spec.color,
            title: spec.title,
          });
        } catch {
          // fall through and recreate
        }
        continue;
      }
      const handle = series.createPriceLine({
        price: spec.price,
        color: spec.color,
        lineWidth: 1,
        lineStyle: spec.style,
        axisLabelVisible: true,
        title: spec.title,
      });
      lines.set(key, handle);
    }

    // Widen the price-scale range so entry/SL/TP overlays are always
    // inside the visible Y-axis. Default autoscale only fits the candle
    // series, so a stop-loss far from current price gets clipped off.
    const overlayPrices: number[] = [];
    for (const spec of wanted.values()) {
      if (Number.isFinite(spec.price) && spec.price > 0) overlayPrices.push(spec.price);
    }
    if (overlayPrices.length > 0) {
      const overlayMin = Math.min(...overlayPrices);
      const overlayMax = Math.max(...overlayPrices);
      series.applyOptions({
        autoscaleInfoProvider: (original: () => { priceRange: { minValue: number; maxValue: number } } | null) => {
          const info = original();
          if (!info) {
            return { priceRange: { minValue: overlayMin, maxValue: overlayMax } };
          }
          return {
            ...info,
            priceRange: {
              minValue: Math.min(info.priceRange.minValue, overlayMin),
              maxValue: Math.max(info.priceRange.maxValue, overlayMax),
            },
          };
        },
      });
    } else {
      // No positions → restore default candle-only autoscale.
      series.applyOptions({ autoscaleInfoProvider: undefined });
    }
  }, [positions, chartType, themed]);

  return (
    <div
      ref={containerRef}
      className={className}
      style={{ width: "100%", height }}
    />
  );
}
