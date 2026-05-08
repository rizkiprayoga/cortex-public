import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type { CandlesResponse, ChartTimeframe } from "@/lib/types";

interface UseCandlesOptions {
  symbol: string;
  timeframe?: ChartTimeframe;
  limit?: number;
  refetchInterval?: number;
  enabled?: boolean;
}

export function useCandles({
  symbol,
  timeframe = "H1",
  limit = 1000,
  refetchInterval = 15_000,
  enabled = true,
}: UseCandlesOptions) {
  return useQuery<CandlesResponse>({
    queryKey: ["candles", symbol, timeframe, limit],
    queryFn: () =>
      api.get<CandlesResponse>(
        `/api/live/candles/${encodeURIComponent(symbol)}?timeframe=${timeframe}&limit=${limit}`,
      ),
    refetchInterval,
    enabled: enabled && !!symbol,
    staleTime: refetchInterval / 2,
  });
}

/**
 * Lightweight per-symbol last-price hook for watchlist-style cards.
 * Returns last close + % change vs previous bar. 2-bar fetch, 30s refresh.
 */
export function useLatestPrice(symbol: string, enabled = true) {
  const q = useQuery<CandlesResponse>({
    queryKey: ["latest-price", symbol],
    queryFn: () =>
      api.get<CandlesResponse>(
        `/api/live/candles/${encodeURIComponent(symbol)}?timeframe=H1&limit=2`,
      ),
    refetchInterval: 30_000,
    enabled: enabled && !!symbol,
    staleTime: 15_000,
  });
  // API response field is `bars`, not `candles` — was silently returning
  // empty on an undefined key, which is why RegimeCard's price + delta
  // slots kept rendering "--".
  const bars = q.data?.bars ?? [];
  let price: number | null = null;
  let changePct: number | null = null;
  if (bars.length >= 1) {
    price = bars[bars.length - 1].close;
  }
  if (bars.length >= 2) {
    const prev = bars[bars.length - 2].close;
    if (prev > 0) changePct = ((bars[bars.length - 1].close - prev) / prev) * 100;
  }
  return { price, changePct, isLoading: q.isLoading };
}
