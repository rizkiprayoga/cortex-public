import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type {
  BalanceOperationsResponse,
  TradeHistoryResponse,
  EquityCurveResponse,
  TradeTimelineResponse,
  TradingMetrics,
} from "@/lib/types";

export function useTradeTimeline(ticket: number | null, enabled = true) {
  return useQuery<TradeTimelineResponse>({
    queryKey: ["trade-timeline", ticket],
    queryFn: () =>
      api.get<TradeTimelineResponse>(
        `/api/history/trade-timeline/${ticket}`,
      ),
    enabled: enabled && ticket != null && ticket > 0,
    staleTime: 30_000,
    retry: 1,
  });
}

// History hooks poll every 30s — aligned with the server-side cache TTL
// in src/api/routes/history.py (_HISTORY_CACHE_TTL_SEC = 30), so the UI
// picks up newly-closed trades without hammering the backend.

export function useTradeHistory(
  page: number = 1,
  pageSize: number = 50,
  symbol?: string,
) {
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
  });
  if (symbol) params.set("symbol", symbol);

  return useQuery<TradeHistoryResponse>({
    queryKey: ["trade-history", page, pageSize, symbol],
    queryFn: () =>
      api.get<TradeHistoryResponse>(`/api/history/trades?${params}`),
    refetchInterval: 30_000,
    staleTime: 15_000,
    retry: 1,
  });
}

export function useEquityCurve(limit: number = 500) {
  return useQuery<EquityCurveResponse>({
    queryKey: ["equity-curve", limit],
    queryFn: () =>
      api.get<EquityCurveResponse>(`/api/history/equity?limit=${limit}`),
    refetchInterval: 30_000,
    staleTime: 15_000,
    retry: 1,
  });
}

export function useTradingMetrics(symbol?: string) {
  const params = symbol ? `?symbol=${symbol}` : "";
  return useQuery<TradingMetrics>({
    queryKey: ["trading-metrics", symbol],
    queryFn: () => api.get<TradingMetrics>(`/api/history/metrics${params}`),
    refetchInterval: 30_000,
    staleTime: 15_000,
    retry: 1,
  });
}

export function useAccountLedger(days: number = 365) {
  return useQuery<BalanceOperationsResponse>({
    queryKey: ["account-ledger", days],
    queryFn: () =>
      api.get<BalanceOperationsResponse>(
        `/api/history/account-ledger?days=${days}`,
      ),
    refetchInterval: 30_000,
    staleTime: 15_000,
    retry: 1,
  });
}
