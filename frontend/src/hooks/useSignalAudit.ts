import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type { SignalAuditResponse } from "@/lib/types";

interface UseSignalAuditOpts {
  symbol?: string;         // filter by symbol (omit = all)
  executed?: boolean;      // true=approved, false=blocked, undefined=both
  blockReason?: string;    // substring match on block_reason (e.g. "broker_reject")
  page?: number;
  pageSize?: number;
}

export function useSignalAudit(opts: UseSignalAuditOpts = {}) {
  const {
    symbol,
    executed,
    blockReason,
    page = 1,
    pageSize = 50,
  } = opts;

  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
  });
  if (symbol) params.set("symbol", symbol);
  if (executed !== undefined) params.set("executed", String(executed));
  if (blockReason) params.set("block_reason", blockReason);

  return useQuery<SignalAuditResponse>({
    queryKey: ["signal-audit", symbol, executed, blockReason, page, pageSize],
    queryFn: () =>
      api.get<SignalAuditResponse>(`/api/history/signal-audit?${params}`),
    retry: 1,
    refetchInterval: 30_000,        // auto-refresh every 30s
    staleTime: 15_000,
  });
}
