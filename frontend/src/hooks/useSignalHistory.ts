import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-client";

interface SignalAuditEntry {
  timestamp: string;
  symbol: string;
  direction: string | null;
  combined_score: number | null;
  should_trade: boolean;
  executed: boolean;
  regime: string | null;
  block_reason: string | null;
}

interface SignalAuditResponse {
  items: SignalAuditEntry[];
  total: number;
  page: number;
  page_size: number;
}

export function useSignalHistory(symbol: string, limit = 200) {
  return useQuery<SignalAuditEntry[]>({
    queryKey: ["signal-history", symbol, limit],
    queryFn: async () => {
      const res = await api.get<SignalAuditResponse>(
        `/api/history/signal-audit?symbol=${symbol}&page=1&page_size=${limit}`
      );
      return res.items;
    },
    staleTime: 60_000,
    retry: 1,
    enabled: !!symbol,
  });
}
