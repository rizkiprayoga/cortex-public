import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type { BacktestDetailResponse } from "@/lib/types";

export function useBacktestDetail(runId: string | null) {
  return useQuery<BacktestDetailResponse>({
    queryKey: ["backtest-detail", runId],
    queryFn: () =>
      api.get<BacktestDetailResponse>(
        `/api/backtest/runs/${encodeURIComponent(runId!)}/detail`,
      ),
    enabled: !!runId,
    staleTime: 30_000,
  });
}
