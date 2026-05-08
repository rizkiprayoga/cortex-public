import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type {
  BacktestRunsResponse,
  BacktestStatusResponse,
  BacktestSubmitResponse,
} from "@/lib/types";

export function useBacktestRuns() {
  return useQuery<BacktestRunsResponse>({
    queryKey: ["backtest-runs"],
    queryFn: () => api.get<BacktestRunsResponse>("/api/backtest/runs"),
    retry: 1,
  });
}

export function useBacktestStatus(runId: string | null) {
  return useQuery<BacktestStatusResponse>({
    queryKey: ["backtest-status", runId],
    queryFn: () =>
      api.get<BacktestStatusResponse>(`/api/backtest/status/${runId}`),
    enabled: !!runId,
    refetchInterval: (query) => {
      const status = query.state.data?.run.status;
      // Poll every 2s while pending/running, stop when done/failed
      return status === "pending" || status === "running" ? 2000 : false;
    },
    retry: 1,
  });
}

export function useSubmitBacktest() {
  const qc = useQueryClient();
  return useMutation<
    BacktestSubmitResponse,
    Error,
    {
      symbol: string;
      timeframe: string;
      start_date: string;
      end_date: string;
      initial_equity: number;
    }
  >({
    mutationFn: (body) =>
      api.post<BacktestSubmitResponse>("/api/backtest/submit", body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["backtest-runs"] });
    },
  });
}
