import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type {
  AccuracyTimeSeriesResponse,
  ModelSummaryResponse,
  ModelVersionHistoryResponse,
} from "@/lib/types";

export function useModelSummary() {
  return useQuery<ModelSummaryResponse>({
    queryKey: ["model-summary"],
    queryFn: () => api.get<ModelSummaryResponse>("/api/models/summary"),
    refetchInterval: 60_000,
    staleTime: 30_000,
  });
}

export function useAccuracyTimeSeries(symbol: string, days: number = 30) {
  return useQuery<AccuracyTimeSeriesResponse>({
    queryKey: ["accuracy-timeseries", symbol, days],
    queryFn: () =>
      api.get<AccuracyTimeSeriesResponse>(
        `/api/models/accuracy/${encodeURIComponent(symbol)}?days=${days}`,
      ),
    enabled: !!symbol,
    staleTime: 5 * 60_000,
  });
}

export function useModelVersionHistory(
  modelName: string | null,
  limit: number = 20,
) {
  return useQuery<ModelVersionHistoryResponse>({
    queryKey: ["model-versions", modelName, limit],
    queryFn: () =>
      api.get<ModelVersionHistoryResponse>(
        `/api/models/versions/${encodeURIComponent(modelName!)}?limit=${limit}`,
      ),
    enabled: !!modelName,
    staleTime: 5 * 60_000,
  });
}
