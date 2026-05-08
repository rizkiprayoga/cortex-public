import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type { SignalData } from "@/lib/types";

export function useSignal(symbol: string) {
  return useQuery<SignalData>({
    queryKey: ["signal", symbol],
    queryFn: () => api.get<SignalData>(`/api/live/signals/${symbol}`),
    refetchInterval: 3000,
    retry: 1,
    enabled: !!symbol,
  });
}
