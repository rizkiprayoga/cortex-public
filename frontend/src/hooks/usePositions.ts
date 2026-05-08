import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type { PositionData } from "@/lib/types";

export function usePositions() {
  return useQuery<PositionData[]>({
    queryKey: ["positions"],
    queryFn: () => api.get<PositionData[]>("/api/live/positions"),
    refetchInterval: 3000,
    retry: 1,
  });
}
