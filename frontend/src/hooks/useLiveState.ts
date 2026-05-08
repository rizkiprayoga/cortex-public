import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type { LiveStateData } from "@/lib/types";

export function useLiveState() {
  return useQuery<LiveStateData>({
    queryKey: ["live-state"],
    queryFn: () => api.get<LiveStateData>("/api/live/state"),
    refetchInterval: 3000,
    retry: 1,
  });
}
