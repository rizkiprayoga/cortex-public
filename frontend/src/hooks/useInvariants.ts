import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-client";

type InvariantSeverity = "WARN" | "ALERT" | "CRITICAL";

interface InvariantFinding {
  ts: string;
  invariant: string;
  severity: InvariantSeverity;
  passed: boolean;
  message: string;
  symbol: string | null;
  context: Record<string, unknown>;
}

interface InvariantFeed {
  findings: InvariantFinding[];
  count: number;
}

export function useInvariants(limit = 50) {
  return useQuery<InvariantFeed>({
    queryKey: ["invariants-recent", limit],
    queryFn: () =>
      api.get<InvariantFeed>(`/api/invariants/recent?limit=${limit}`),
    staleTime: 30_000,
    refetchInterval: 60_000,
    retry: 1,
  });
}
