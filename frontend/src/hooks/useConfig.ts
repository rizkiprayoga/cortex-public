import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type { RiskConfig, RiskConfigUpdate } from "@/lib/types";

export function useRiskConfig() {
  return useQuery<RiskConfig>({
    queryKey: ["risk-config"],
    queryFn: () => api.get<RiskConfig>("/api/config/risk"),
    retry: 1,
  });
}

export function useUpdateRiskConfig() {
  const qc = useQueryClient();
  return useMutation<RiskConfig, Error, RiskConfigUpdate>({
    mutationFn: (body) => api.post<RiskConfig>("/api/config/risk", body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["risk-config"] });
    },
  });
}
