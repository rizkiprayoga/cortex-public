import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type {
  AccountInfo,
  AccountSlotsResponse,
  AccountSwitchRequest,
  AccountRegisterRequest,
} from "@/lib/types";

export function useCurrentAccount() {
  return useQuery<AccountInfo>({
    queryKey: ["current-account"],
    queryFn: () => api.get<AccountInfo>("/api/accounts/current"),
    staleTime: 30_000,
    retry: 1,
  });
}

export function useAccountSlots() {
  return useQuery<AccountSlotsResponse>({
    queryKey: ["account-slots"],
    queryFn: () => api.get<AccountSlotsResponse>("/api/accounts/slots"),
    staleTime: 30_000,
    retry: 1,
  });
}

export function useSwitchAccount() {
  const qc = useQueryClient();
  return useMutation<AccountInfo, Error, AccountSwitchRequest>({
    mutationFn: (body) => api.post<AccountInfo>("/api/accounts/switch", body),
    onSuccess: () => {
      // Clear all cached data — old account's data must not linger.
      // removeQueries deletes cache entries; components re-fetch on mount.
      qc.removeQueries();
      qc.invalidateQueries();
    },
  });
}

export function useRegisterAccount() {
  const qc = useQueryClient();
  return useMutation<AccountInfo, Error, AccountRegisterRequest>({
    mutationFn: (body) =>
      api.post<AccountInfo>("/api/accounts/register", body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["account-slots"] });
      qc.invalidateQueries({ queryKey: ["current-account"] });
    },
  });
}
