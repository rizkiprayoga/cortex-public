import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type { SystemStatusData, BotStatusData, LockStatusData } from "@/lib/types";

// Polling cadences tuned 2026-04-19 (P-1b). These endpoints are mostly
// static (bot state, lock state change manually; system health is
// slower-moving than per-second tick data). Mutations (pause/resume/lock)
// should `queryClient.invalidateQueries` the affected key for instant UI
// feedback rather than leaning on short polls.

export function useSystemStatus() {
  return useQuery<SystemStatusData>({
    queryKey: ["system-status"],
    queryFn: () => api.get<SystemStatusData>("/api/system/status"),
    refetchInterval: 15_000,
    retry: 1,
  });
}

export function useBotStatus() {
  return useQuery<BotStatusData>({
    queryKey: ["bot-status"],
    queryFn: () => api.get<BotStatusData>("/api/bot/status"),
    refetchInterval: 60_000,
    retry: 1,
  });
}

export function useLockStatus() {
  return useQuery<LockStatusData>({
    queryKey: ["lock-status"],
    queryFn: () => api.get<LockStatusData>("/api/system/lock-status"),
    refetchInterval: 60_000,
  });
}
