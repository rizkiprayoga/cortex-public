import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-client";

export interface NewsEvent {
  cb: string;
  event_utc: string;
  blackout_start_utc: string;
  blackout_end_utc: string;
}

export interface NewsSymbolEntry {
  symbol: string;
  central_banks: string[];
  state: "clear" | "blackout" | "post_news";
  active_event: NewsEvent | null;
  next_event: NewsEvent | null;
  exempt: boolean;
}

interface NewsBlackoutResponse {
  generated_at: string;
  symbols: NewsSymbolEntry[];
}

export function useNewsBlackouts() {
  return useQuery<NewsBlackoutResponse>({
    queryKey: ["news-blackouts"],
    queryFn: () => api.get<NewsBlackoutResponse>("/api/news/blackouts"),
    refetchInterval: 60_000,
    staleTime: 30_000,
  });
}

export function useNewsBlackoutForSymbol(symbol: string) {
  const query = useNewsBlackouts();
  const entry = query.data?.symbols.find((s) => s.symbol === symbol) ?? null;
  return { ...query, entry };
}
