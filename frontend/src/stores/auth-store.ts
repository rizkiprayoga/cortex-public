import { create } from "zustand";
import { api } from "@/lib/api-client";
import type { TokenResponse } from "@/lib/types";

interface AuthState {
  isAuthenticated: boolean;
  username: string | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  checkAuth: () => Promise<void>;
}

export const useAuthStore = create<AuthState>((set) => ({
  isAuthenticated: api.hasToken(),
  username: null,

  login: async (username: string, password: string) => {
    const data = await api.post<TokenResponse>("/api/auth/login", {
      username,
      password,
    });
    api.setToken(data.access_token);
    set({ isAuthenticated: true, username });
  },

  logout: () => {
    api.clearToken();
    set({ isAuthenticated: false, username: null });
  },

  checkAuth: async () => {
    if (!api.hasToken()) {
      set({ isAuthenticated: false, username: null });
      return;
    }
    try {
      const data = await api.get<{ username: string }>("/api/auth/me");
      set({ isAuthenticated: true, username: data.username });
    } catch {
      api.clearToken();
      set({ isAuthenticated: false, username: null });
    }
  },
}));
