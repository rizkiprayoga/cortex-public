import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { Lock } from "lucide-react";
import { useAuthStore } from "@/stores/auth-store";
import { useLockStatus } from "@/hooks/useSystemStatus";
import { Logo } from "@/components/Logo";

export function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const login = useAuthStore((s) => s.login);
  const navigate = useNavigate();
  const { data: lockStatus } = useLockStatus();
  const isLocked = lockStatus?.locked ?? false;

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await login(username, password);
      navigate("/ui");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--color-bg)] px-4 relative overflow-hidden">
      {/* Subtle radial accent behind the card */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse at top, rgba(6,182,212,0.12), transparent 55%)",
        }}
      />

      <div className="w-full max-w-sm relative">
        <div className="card p-8 shadow-2xl">
          <div className="text-center mb-8">
            <div className="inline-flex h-12 w-12 items-center justify-center rounded-xl bg-[var(--color-panel-hi)] border border-[var(--color-border)] mb-3">
              <Logo size={28} />
            </div>
            <h1 className="text-brand-gradient text-xl font-semibold tracking-tight">
              Cortex Trading Bot
            </h1>
            <p className="text-xs text-[var(--color-text-muted)] mt-1">
              Sign in to your dashboard
            </p>
          </div>

          {isLocked ? (
            <div className="text-center py-6">
              <div className="inline-flex h-12 w-12 items-center justify-center rounded-full bg-[var(--color-warn)]/15 text-[var(--color-warn)] mb-4">
                <Lock size={20} />
              </div>
              <p className="text-sm text-[var(--color-text)] font-medium">
                Dashboard locked
              </p>
              <p className="text-xs text-[var(--color-text-muted)] mt-2">
                Unlock from the local machine:
              </p>
              <p className="num text-[11px] text-[var(--color-text-dim)] mt-2 font-mono break-all">
                POST http://127.0.0.1:8787/api/system/unlock
              </p>
              <p className="text-[11px] text-[var(--color-text-dim)] mt-3">
                (or run scripts\unlock_dashboard.ps1)
              </p>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-xs text-[var(--color-text-muted)] mb-1.5">
                  Username
                </label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="w-full px-3 py-2.5 rounded-lg bg-[var(--color-panel-hi)] border border-[var(--color-border)] text-[var(--color-text)] text-sm focus:outline-none focus:border-[var(--color-primary)] transition-colors"
                  autoFocus
                  autoComplete="username"
                />
              </div>

              <div>
                <label className="block text-xs text-[var(--color-text-muted)] mb-1.5">
                  Password
                </label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-3 py-2.5 rounded-lg bg-[var(--color-panel-hi)] border border-[var(--color-border)] text-[var(--color-text)] text-sm focus:outline-none focus:border-[var(--color-primary)] transition-colors"
                  autoComplete="current-password"
                />
              </div>

              {error && (
                <p className="text-[var(--color-loss)] text-xs text-center">{error}</p>
              )}

              <button
                type="submit"
                disabled={loading || !username || !password}
                className="w-full py-2.5 rounded-lg bg-[var(--color-primary)] hover:brightness-110 text-[var(--color-bg)] font-semibold text-sm transition-all disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {loading ? "Signing in..." : "Sign in"}
              </button>

              <p className="text-[11px] text-[var(--color-text-dim)] text-center pt-2">
                Session lasts 12 hours
              </p>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}
