import { useEffect, useRef, useState } from "react";
import { Link, Outlet, useLocation } from "react-router-dom";
import { isDev } from "@/lib/env";
import {
  BarChart3,
  LineChart,
  Briefcase,
  History as HistoryIcon,
  FlaskConical,
  Settings,
  Monitor,
  Brain,
  ScrollText,
  LogOut,
  Lock,
  MoreHorizontal,
} from "lucide-react";
import { useAuthStore } from "@/stores/auth-store";
import { useBotStatus } from "@/hooks/useSystemStatus";
import { useCurrentAccount } from "@/hooks/useAccount";
import { StatusBadge } from "./StatusBadge";
import { AccountSwitcher } from "./AccountSwitcher";
import { NavBar } from "./NavBar";
import { Logo } from "./Logo";
import { api } from "@/lib/api-client";

type NavItem = {
  path: string;
  label: string;
  Icon: typeof BarChart3;
  exact?: boolean;
};

const PRIMARY_NAV: readonly NavItem[] = [
  { path: "/ui", label: "Overview", Icon: BarChart3, exact: true },
  { path: "/ui/signals", label: "Signals", Icon: LineChart },
  { path: "/ui/positions", label: "Positions", Icon: Briefcase },
  { path: "/ui/history", label: "History", Icon: HistoryIcon },
  { path: "/ui/backtest", label: "Backtest", Icon: FlaskConical },
  { path: "/ui/models", label: "Models", Icon: Brain },
] as const;

const TOOLS_NAV: readonly NavItem[] = [
  { path: "/ui/signals-log", label: "Signals Log", Icon: ScrollText },
  { path: "/ui/config", label: "Config", Icon: Settings },
  { path: "/ui/system", label: "System", Icon: Monitor },
] as const;

// Items hidden from the mobile bottom NavBar — surfaced via a "More" menu.
const MORE_ITEMS: readonly NavItem[] = [
  { path: "/ui/backtest", label: "Backtest", Icon: FlaskConical },
  { path: "/ui/models", label: "Models", Icon: Brain },
  { path: "/ui/signals-log", label: "Signals Log", Icon: ScrollText },
  { path: "/ui/config", label: "Config", Icon: Settings },
] as const;

function isActive(pathname: string, path: string, exact?: boolean): boolean {
  if (exact) return pathname === path || pathname === `${path}/`;
  return pathname.startsWith(path);
}

const STATUS_BLOCK: Record<string, { border: string; bg: string; text: string; label: string }> = {
  running: {
    border: "rgba(16,185,129,0.24)",
    bg: "rgba(16,185,129,0.06)",
    text: "var(--chip-profit-fg)",
    label: "BOT RUNNING",
  },
  paused: {
    border: "rgba(245,158,11,0.28)",
    bg: "rgba(245,158,11,0.06)",
    text: "var(--chip-warn-fg)",
    label: "BOT PAUSED",
  },
  stopped: {
    border: "rgba(244,63,94,0.28)",
    bg: "rgba(244,63,94,0.06)",
    text: "var(--chip-loss-fg)",
    label: "BOT STOPPED",
  },
};

export function Layout() {
  const location = useLocation();
  const logout = useAuthStore((s) => s.logout);
  const username = useAuthStore((s) => s.username);
  const { data: botStatus } = useBotStatus();
  const { data: account } = useCurrentAccount();
  const [moreOpen, setMoreOpen] = useState(false);
  const moreRef = useRef<HTMLDivElement | null>(null);
  const dev = isDev();

  // Prefix browser tab title with [DEV] so the user can tell tabs apart.
  useEffect(() => {
    if (!dev) return;
    const prev = document.title;
    if (!prev.startsWith("[DEV]")) {
      document.title = `[DEV] ${prev}`;
    }
  }, [dev]);

  // Close the "More" dropdown on outside click or route change
  useEffect(() => {
    if (!moreOpen) return;
    const onDocClick = (e: MouseEvent) => {
      if (moreRef.current && !moreRef.current.contains(e.target as Node)) {
        setMoreOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [moreOpen]);

  useEffect(() => {
    setMoreOpen(false);
  }, [location.pathname]);

  const handleLock = async () => {
    try {
      await api.post("/api/system/lock");
    } catch {
      // ignore
    }
    logout();
    window.location.href = "/ui/login";
  };

  const handleSignOut = () => {
    logout();
    window.location.href = "/ui/login";
  };

  const status = (botStatus?.status ?? "stopped").toLowerCase();
  const statusStyle = STATUS_BLOCK[status] ?? STATUS_BLOCK.stopped;

  const renderNavItem = ({ path, label, Icon, exact }: NavItem) => {
    const active = isActive(location.pathname, path, exact);
    return (
      <Link
        key={path}
        to={path}
        title={label}
        className={`flex items-center gap-3 px-3 py-2.5 mb-0.5 rounded-lg text-sm transition-colors justify-center lg:justify-start ${
          active
            ? "font-medium"
            : "text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-panel-hi)]"
        }`}
        style={
          active
            ? {
                background: "rgba(99,102,241,0.12)",
                color: "var(--chip-info-fg)",
              }
            : undefined
        }
      >
        <Icon size={16} strokeWidth={active ? 2.2 : 1.8} />
        <span className="hidden lg:inline">{label}</span>
      </Link>
    );
  };

  return (
    <div className="flex h-screen overflow-hidden bg-[var(--color-bg)]">
      {/* Desktop sidebar — icon-only at md (768-1023), full at lg+ */}
      <aside className="hidden md:flex md:w-16 lg:w-60 shrink-0 flex-col border-r border-[var(--color-border)] bg-[var(--color-panel)]">
        <div className="h-16 flex items-center gap-2.5 justify-center lg:justify-start px-0 lg:px-5 border-b border-[var(--color-border)]">
          <Logo size={28} />
          <span className="hidden lg:inline text-brand-gradient text-lg font-semibold tracking-tight">
            Cortex
          </span>
          {dev && (
            <span
              title="Development environment (port 8788, trading_bot_dev)"
              className="hidden lg:inline mono text-[9px] font-bold tracking-wider px-1.5 py-0.5 rounded"
              style={{
                border: "1px solid rgba(245,158,11,0.45)",
                background: "rgba(245,158,11,0.12)",
                color: "var(--chip-warn-fg)",
              }}
            >
              DEV
            </span>
          )}
        </div>

        <div className="hidden lg:block">
          <AccountSwitcher />
        </div>

        <nav className="px-2 flex-1 overflow-y-auto py-2">
          {PRIMARY_NAV.map(renderNavItem)}
          <p className="hidden lg:block mt-5 mb-2 px-3 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-dim)]">
            Tools
          </p>
          {TOOLS_NAV.map(renderNavItem)}
        </nav>

        {/* Bot status block — full card at lg, status dot only at md */}
        <div
          className="hidden lg:block m-3 p-3 rounded-xl border"
          style={{ borderColor: statusStyle.border, background: statusStyle.bg }}
        >
          <div className="flex items-center gap-2">
            <span className="live-pulse w-2 h-2 rounded-full" style={{ background: statusStyle.text }} />
            <span className="text-xs font-semibold" style={{ color: statusStyle.text }}>
              {statusStyle.label}
            </span>
          </div>
          {botStatus?.last_change && (
            <p className="mono mt-1.5 text-[10px] text-[var(--color-text-muted)]">
              since {new Date(botStatus.last_change).toLocaleTimeString()}
            </p>
          )}
        </div>
        <div
          className="lg:hidden mx-auto my-3 w-2 h-2 rounded-full live-pulse"
          title={statusStyle.label}
          style={{ background: statusStyle.text }}
        />

        <div className="border-t border-[var(--color-border)] p-3 space-y-1">
          <button
            onClick={handleLock}
            title="Lock dashboard"
            className="w-full flex items-center gap-2 justify-center lg:justify-start px-3 py-2 text-xs text-[var(--color-text-muted)] hover:text-[var(--color-loss)] rounded transition-colors"
          >
            <Lock size={14} /> <span className="hidden lg:inline">Lock dashboard</span>
          </button>
          <button
            onClick={handleSignOut}
            title="Sign out"
            className="w-full flex items-center gap-2 justify-center lg:justify-start px-3 py-2 text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text)] rounded transition-colors"
          >
            <LogOut size={14} /> <span className="hidden lg:inline">Sign out {username && `(${username})`}</span>
          </button>
        </div>
      </aside>

      {/* Main column */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top bar — full content on mobile (<md), sidebar owns nav from md+ */}
        <header className="h-14 shrink-0 flex items-center gap-3 px-4 border-b border-[var(--color-border)] bg-[var(--color-panel)] md:bg-transparent">
          <span className="md:hidden">
            <Logo variant="full" size={24} />
          </span>
          {dev && (
            <span
              title="Development environment"
              className="mono text-[9px] font-bold tracking-wider px-1.5 py-0.5 rounded md:hidden"
              style={{
                border: "1px solid rgba(245,158,11,0.45)",
                background: "rgba(245,158,11,0.12)",
                color: "var(--chip-warn-fg)",
              }}
            >
              DEV
            </span>
          )}
          {botStatus && <span className="md:hidden"><StatusBadge status={botStatus.status} /></span>}
          {account?.account_id && (
            <span className="md:hidden text-[10px] text-[var(--color-text-muted)] truncate max-w-24">
              #{account.account_id}
            </span>
          )}

          <div className="ml-auto flex items-center gap-2 relative" ref={moreRef}>
            <button
              onClick={() => setMoreOpen((v) => !v)}
              className="md:hidden p-2 rounded-md text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-panel-hi)]"
              aria-label="More"
              aria-expanded={moreOpen}
            >
              <MoreHorizontal size={18} />
            </button>

            {moreOpen && (
              <div className="absolute right-0 top-full mt-1 w-48 rounded-lg border border-[var(--color-border)] bg-[var(--color-panel-hi)] shadow-xl z-30 py-1">
                {MORE_ITEMS.map(({ path, label, Icon }) => (
                  <Link
                    key={path}
                    to={path}
                    className="flex items-center gap-2 px-3 py-2 text-sm text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-panel)]"
                  >
                    <Icon size={14} /> {label}
                  </Link>
                ))}
                <div className="my-1 border-t border-[var(--color-border)]" />
                <button
                  onClick={handleLock}
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm text-[var(--color-text-muted)] hover:text-[var(--color-loss)] hover:bg-[var(--color-panel)]"
                >
                  <Lock size={14} /> Lock
                </button>
                <button
                  onClick={handleSignOut}
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-panel)]"
                >
                  <LogOut size={14} /> Sign out
                </button>
              </div>
            )}
          </div>
        </header>

        <main className="flex-1 overflow-y-auto p-4 lg:p-6 pb-20 md:pb-6">
          <Outlet />
        </main>

        {/* Mobile-only bottom tab bar (hidden at md+ where sidebar handles nav) */}
        <NavBar />
      </div>
    </div>
  );
}
