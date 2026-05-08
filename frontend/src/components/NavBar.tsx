import { Link, useLocation } from "react-router-dom";
import {
  BarChart3,
  LineChart,
  Briefcase,
  History as HistoryIcon,
  Monitor,
} from "lucide-react";

const PRIMARY_TABS = [
  { path: "/ui", label: "Home", Icon: BarChart3, exact: true },
  { path: "/ui/signals", label: "Signals", Icon: LineChart },
  { path: "/ui/positions", label: "Positions", Icon: Briefcase },
  { path: "/ui/history", label: "History", Icon: HistoryIcon },
  { path: "/ui/system", label: "System", Icon: Monitor },
] as const;

export function NavBar() {
  const location = useLocation();

  function isActive(path: string, exact?: boolean): boolean {
    if (exact) return location.pathname === path || location.pathname === `${path}/`;
    return location.pathname.startsWith(path);
  }

  return (
    <nav
      aria-label="Primary"
      className="glass fixed bottom-0 inset-x-0 z-20 border-t border-[var(--color-border)] md:hidden"
      style={{ paddingBottom: "env(safe-area-inset-bottom)" }}
    >
      <ul className="grid grid-cols-5">
        {PRIMARY_TABS.map(({ path, label, Icon, exact }) => {
          const active = isActive(path, exact);
          return (
            <li key={path}>
              <Link
                to={path}
                className={`flex flex-col items-center justify-center gap-0.5 py-2 text-[10px] transition-colors ${
                  active
                    ? "text-[var(--color-primary)]"
                    : "text-[var(--color-text-dim)] hover:text-[var(--color-text-muted)]"
                }`}
              >
                <Icon size={20} strokeWidth={active ? 2.2 : 1.8} />
                <span className="font-medium">{label}</span>
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
