import { useEffect, useRef, useState } from "react";
import { ChevronDown, Plus, Check, Loader2 } from "lucide-react";
import {
  useCurrentAccount,
  useAccountSlots,
  useSwitchAccount,
  useRegisterAccount,
} from "@/hooks/useAccount";
import { useBotStatus } from "@/hooks/useSystemStatus";

export function AccountSwitcher() {
  const { data: account } = useCurrentAccount();
  const { data: slotsData } = useAccountSlots();
  const { data: botStatus } = useBotStatus();
  const switchMut = useSwitchAccount();
  const registerMut = useRegisterAccount();

  const [open, setOpen] = useState(false);
  const [showForm, setShowForm] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Form state
  const [slotName, setSlotName] = useState("");
  const [login, setLogin] = useState("");
  const [password, setPassword] = useState("");
  const [server, setServer] = useState("");
  const [autoSwitch, setAutoSwitch] = useState(true);
  const [error, setError] = useState("");

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
        setShowForm(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const isRunning = botStatus?.status === "running";
  const slots = slotsData?.slots ?? [];

  const handleSwitch = (slot: string) => {
    if (isRunning) return;
    switchMut.mutate(
      { slot },
      {
        onSuccess: () => {
          setOpen(false);
        },
        onError: (err) => {
          setError(err.message);
        },
      },
    );
  };

  const handleRegister = (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    const loginNum = parseInt(login, 10);
    if (!slotName || !loginNum || !password || !server) {
      setError("All fields are required.");
      return;
    }
    registerMut.mutate(
      {
        slot_name: slotName,
        login: loginNum,
        password,
        server,
        auto_switch: autoSwitch,
      },
      {
        onSuccess: () => {
          setShowForm(false);
          setOpen(false);
          setSlotName("");
          setLogin("");
          setPassword("");
          setServer("");
          setError("");
        },
        onError: (err) => {
          setError(err.message);
        },
      },
    );
  };

  const isSwitching = switchMut.isPending || registerMut.isPending;

  return (
    <div className="relative" ref={ref}>
      {/* Account chip */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-4 py-2 text-xs border-b border-[var(--color-border)] hover:bg-[var(--color-panel-hi)] transition-colors"
      >
        <div className="flex-1 text-left min-w-0">
          <div className="text-[var(--color-text)] font-medium truncate">
            {account?.account_id ?? "No account"}
          </div>
          <div className="text-[var(--color-text-muted)] truncate text-[10px]">
            {account?.server ?? "Not connected"}
            {account?.is_demo && (
              <span className="ml-1 text-yellow-500">DEMO</span>
            )}
          </div>
        </div>
        {isSwitching ? (
          <Loader2 size={12} className="animate-spin text-[var(--color-text-muted)]" />
        ) : (
          <ChevronDown
            size={12}
            className={`text-[var(--color-text-muted)] transition-transform ${open ? "rotate-180" : ""}`}
          />
        )}
      </button>

      {/* Dropdown */}
      {open && (
        <div className="absolute left-0 right-0 top-full z-30 border border-[var(--color-border)] bg-[var(--color-panel)] shadow-xl rounded-b-lg max-h-80 overflow-y-auto">
          {isRunning && (
            <div className="px-3 py-2 text-[10px] text-yellow-500 bg-yellow-500/10 border-b border-[var(--color-border)]">
              Pause the bot before switching accounts
            </div>
          )}

          {/* Slot list */}
          {slots.map((s) => (
            <button
              key={s.slot}
              onClick={() => !s.is_current && handleSwitch(s.slot)}
              disabled={isRunning || s.is_current}
              className={`w-full flex items-center gap-2 px-3 py-2 text-left text-xs transition-colors ${
                s.is_current
                  ? "bg-[var(--color-panel-hi)] text-[var(--color-primary)]"
                  : isRunning
                    ? "text-[var(--color-text-muted)] opacity-50 cursor-not-allowed"
                    : "text-[var(--color-text-muted)] hover:bg-[var(--color-panel-hi)] hover:text-[var(--color-text)]"
              }`}
            >
              <div className="flex-1 min-w-0">
                <div className="font-medium truncate">{s.slot}</div>
                <div className="text-[10px] truncate opacity-70">
                  {s.login} &middot; {s.server}
                </div>
              </div>
              {s.is_current && (
                <Check size={12} className="shrink-0 text-[var(--color-primary)]" />
              )}
            </button>
          ))}

          {slots.length === 0 && (
            <div className="px-3 py-2 text-[10px] text-[var(--color-text-muted)]">
              No account slots configured
            </div>
          )}

          {/* Add Account button / form */}
          {!showForm ? (
            <button
              onClick={() => setShowForm(true)}
              className="w-full flex items-center gap-2 px-3 py-2 text-xs text-[var(--color-text-muted)] hover:text-[var(--color-primary)] hover:bg-[var(--color-panel-hi)] border-t border-[var(--color-border)] transition-colors"
            >
              <Plus size={12} /> Add Account
            </button>
          ) : (
            <form
              onSubmit={handleRegister}
              className="border-t border-[var(--color-border)] p-3 space-y-2"
            >
              <div className="text-xs font-medium text-[var(--color-text)] mb-1">
                New Account
              </div>
              <input
                type="text"
                placeholder="Slot name (e.g. live2)"
                value={slotName}
                onChange={(e) => setSlotName(e.target.value)}
                className="w-full px-2 py-1.5 text-xs rounded border border-[var(--color-border)] bg-[var(--color-bg)] text-[var(--color-text)] placeholder:text-[var(--color-text-muted)]"
              />
              <input
                type="number"
                placeholder="MT5 Login"
                value={login}
                onChange={(e) => setLogin(e.target.value)}
                className="w-full px-2 py-1.5 text-xs rounded border border-[var(--color-border)] bg-[var(--color-bg)] text-[var(--color-text)] placeholder:text-[var(--color-text-muted)]"
              />
              <input
                type="password"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-2 py-1.5 text-xs rounded border border-[var(--color-border)] bg-[var(--color-bg)] text-[var(--color-text)] placeholder:text-[var(--color-text-muted)]"
              />
              <input
                type="text"
                placeholder="Server (e.g. ICMarkets-Live)"
                value={server}
                onChange={(e) => setServer(e.target.value)}
                className="w-full px-2 py-1.5 text-xs rounded border border-[var(--color-border)] bg-[var(--color-bg)] text-[var(--color-text)] placeholder:text-[var(--color-text-muted)]"
              />
              <label className="flex items-center gap-2 text-[10px] text-[var(--color-text-muted)]">
                <input
                  type="checkbox"
                  checked={autoSwitch}
                  onChange={(e) => setAutoSwitch(e.target.checked)}
                  className="rounded"
                />
                Switch to this account after adding
              </label>

              {error && (
                <div className="text-[10px] text-[var(--color-loss)]">
                  {error}
                </div>
              )}

              <div className="flex gap-2">
                <button
                  type="submit"
                  disabled={registerMut.isPending}
                  className="flex-1 px-2 py-1.5 text-xs rounded bg-[var(--color-primary)] text-white hover:opacity-90 disabled:opacity-50 transition-opacity"
                >
                  {registerMut.isPending ? "Adding..." : "Add"}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setShowForm(false);
                    setError("");
                  }}
                  className="px-2 py-1.5 text-xs rounded border border-[var(--color-border)] text-[var(--color-text-muted)] hover:text-[var(--color-text)] transition-colors"
                >
                  Cancel
                </button>
              </div>
            </form>
          )}
        </div>
      )}
    </div>
  );
}
