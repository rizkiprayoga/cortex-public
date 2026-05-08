import { lazy, Suspense } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Layout } from "@/components/Layout";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { Login } from "@/screens/Login";
import { Overview } from "@/screens/Overview";
import { Positions } from "@/screens/Positions";
import { SignalsIndex, SignalPage } from "@/screens/Signals";
import { History } from "@/screens/History";
import { Config } from "@/screens/Config";
import { SignalsLog } from "@/screens/SignalsLog";
import { System } from "@/screens/System";

// Code-split: heavy secondary screens are loaded on demand (P-1b).
// Backtest + Models pull in recharts / drawer logic; deferring them keeps
// the first-paint bundle lean for the Overview + Signals + Positions
// critical path. Screens use named exports, so we unwrap via .then().
const Backtest = lazy(() =>
  import("@/screens/Backtest").then((m) => ({ default: m.Backtest })),
);
const Models = lazy(() =>
  import("@/screens/Models").then((m) => ({ default: m.Models })),
);

function RouteLoading() {
  return (
    <div className="flex items-center justify-center h-64 text-xs text-[var(--color-text-dim)]">
      Loading…
    </div>
  );
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 10_000,
      refetchOnWindowFocus: false,
      // Keep polling even when the browser tab is in background. Without
      // this, refetchInterval pauses the moment the tab loses focus and
      // the user sees stale data on return.
      refetchIntervalInBackground: true,
      // Refetch when the network reconnects after a drop.
      refetchOnReconnect: true,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/ui/login" element={<Login />} />

          <Route
            path="/ui"
            element={
              <ProtectedRoute>
                <Layout />
              </ProtectedRoute>
            }
          >
            <Route index element={<Overview />} />
            <Route path="positions" element={<Positions />} />
            <Route path="signals" element={<SignalsIndex />} />
            <Route path="signals/:symbol" element={<SignalPage />} />
            <Route path="history" element={<History />} />
            <Route
              path="backtest"
              element={
                <Suspense fallback={<RouteLoading />}>
                  <Backtest />
                </Suspense>
              }
            />
            <Route
              path="models"
              element={
                <Suspense fallback={<RouteLoading />}>
                  <Models />
                </Suspense>
              }
            />
            <Route path="signals-log" element={<SignalsLog />} />
            <Route path="config" element={<Config />} />
            <Route path="system" element={<System />} />
          </Route>

          <Route path="*" element={<Navigate to="/ui" replace />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
