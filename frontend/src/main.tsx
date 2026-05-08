import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App";
import { applyTheme, readStoredTheme } from "@/hooks/useTheme";
import { applyDensity, readStoredDensity } from "@/hooks/useDensity";

applyTheme(readStoredTheme());
applyDensity(readStoredDensity());

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
