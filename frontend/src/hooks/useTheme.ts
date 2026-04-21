import { useCallback, useEffect, useState } from "react";

type Theme = "light" | "dark";

const STORAGE_KEY = "theme";

const getInitialTheme = (): Theme => {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored === "light" || stored === "dark") return stored;
  return "dark";
};

const applyTheme = (theme: Theme) => {
  if (theme === "dark") {
    document.documentElement.classList.add("dark");
  } else {
    document.documentElement.classList.remove("dark");
  }
};

export const useTheme = () => {
  const [theme, setThemeState] = useState<Theme>(getInitialTheme);

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  const toggle = useCallback(() => {
    setThemeState((prev) => {
      const next = prev === "dark" ? "light" : "dark";
      localStorage.setItem(STORAGE_KEY, next);
      return next;
    });
  }, []);

  return { theme, toggle } as const;
};

/** Theme colors for canvas/SVG rendering that can't use CSS variables */
export const getCanvasTheme = () => {
  const isDark = document.documentElement.classList.contains("dark");
  return isDark
    ? {
        background: "#1a1a1a",
        edge: "#555555",
        text: "#ffffff",
        node: "#666666",
        nodeSelected: "#999999",
        errorDot: "#ff0000",
        correctColor: "#00ff00",
        dimColor: "#888888",
      }
    : {
        background: "#f0f0f0",
        edge: "#bbbbbb",
        text: "#1a1a1a",
        node: "#c0c0c0",
        nodeSelected: "#a0a0a0",
        errorDot: "#dd0000",
        correctColor: "#008800",
        dimColor: "#888888",
      };
};
