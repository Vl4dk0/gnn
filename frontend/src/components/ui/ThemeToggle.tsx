import { useLocation } from "react-router-dom";

import { useTheme } from "../../hooks/useTheme";

const EDITOR_ROUTES = ["/degree", "/min_cycle", "/cage"];

export const ThemeToggle = () => {
  const { theme, toggle } = useTheme();
  const { pathname } = useLocation();

  const isEditor = EDITOR_ROUTES.some((r) => pathname === r || pathname.startsWith(r + "/"));
  if (isEditor) return null;

  const isDark = theme === "dark";

  return (
    <button
      type="button"
      onClick={toggle}
      className="ui-action fixed bottom-6 left-6 z-50 flex h-12 w-12 items-center justify-center rounded-full border-2 border-line2 bg-bg1 text-textDim shadow-card backdrop-blur-sm transition-all duration-300 hover:border-textDim hover:text-textMain hover:shadow-cardHover"
      aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
      title={isDark ? "Switch to light mode" : "Switch to dark mode"}
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="22"
        height="22"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="transition-transform duration-300"
        style={{ transform: isDark ? "rotate(0deg)" : "rotate(360deg)" }}
      >
        {isDark ? (
          <>
            <circle cx="12" cy="12" r="5" />
            <line x1="12" y1="1" x2="12" y2="3" />
            <line x1="12" y1="21" x2="12" y2="23" />
            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
            <line x1="1" y1="12" x2="3" y2="12" />
            <line x1="21" y1="12" x2="23" y2="12" />
            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
          </>
        ) : (
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        )}
      </svg>
    </button>
  );
};
