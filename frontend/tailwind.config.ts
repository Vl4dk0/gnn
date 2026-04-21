import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./index.html",
    "./src/**/*.{ts,tsx}"
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        bg0: "var(--color-bg0)",
        bg1: "var(--color-bg1)",
        bg2: "var(--color-bg2)",
        line: "var(--color-line)",
        line2: "var(--color-line2)",
        textMain: "var(--color-textMain)",
        textMuted: "var(--color-textMuted)",
        textDim: "var(--color-textDim)",
        accent: "var(--color-accent)",
        accentHover: "var(--color-accentHover)"
      },
      boxShadow: {
        card: "var(--shadow-card)",
        cardHover: "var(--shadow-cardHover)"
      }
    }
  },
  plugins: []
};

export default config;
