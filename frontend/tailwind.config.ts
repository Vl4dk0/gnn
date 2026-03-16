import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./index.html",
    "./src/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        bg0: "#0f0f0f",
        bg1: "#1a1a1a",
        bg2: "#2a2a2a",
        line: "#444444",
        line2: "#555555",
        textMain: "#ffffff",
        textMuted: "#cccccc",
        textDim: "#888888",
        accent: "#666666",
        accentHover: "#777777"
      },
      boxShadow: {
        card: "0 4px 15px rgba(0, 0, 0, 0.3)",
        cardHover: "0 6px 20px rgba(0, 0, 0, 0.4)"
      }
    }
  },
  plugins: []
};

export default config;
