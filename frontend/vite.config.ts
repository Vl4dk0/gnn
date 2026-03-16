import { resolve } from "node:path";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:5555",
      "/config.js": "http://127.0.0.1:5555"
    }
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    // SPA build: single entry point
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html")
      }
    }
  }
});
