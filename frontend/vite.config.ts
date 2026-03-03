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
    rollupOptions: {
      input: {
        overview: resolve(__dirname, "index.html"),
        degree: resolve(__dirname, "degree/index.html"),
        minCycle: resolve(__dirname, "min_cycle/index.html"),
        cage: resolve(__dirname, "cage/index.html"),
        docsGnns: resolve(__dirname, "docs/gnns.html"),
        docsArchitecture: resolve(__dirname, "docs/architecture.html"),
        docsDegree: resolve(__dirname, "docs/module-degree.html"),
        docsMinCycle: resolve(__dirname, "docs/module-min-cycle.html"),
        docsAssessment: resolve(__dirname, "docs/module-assessment.html"),
        docsCage: resolve(__dirname, "docs/module-cage.html"),
        docsTraining: resolve(__dirname, "docs/training.html")
      }
    }
  }
});
