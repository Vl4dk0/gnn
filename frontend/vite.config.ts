import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

const THESIS_PDF_PATH = resolve(__dirname, "../thesis/main.pdf");

function thesisPdfPlugin(): Plugin {
  return {
    name: "thesis-pdf",
    configureServer(server) {
      server.middlewares.use("/thesis.pdf", (_req, res) => {
        try {
          const buf = readFileSync(THESIS_PDF_PATH);
          res.setHeader("Content-Type", "application/pdf");
          res.end(buf);
        } catch (err) {
          res.statusCode = 404;
          res.end(`thesis/main.pdf not found: ${(err as Error).message}`);
        }
      });
    },
    generateBundle() {
      this.emitFile({
        type: "asset",
        fileName: "thesis.pdf",
        source: readFileSync(THESIS_PDF_PATH)
      });
    }
  };
}

export default defineConfig({
  plugins: [react(), thesisPdfPlugin()],
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
