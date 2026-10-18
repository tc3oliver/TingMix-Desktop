import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";

// Serves the real app from src/ with the repo as root; only the entry (harness.html) and the mocked host differ.
export default defineConfig({
  root: fileURLToPath(new URL("../..", import.meta.url)),
  plugins: [react()],
  clearScreen: false,
  server: { port: 1431, strictPort: true },
});
