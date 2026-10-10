import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Port 1420 is fixed because the sidecar's CORS allow-list names it. Changing
// it here means changing ALLOWED_ORIGINS in src/meeting_notes/server.py too.
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
  },
  build: {
    target: "es2020",
    sourcemap: true,
  },
});
