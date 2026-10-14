import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `pnpm dev` proxies the bridge so the UI code always talks to its own origin.
const bridge = "http://localhost:8765";
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/events": bridge, "/cmd": bridge, "/scorecard.json": bridge, "/video.mjpg": bridge, "/frame.jpg": bridge } },
});
