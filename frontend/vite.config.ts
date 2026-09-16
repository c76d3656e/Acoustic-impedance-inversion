import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Root base for Vercel (served at the domain root).
export default defineConfig({
  base: "/",
  plugins: [react()],
  build: {
    target: "es2020",
    chunkSizeWarningLimit: 2000,
  },
});
