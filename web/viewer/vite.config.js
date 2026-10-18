import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  server: { port: 5173, strictPort: true },
  preview: { port: 8088, strictPort: true },
  build: { target: "es2022", outDir: "dist", emptyOutDir: true },
});
