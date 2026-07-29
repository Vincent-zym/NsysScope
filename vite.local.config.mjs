import { defineConfig, transformWithOxc } from "vite";
import { fileURLToPath } from "node:url";

const project = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  root: `${project}/local-ui`,
  publicDir: `${project}/public`,
  base: "/",
  plugins: [{
    name: "nsysscope-jsx",
    async transform(code, id) {
      if (id.endsWith("/app/page.js")) {
        return transformWithOxc(code, id, {
          lang: "jsx",
          jsx: { runtime: "automatic" },
        });
      }
      return null;
    },
  }],
  build: {
    outDir: `${project}/backend/static`,
    emptyOutDir: true,
    sourcemap: false,
  },
});
