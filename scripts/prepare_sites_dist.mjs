import { access, cp, mkdir, writeFile } from "node:fs/promises";

await mkdir("dist/.openai", { recursive: true });
await cp(".openai/hosting.json", "dist/.openai/hosting.json");
try {
  await access("dist/server/index.js");
} catch {
  await writeFile(
    "dist/server/index.js",
    'export { default } from "./index.mjs";\nexport * from "./index.mjs";\n',
  );
}
