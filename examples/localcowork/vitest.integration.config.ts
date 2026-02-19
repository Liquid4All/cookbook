import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["tests/integration/**/*.test.ts"],
    globals: false,
    testTimeout: 30000,
    hookTimeout: 15000,
  },
});