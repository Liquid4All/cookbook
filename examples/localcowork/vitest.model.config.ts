import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['tests/model-behavior/**/*.test.ts'],
    globals: false,
    environment: 'node',
    testTimeout: 30000,
  },
});
