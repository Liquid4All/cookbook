/**
 * Model Behavior Test Runner — vitest-compatible test file.
 *
 * Structural validation mode (default): validates all test definitions
 * are well-formed, IDs are unique, and all referenced tools are valid.
 *
 * Live model mode (LOCALCOWORK_MODEL_ENDPOINT set): sends prompts to
 * the model endpoint and checks tool-calling accuracy.
 */

import { describe, it, expect } from 'vitest';
import {
  validateToolSelectionTest,
  validateMultiStepTest,
  validateEdgeCaseTest,
  checkUniqueIds,
} from './framework';
import { allToolSelectionTests } from './tool-selection';
import { allMultiStepTests } from './multi-step-chains';
import { allEdgeCaseTests } from './edge-cases';
import { VALID_TOOL_SET } from './types';

// ---------------------------------------------------------------------------
// Structural Validation Tests (always run, no model required)
// ---------------------------------------------------------------------------

describe('Model Behavior Test Suite — Structural Validation', () => {
  describe('Test Definition Counts', () => {
    it('should have exactly 100 tool-selection tests', () => {
      expect(allToolSelectionTests).toHaveLength(100);
    });

    it('should have exactly 50 multi-step chain tests', () => {
      expect(allMultiStepTests).toHaveLength(50);
    });

    it('should have exactly 30 edge-case tests', () => {
      expect(allEdgeCaseTests).toHaveLength(30);
    });

    it('should have 180 total tests across all suites', () => {
      const total =
        allToolSelectionTests.length +
        allMultiStepTests.length +
        allEdgeCaseTests.length;
      expect(total).toBe(180);
    });
  });

  describe('Unique IDs', () => {
    it('should have no duplicate IDs across all test suites', () => {
      const errors = checkUniqueIds(
        allToolSelectionTests,
        allMultiStepTests,
        allEdgeCaseTests,
      );
      if (errors.length > 0) {
        throw new Error(`Duplicate IDs found:\n${errors.join('\n')}`);
      }
      expect(errors).toHaveLength(0);
    });
  });

  describe('Tool Selection Tests — Validation', () => {
    for (const test of allToolSelectionTests) {
      it(`[${test.id}] should be well-formed`, () => {
        const errors = validateToolSelectionTest(test);
        if (errors.length > 0) {
          throw new Error(`Validation errors:\n${errors.join('\n')}`);
        }
        expect(errors).toHaveLength(0);
      });
    }
  });

  describe('Multi-Step Chain Tests — Validation', () => {
    for (const test of allMultiStepTests) {
      it(`[${test.id}] should be well-formed`, () => {
        const errors = validateMultiStepTest(test);
        if (errors.length > 0) {
          throw new Error(`Validation errors:\n${errors.join('\n')}`);
        }
        expect(errors).toHaveLength(0);
      });
    }
  });

  describe('Edge Case Tests — Validation', () => {
    for (const test of allEdgeCaseTests) {
      it(`[${test.id}] should be well-formed`, () => {
        const errors = validateEdgeCaseTest(test);
        if (errors.length > 0) {
          throw new Error(`Validation errors:\n${errors.join('\n')}`);
        }
        expect(errors).toHaveLength(0);
      });
    }
  });

  describe('Tool Name Coverage', () => {
    it('should reference tools from all 13 MCP servers', () => {
      const serversReferenced = new Set<string>();

      for (const test of allToolSelectionTests) {
        for (const tool of test.expectedTools) {
          const server = tool.split('.')[0];
          serversReferenced.add(server);
        }
      }

      const expectedServers = [
        'filesystem', 'document', 'ocr', 'data', 'audit',
        'knowledge', 'security', 'task', 'calendar', 'email',
        'meeting', 'clipboard', 'system',
      ];

      for (const server of expectedServers) {
        expect(
          serversReferenced.has(server),
          `Missing tests for server: ${server}`,
        ).toBe(true);
      }
    });

    it('should only reference valid tool names', () => {
      const invalidTools: string[] = [];

      for (const test of allToolSelectionTests) {
        for (const tool of test.expectedTools) {
          if (!VALID_TOOL_SET.has(tool)) {
            invalidTools.push(`[${test.id}] ${tool}`);
          }
        }
      }

      for (const test of allMultiStepTests) {
        for (const step of test.steps) {
          for (const tool of step.expectedTools) {
            if (!VALID_TOOL_SET.has(tool)) {
              invalidTools.push(`[${test.id}] ${tool}`);
            }
          }
        }
      }

      if (invalidTools.length > 0) {
        throw new Error(`Invalid tool references:\n${invalidTools.join('\n')}`);
      }
      expect(invalidTools).toHaveLength(0);
    });
  });

  describe('Difficulty Distribution', () => {
    it('should have a reasonable difficulty distribution for tool-selection tests', () => {
      const counts = { easy: 0, medium: 0, hard: 0 };
      for (const test of allToolSelectionTests) {
        counts[test.difficulty]++;
      }
      // At least 30% easy, at least 10% medium
      expect(counts.easy).toBeGreaterThanOrEqual(30);
      expect(counts.medium).toBeGreaterThanOrEqual(10);
    });

    it('should have a reasonable difficulty distribution for multi-step tests', () => {
      const counts = { easy: 0, medium: 0, hard: 0 };
      for (const test of allMultiStepTests) {
        counts[test.difficulty]++;
      }
      expect(counts.easy).toBeGreaterThanOrEqual(10);
      expect(counts.medium).toBeGreaterThanOrEqual(10);
      expect(counts.hard).toBeGreaterThanOrEqual(10);
    });
  });

  describe('Category Coverage', () => {
    it('should cover all expected tool-selection categories', () => {
      const categories = new Set(allToolSelectionTests.map((t) => t.category));
      const expected = [
        'file-operations',
        'document-processing',
        'data-operations',
        'ocr-vision',
        'security-privacy',
        'task-management',
        'calendar',
        'email',
        'meeting-audio',
        'knowledge-search',
        'system-clipboard',
        'audit',
      ];
      for (const cat of expected) {
        expect(categories.has(cat), `Missing category: ${cat}`).toBe(true);
      }
    });

    it('should cover all expected edge-case categories', () => {
      const categories = new Set(allEdgeCaseTests.map((t) => t.category));
      const expected = ['ambiguous', 'error-condition', 'malformed', 'boundary'];
      for (const cat of expected) {
        expect(categories.has(cat), `Missing category: ${cat}`).toBe(true);
      }
    });

    it('should cover all expected multi-step chain categories', () => {
      const categories = new Set(allMultiStepTests.map((t) => t.category));
      const expected = ['simple-chain', 'medium-chain', 'complex-chain'];
      for (const cat of expected) {
        expect(categories.has(cat), `Missing category: ${cat}`).toBe(true);
      }
    });
  });
});
