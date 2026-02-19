/**
 * Multi-Step Chain Tests — Main export file.
 *
 * Re-exports all 50 multi-step chain test definitions and provides
 * a single combined array.
 */

import type { MultiStepTest } from './types';
import { simpleChainTestsA } from './multi-step-chains-simple-a';
import { simpleChainTestsB } from './multi-step-chains-simple-b';
import { mediumChainTestsA } from './multi-step-chains-medium-a';
import { mediumChainTestsB } from './multi-step-chains-medium-b';
import { complexChainTestsA } from './multi-step-chains-complex-a';
import { complexChainTestsB } from './multi-step-chains-complex-b';
import { complexChainTestsC } from './multi-step-chains-complex-c';

/** All simple chain tests combined (15 total). */
export const simpleChainTests: readonly MultiStepTest[] = [
  ...simpleChainTestsA,   //  8
  ...simpleChainTestsB,   //  7
];

/** All medium chain tests combined (20 total). */
export const mediumChainTests: readonly MultiStepTest[] = [
  ...mediumChainTestsA,   // 10
  ...mediumChainTestsB,   // 10
];

/** All complex chain tests combined (15 total). */
export const complexChainTests: readonly MultiStepTest[] = [
  ...complexChainTestsA,  //  5
  ...complexChainTestsB,  //  5
  ...complexChainTestsC,  //  5
];

/** All 50 multi-step chain test definitions. */
export const allMultiStepTests: readonly MultiStepTest[] = [
  ...simpleChainTests,    // 15
  ...mediumChainTests,    // 20
  ...complexChainTests,   // 15
];                        // Total: 50
