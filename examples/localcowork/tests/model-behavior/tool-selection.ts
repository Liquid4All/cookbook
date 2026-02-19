/**
 * Tool Selection Tests — Main export file.
 *
 * Re-exports all 100 tool-selection test definitions and provides
 * a single combined array.
 */

import type { ToolSelectionTest } from './types';
import { fileOperationTests } from './tool-selection-part1';
import { documentProcessingTests, dataOperationTests } from './tool-selection-part1b';
import {
  ocrVisionTests,
  securityPrivacyTests,
  taskManagementTests,
  calendarTests,
} from './tool-selection-part2';
import {
  emailTests,
  meetingAudioTests,
  knowledgeSearchTests,
  systemClipboardTests,
  auditTests,
} from './tool-selection-part3';

/** All 100 tool-selection test definitions. */
export const allToolSelectionTests: readonly ToolSelectionTest[] = [
  ...fileOperationTests,        // 15
  ...documentProcessingTests,   // 12
  ...dataOperationTests,        // 10
  ...ocrVisionTests,            //  8
  ...securityPrivacyTests,      // 10
  ...taskManagementTests,       //  8
  ...calendarTests,             //  7
  ...emailTests,                //  8
  ...meetingAudioTests,         //  7
  ...knowledgeSearchTests,      //  7
  ...systemClipboardTests,      //  5
  ...auditTests,                //  3
];                              // Total: 100

export {
  fileOperationTests,
  documentProcessingTests,
  dataOperationTests,
  ocrVisionTests,
  securityPrivacyTests,
  taskManagementTests,
  calendarTests,
  emailTests,
  meetingAudioTests,
  knowledgeSearchTests,
  systemClipboardTests,
  auditTests,
};
