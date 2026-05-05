# Testing Telco Triage

The visible app, Xcode project, app target, source module, and test bundle are
all named **Telco Triage** / `TelcoTriage`.

## Fast Test Pass

This pass avoids model loading and is the best first check for cookbook users.

```bash
cd examples/telco-triage-ios
xcodegen generate
xcodebuild test \
  -project TelcoTriage.xcodeproj \
  -scheme TelcoTriage \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -skip-testing:TelcoTriageTests/LFMValidationTests \
  -skip-testing:TelcoTriageTests/LlamaBackendSmokeTests
```

It covers the deterministic router, topic gate, tool registry, PII masker,
conversation starters, support signals, branding, and chat view model paths.

## Full Local Model Pass

Copy the GGUFs first:

```bash
cd examples/telco-triage-ios
TELCO_MODELS_DIR=/path/to/telco-models ./bootstrap-models.sh
xcodegen generate
xcodebuild test \
  -project TelcoTriage.xcodeproj \
  -scheme TelcoTriage \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro'
```

The full pass includes:

- `LFMValidationTests`: verifies the bundled model names and runtime stack.
- `LlamaBackendSmokeTests`: verifies the llama.cpp bridge can load the local
  artifacts.

## Pre-Demo Checklist

1. `./bootstrap-models.sh`
2. `xcodegen generate`
3. Run the fast test pass.
4. Run the app on a physical iPhone if latency matters for the recording.
5. Test these chat paths:
   - local answer: `What do the lights on my router mean?`
   - local tool: `Restart my router`
   - tool arguments: `Block my son's tablet from the internet`
   - cloud assist: `Is there an outage in my area?`
   - human escalation: `I want to talk to a person`
   - PII: `My account number is 9876543210, why is my bill high?`

## Expected Behavior

- Simple support questions stay on device.
- Tool requests show a confirmation surface before execution.
- Cloud-assist requests show why cloud is needed and what context would be
  sent.
- PII is detected and redacted before cloud egress.
- The trace card shows model route, confidence, latency, and selected tool.
