# Demo Flow

Use this script to show Telco Triage as a real on-device support assistant,
not a canned frontend.

## 20-Second Intro

> Imagine if a home internet customer could talk to a support assistant inside
> the carrier app and get help without sending every request to the cloud. Telco
> Triage shows that architecture: Liquid models run locally, understand the
> customer request, choose the right support path, and use cloud assist only
> when live systems are actually needed.

## Recording Flow

### 1. Start On The App Home

Show the customer support surface and the on-device status.

Narration:

> The important claim is simple: the first layer of support intelligence is
> inside the app. The model understands the request locally, then a deterministic
> router decides whether to answer, propose a tool, escalate, or ask for cloud
> assist.

### 2. Local KB Answer

Prompt:

```text
What do the lights on my router mean?
```

Narration:

> This is a common support question. The model routes it to local knowledge
> instead of a cloud chatbot, so the answer is fast and grounded in the bundled
> carrier support corpus.

Open the trace card if engineering mode is enabled.

### 3. Local Tool Proposal

Prompt:

```text
Restart my router
```

Narration:

> Now the assistant has moved from Q&A into an agentic workflow. It identifies
> the tool path, shows confidence, and asks for confirmation before executing.
> The model proposes; policy governs.

### 4. Tool Arguments From Natural Language

Prompt:

```text
Block my son's tablet from the internet
```

Narration:

> This shows natural-language argument extraction. The request is not just
> classified as parental controls; the assistant also extracts the target device
> and prepares the right local action.

### 5. Cloud Assist By Design

Prompt:

```text
Is there an outage in my area?
```

Narration:

> Some requests should not stay local. Outage status requires fresh network
> systems, so the app routes to cloud assist with a structured, redacted payload.
> Cloud is not a failure mode; it is the right lane for live backend context.

### 6. Human Escalation

Prompt:

```text
I want to talk to a person
```

Narration:

> The router can also decide that automation should stop. That matters for
> frustrated customers, policy-sensitive cases, and workflows that need identity
> proofing.

### 7. Privacy Boundary

Prompt:

```text
My account number is 9876543210, why is my bill high?
```

Narration:

> Sensitive data is detected before egress. The customer gets transparency, and
> the carrier can keep private support context local unless cloud assist is
> explicitly needed.

## Closing

> The speed is not a gimmick. It is evidence of the architecture: a small Liquid
> model running locally, emitting typed support decisions, with deterministic
> policy around tools, privacy, and cloud handoff.

## What To Emphasize

- Local model inference happens before cloud.
- The trace card makes model route, confidence, latency, and tool choice visible.
- Local tools are proposed with confirmation.
- Cloud assist is reserved for live systems such as outage, billing, account,
  appointment, and device inventory.
- The app is carrier-agnostic: swap KB, branding, tools, and classifier labels.
