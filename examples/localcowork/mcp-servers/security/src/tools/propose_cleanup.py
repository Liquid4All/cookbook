"""
security.propose_cleanup — Generate cleanup proposals for security findings.

Takes a list of Finding objects and generates ProposedAction items
with appropriate severity levels and action types.
Non-destructive: no confirmation required.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from mcp_base import MCPResult, MCPTool

from patterns import Finding, ProposedAction

# ─── Severity and Action Mappings ──────────────────────────────────────────

FINDING_SEVERITY: dict[str, str] = {
    "ssn": "high",
    "credit_card": "high",
    "aws_key": "high",
    "github_token": "high",
    "stripe_key": "high",
    "private_key": "high",
    "generic_api_key": "medium",
    "email": "low",
    "phone": "low",
}

FINDING_ACTION: dict[str, str] = {
    "ssn": "redact",
    "credit_card": "redact",
    "email": "redact",
    "phone": "redact",
    "aws_key": "rotate",
    "github_token": "rotate",
    "stripe_key": "rotate",
    "private_key": "move",
    "generic_api_key": "rotate",
}

FINDING_DESCRIPTION: dict[str, str] = {
    "ssn": "Redact Social Security Number from file",
    "credit_card": "Redact credit card number from file",
    "email": "Redact email address from file",
    "phone": "Redact phone number from file",
    "aws_key": "Rotate exposed AWS access key and remove from file",
    "github_token": "Rotate exposed GitHub token and remove from file",
    "stripe_key": "Rotate exposed Stripe key and remove from file",
    "private_key": "Move private key to a secure location with restricted permissions",
    "generic_api_key": "Rotate exposed secret/password/token and remove from file",
}


# ─── Params / Result Models ────────────────────────────────────────────────


class Params(BaseModel):
    """Parameters for security.propose_cleanup."""

    findings: list[Finding] = Field(description="Security findings to generate proposals for")


class Result(BaseModel):
    """Return value for security.propose_cleanup."""

    actions: list[ProposedAction]


# ─── Tool Implementation ───────────────────────────────────────────────────


class ProposeCleanup(MCPTool[Params, Result]):
    """Generate cleanup proposals for security findings."""

    name = "security.propose_cleanup"
    description = "Generate cleanup proposals for PII and secret findings"
    confirmation_required = False
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[Result]:
        """Generate proposed actions for the given findings."""
        actions: list[ProposedAction] = []
        seen: set[tuple[str, str]] = set()

        for finding in params.findings:
            # Deduplicate by (file_path, finding_type) so we propose
            # one action per file per finding type, not per occurrence.
            dedup_key = (finding.file_path, finding.finding_type)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            action = _build_action(finding)
            actions.append(action)

        return MCPResult(success=True, data=Result(actions=actions))


# ─── Helper Functions ──────────────────────────────────────────────────────


def _build_action(finding: Finding) -> ProposedAction:
    """Build a ProposedAction for a single finding."""
    action_type = FINDING_ACTION.get(finding.finding_type, "redact")
    severity = FINDING_SEVERITY.get(finding.finding_type, "medium")
    description = FINDING_DESCRIPTION.get(
        finding.finding_type,
        f"Address {finding.finding_type} finding in file",
    )

    return ProposedAction(
        action_type=action_type,
        target_path=finding.file_path,
        description=f"{description}: {finding.file_path}",
        severity=severity,
    )
