from dataclasses import dataclass
from enum import Enum
from typing import Any


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(frozen=True)
class PolicyResult:
    decision: PolicyDecision
    reason: str


class PolicyEngine:

    def evaluate(self, action: str, context: Any) -> PolicyResult:

        if action == "read_repository":
            return PolicyResult(
                decision=PolicyDecision.ALLOW,
                reason="Reading repository information is allowed."
            )

        if action == "merge_pull_request":
            return PolicyResult(
                decision=PolicyDecision.REQUIRE_APPROVAL,
                reason="Merging a pull request requires human approval."
            )

        if action == "delete_repository":
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason="Repository deletion is not allowed."
            )

        return PolicyResult(
            decision=PolicyDecision.DENY,
            reason="Action is not recognized by the policy engine."
        )