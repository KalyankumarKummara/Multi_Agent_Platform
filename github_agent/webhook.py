"""Secure, framework-independent ingestion for GitHub webhooks.

The webhook edge is separate from GitHubAgent so transport and signature
checks stay outside domain processing. Payloads are verified as their original
bytes before JSON parsing, then reduced to a small allowlisted event model.
"""

from dataclasses import dataclass, field
import hashlib
import hmac
import json
import re
from threading import Lock
from typing import Any, Mapping

from shared.agent_core.errors import AgentError, ErrorCategory


SUPPORTED_EVENTS = frozenset(
    {
        "repository",
        "push",
        "create",
        "pull_request",
        "pull_request_review",
        "pull_request_review_comment",
        "issues",
        "issue_comment",
    }
)
_SIGNATURE_RE = re.compile(r"^sha256=[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class NormalizedGitHubEvent:
    """Small internal event with only selected domain fields."""

    event_id: str
    source: str
    event_type: str
    action: str | None
    repository: dict[str, Any]
    actor: dict[str, Any] | None
    occurred_at: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WebhookResult:
    """Non-sensitive outcome of processing one webhook delivery."""

    status: str
    delivery_id: str
    result: Any = None


class GitHubWebhookSignatureVerifier:
    """Verify GitHub's SHA-256 HMAC using the untouched request body."""

    def verify(self, raw_body: bytes, signature: str | None, secret: str) -> None:
        if not isinstance(secret, str) or not secret.strip():
            raise AgentError(
                "WEBHOOK_SECRET_MISSING",
                "GitHub webhook secret is not configured.",
                ErrorCategory.CONFIGURATION,
                False,
            )
        if not signature:
            raise AgentError(
                "WEBHOOK_SIGNATURE_MISSING",
                "GitHub webhook signature is missing.",
                ErrorCategory.AUTHENTICATION,
                False,
            )
        if not isinstance(signature, str) or not _SIGNATURE_RE.fullmatch(signature):
            raise AgentError(
                "WEBHOOK_SIGNATURE_MALFORMED",
                "GitHub webhook signature has an invalid format.",
                ErrorCategory.AUTHENTICATION,
                False,
            )
        if not isinstance(raw_body, bytes):
            raise AgentError(
                "WEBHOOK_BODY_INVALID",
                "GitHub webhook body must be raw bytes.",
                ErrorCategory.VALIDATION,
                False,
            )

        expected = "sha256=" + hmac.new(
            secret.encode("utf-8"), raw_body, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise AgentError(
                "WEBHOOK_SIGNATURE_INVALID",
                "GitHub webhook signature verification failed.",
                ErrorCategory.AUTHENTICATION,
                False,
            )


class GitHubEventNormalizer:
    """Validate required event structures and retain only useful fields."""

    @staticmethod
    def _invalid(message: str, field_name: str | None = None) -> AgentError:
        details = {"field": field_name} if field_name else None
        return AgentError(
            "WEBHOOK_PAYLOAD_INVALID",
            message,
            ErrorCategory.VALIDATION,
            False,
            details,
        )

    @classmethod
    def _mapping(cls, parent: dict[str, Any], key: str) -> dict[str, Any]:
        value = parent.get(key)
        if not isinstance(value, dict):
            raise cls._invalid(f"Webhook payload field '{key}' must be an object.", key)
        return value

    @classmethod
    def _text(cls, parent: dict[str, Any], key: str) -> str:
        value = parent.get(key)
        if not isinstance(value, str) or not value.strip():
            raise cls._invalid(f"Webhook payload field '{key}' must be a non-empty string.", key)
        return value.strip()

    @classmethod
    def _repository(cls, payload: dict[str, Any]) -> dict[str, Any]:
        source = cls._mapping(payload, "repository")
        name = cls._text(source, "name")
        owner_source = source.get("owner")
        owner = None
        if isinstance(owner_source, dict):
            owner = owner_source.get("login")
        full_name = source.get("full_name")
        if not isinstance(full_name, str) or not full_name.strip():
            full_name = f"{owner}/{name}" if isinstance(owner, str) and owner else name
        repository: dict[str, Any] = {
            "name": name,
            "full_name": full_name,
        }
        if isinstance(source.get("id"), int) and not isinstance(source.get("id"), bool):
            repository["id"] = source["id"]
        if isinstance(owner, str) and owner:
            repository["owner"] = owner
        if isinstance(source.get("html_url"), str):
            repository["url"] = source["html_url"]
        if isinstance(source.get("default_branch"), str):
            repository["default_branch"] = source["default_branch"]
        return repository

    @staticmethod
    def _actor(payload: dict[str, Any], event_type: str) -> dict[str, Any] | None:
        value = payload.get("sender")
        if not isinstance(value, dict) and event_type == "push":
            value = payload.get("pusher")
        if not isinstance(value, dict):
            return None
        actor: dict[str, Any] = {}
        for source_key, target_key in (("login", "login"), ("name", "name"), ("id", "id"), ("type", "type")):
            item = value.get(source_key)
            if isinstance(item, (str, int)) and not isinstance(item, bool):
                actor[target_key] = item
        return actor or None

    @classmethod
    def normalize(
        cls,
        event_id: str,
        event_type: str,
        payload: Any,
    ) -> NormalizedGitHubEvent:
        if event_type not in SUPPORTED_EVENTS:
            raise AgentError(
                "WEBHOOK_EVENT_UNSUPPORTED",
                "GitHub webhook event type is not supported.",
                ErrorCategory.VALIDATION,
                False,
                {"event_type": event_type},
            )
        if not isinstance(payload, dict):
            raise cls._invalid("Webhook JSON payload must be an object.")

        repository = cls._repository(payload)
        action = payload.get("action")
        if action is not None and (not isinstance(action, str) or not action.strip()):
            raise cls._invalid("Webhook payload field 'action' must be a non-empty string when present.", "action")
        if isinstance(action, str):
            action = action.strip()

        metadata: dict[str, Any] = {}
        occurred_at = None

        if event_type == "repository":
            if action is None:
                raise cls._invalid("Repository event requires an action.", "action")
            metadata["repository_action"] = action
            occurred_at = payload.get("repository", {}).get("updated_at")
        elif event_type == "push":
            metadata["ref"] = cls._text(payload, "ref")
            for field_name in ("before", "after", "compare"):
                value = payload.get(field_name)
                if isinstance(value, str):
                    metadata[field_name] = value
            for field_name in ("created", "deleted", "forced"):
                value = payload.get(field_name)
                if isinstance(value, bool):
                    metadata[field_name] = value
            commits = payload.get("commits", [])
            if not isinstance(commits, list):
                raise cls._invalid("Webhook payload field 'commits' must be an array.", "commits")
            metadata["commit_count"] = len(commits)
            metadata["commits"] = [
                {
                    key: commit[key]
                    for key in ("id", "message", "timestamp")
                    if isinstance(commit, dict) and isinstance(commit.get(key), str)
                }
                for commit in commits[:20]
                if isinstance(commit, dict)
            ]
            head_commit = payload.get("head_commit")
            if isinstance(head_commit, dict) and isinstance(head_commit.get("timestamp"), str):
                occurred_at = head_commit["timestamp"]
        elif event_type == "create":
            metadata["ref"] = cls._text(payload, "ref")
            metadata["ref_type"] = cls._text(payload, "ref_type")
            if isinstance(payload.get("master_branch"), str):
                metadata["master_branch"] = payload["master_branch"]
        elif event_type in ("pull_request", "pull_request_review", "pull_request_review_comment"):
            pull = cls._mapping(payload, "pull_request")
            metadata["pull_request"] = cls._compact(
                pull,
                ("number", "title", "state", "merged", "draft", "html_url", "updated_at"),
            )
            base = pull.get("base")
            head = pull.get("head")
            if isinstance(base, dict) and isinstance(base.get("ref"), str):
                metadata["pull_request"]["base_ref"] = base["ref"]
            if isinstance(head, dict) and isinstance(head.get("ref"), str):
                metadata["pull_request"]["head_ref"] = head["ref"]
            if event_type == "pull_request":
                if action is None:
                    raise cls._invalid("Pull request event requires an action.", "action")
                occurred_at = pull.get("updated_at")
            else:
                child_key = "review" if event_type == "pull_request_review" else "comment"
                child = cls._mapping(payload, child_key)
                metadata[child_key] = cls._compact(
                    child,
                    ("id", "state", "commit_id", "path", "line", "created_at", "submitted_at", "updated_at", "html_url"),
                )
                if action is None:
                    raise cls._invalid(f"{event_type} event requires an action.", "action")
                occurred_at = child.get("submitted_at") or child.get("created_at")
        elif event_type in ("issues", "issue_comment"):
            issue = cls._mapping(payload, "issue")
            metadata["issue"] = cls._compact(issue, ("number", "title", "state", "html_url", "updated_at"))
            if event_type == "issues":
                if action is None:
                    raise cls._invalid("Issues event requires an action.", "action")
                occurred_at = issue.get("updated_at")
            else:
                comment = cls._mapping(payload, "comment")
                metadata["comment"] = cls._compact(comment, ("id", "created_at", "updated_at", "html_url"))
                occurred_at = comment.get("created_at")

        if not isinstance(occurred_at, str):
            occurred_at = None
        else:
            occurred_at = occurred_at.strip() or None

        return NormalizedGitHubEvent(
            event_id=event_id,
            source="github",
            event_type=event_type,
            action=action,
            repository=repository,
            actor=cls._actor(payload, event_type),
            occurred_at=occurred_at,
            metadata=metadata,
        )

    @staticmethod
    def _compact(source: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
        return {
            key: source[key]
            for key in fields
            if key in source
            and (isinstance(source[key], (str, int, float, bool)) or source[key] is None)
        }


class InMemoryDeliveryStore:
    """Atomically remember delivery IDs; failed processing releases its ID."""

    def __init__(self) -> None:
        self._delivery_ids: set[str] = set()
        self._lock = Lock()

    def claim(self, delivery_id: str) -> bool:
        with self._lock:
            if delivery_id in self._delivery_ids:
                return False
            self._delivery_ids.add(delivery_id)
            return True

    def release(self, delivery_id: str) -> None:
        with self._lock:
            self._delivery_ids.discard(delivery_id)

    def clear(self) -> None:
        with self._lock:
            self._delivery_ids.clear()


class GitHubWebhookHandler:
    """Verify, normalize, deduplicate, then hand events to the specialist."""

    def __init__(
        self,
        *,
        signature_verifier: GitHubWebhookSignatureVerifier | None = None,
        normalizer: GitHubEventNormalizer | None = None,
        delivery_store: InMemoryDeliveryStore | None = None,
    ) -> None:
        self.signature_verifier = signature_verifier or GitHubWebhookSignatureVerifier()
        self.normalizer = normalizer or GitHubEventNormalizer()
        self.delivery_store = delivery_store or InMemoryDeliveryStore()

    @staticmethod
    def _headers(headers: Mapping[str, str]) -> dict[str, str]:
        return {str(key).lower(): value for key, value in headers.items()}

    def handle(
        self,
        raw_body: bytes,
        headers: Mapping[str, str],
        webhook_secret: str,
        agent: Any,
    ) -> WebhookResult:
        lowered = self._headers(headers)
        # Verify original bytes before decoding/parsing or otherwise transforming them.
        self.signature_verifier.verify(raw_body, lowered.get("x-hub-signature-256"), webhook_secret)

        event_type = lowered.get("x-github-event")
        delivery_id = lowered.get("x-github-delivery")
        if not isinstance(event_type, str) or not event_type.strip():
            raise AgentError(
                "WEBHOOK_EVENT_HEADER_MISSING",
                "GitHub webhook event header is missing.",
                ErrorCategory.VALIDATION,
                False,
            )
        event_type = event_type.strip()
        if event_type not in SUPPORTED_EVENTS and event_type != "ping":
            raise AgentError(
                "WEBHOOK_EVENT_UNSUPPORTED",
                "GitHub webhook event type is not supported.",
                ErrorCategory.VALIDATION,
                False,
                {"event_type": event_type},
            )
        if not isinstance(delivery_id, str) or not delivery_id.strip():
            raise AgentError(
                "WEBHOOK_DELIVERY_ID_MISSING",
                "GitHub webhook delivery ID is missing.",
                ErrorCategory.VALIDATION,
                False,
            )
        delivery_id = delivery_id.strip()

        # GitHub's ping event is only a connectivity handshake. It shares the
        # same delivery claim behavior but never enters domain normalization.
        if event_type == "ping":
            if not self.delivery_store.claim(delivery_id):
                return WebhookResult(status="duplicate", delivery_id=delivery_id)
            return WebhookResult(status="processed", delivery_id=delivery_id)

        try:
            payload = json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
            raise AgentError(
                "WEBHOOK_JSON_INVALID",
                "GitHub webhook body is not valid JSON.",
                ErrorCategory.VALIDATION,
                False,
            ) from None

        event = self.normalizer.normalize(delivery_id, event_type, payload)
        # Delivery IDs make retries idempotent. A claim is held during dispatch;
        # failures release it so GitHub can retry processing.
        if not self.delivery_store.claim(delivery_id):
            return WebhookResult(status="duplicate", delivery_id=delivery_id)

        try:
            context = agent.create_context(event=event)
            result = agent.handle_event(event, context)
        except Exception:
            self.delivery_store.release(delivery_id)
            raise AgentError(
                "WEBHOOK_AGENT_PROCESSING_FAILED",
                "GitHub Agent could not process the webhook event.",
                ErrorCategory.TOOL,
                True,
                {"event_id": delivery_id, "event_type": event_type},
            ) from None

        return WebhookResult(status="processed", delivery_id=delivery_id, result=result)
