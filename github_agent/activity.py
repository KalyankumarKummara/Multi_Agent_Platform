"""Activity records and storage abstractions for GitHub domain history."""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any
import uuid

from .webhook import NormalizedGitHubEvent


@dataclass(frozen=True)
class ActivityRecord:
    """Minimized, normalized history for one processed GitHub delivery."""

    activity_id: str
    event_id: str
    source: str
    agent: str
    event_type: str
    action: str | None
    repository: dict[str, Any]
    actor: dict[str, Any] | None
    target: dict[str, Any]
    occurred_at: str | None
    processed_at: datetime
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_event(
        cls,
        event: NormalizedGitHubEvent,
        *,
        agent_id: str,
    ) -> "ActivityRecord":
        """Create a history record from the existing normalized event model."""
        if not isinstance(event, NormalizedGitHubEvent):
            raise TypeError("Activity records require a NormalizedGitHubEvent.")

        target: dict[str, Any]
        if event.event_type in {"pull_request", "pull_request_review", "pull_request_review_comment"}:
            pull_request = event.metadata.get("pull_request", {})
            target = {
                "type": "pull_request",
                "number": pull_request.get("number"),
            }
        elif event.event_type in {"issues", "issue_comment"}:
            issue = event.metadata.get("issue", {})
            target = {"type": "issue", "number": issue.get("number")}
        elif event.event_type in {"push", "create"}:
            target = {
                "type": event.metadata.get("ref_type", "ref"),
                "ref": event.metadata.get("ref"),
            }
        else:
            target = {
                "type": "repository",
                "name": event.repository.get("full_name"),
            }

        return cls(
            activity_id=f"activity-{uuid.uuid4()}",
            event_id=event.event_id,
            source=event.source,
            agent=agent_id,
            event_type=event.event_type,
            action=event.action,
            repository=deepcopy(event.repository),
            actor=deepcopy(event.actor),
            target=target,
            occurred_at=event.occurred_at,
            processed_at=datetime.now(timezone.utc),
            status="processed",
            metadata=deepcopy(event.metadata),
        )


class ActivityStore(ABC):
    """Storage contract that can later be implemented by a database adapter."""

    @abstractmethod
    def save(self, activity: ActivityRecord) -> ActivityRecord:
        """Persist a new activity and return the stored record."""

    @abstractmethod
    def get(self, activity_id: str) -> ActivityRecord | None:
        """Return an activity by its unique ID, if present."""

    @abstractmethod
    def list_activities(
        self,
        *,
        repository: str | None = None,
        event_type: str | None = None,
        status: str | None = None,
    ) -> list[ActivityRecord]:
        """List activities, optionally filtering by common history fields."""


class InMemoryActivityStore(ActivityStore):
    """Thread-safe development store that keeps defensive record copies."""

    def __init__(self) -> None:
        self._activities: dict[str, ActivityRecord] = {}
        self._lock = Lock()

    def save(self, activity: ActivityRecord) -> ActivityRecord:
        if not isinstance(activity, ActivityRecord):
            raise TypeError("Only ActivityRecord values can be saved.")
        with self._lock:
            if activity.activity_id in self._activities:
                raise ValueError("Activity ID already exists.")
            stored = deepcopy(activity)
            self._activities[activity.activity_id] = stored
            return deepcopy(stored)

    def get(self, activity_id: str) -> ActivityRecord | None:
        with self._lock:
            activity = self._activities.get(activity_id)
            return deepcopy(activity) if activity is not None else None

    def list_activities(
        self,
        *,
        repository: str | None = None,
        event_type: str | None = None,
        status: str | None = None,
    ) -> list[ActivityRecord]:
        with self._lock:
            activities = list(self._activities.values())
        if repository is not None:
            activities = [
                item for item in activities
                if item.repository.get("full_name") == repository
            ]
        if event_type is not None:
            activities = [item for item in activities if item.event_type == event_type]
        if status is not None:
            activities = [item for item in activities if item.status == status]
        return deepcopy(activities)
