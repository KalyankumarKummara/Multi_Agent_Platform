"""Activity records and storage abstractions for GitHub domain history."""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any
import uuid

from .webhook import NormalizedGitHubEvent


def as_utc_datetime(value: datetime | str | None) -> datetime | None:
    """Parse an activity timestamp and normalize it to UTC."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("Activity timestamp must be valid ISO-8601.") from None
    if not isinstance(value, datetime):
        raise ValueError("Activity timestamp must be a datetime or ISO-8601 string.")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


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
    significance: str = "low"

    @classmethod
    def from_event(
        cls,
        event: NormalizedGitHubEvent,
        *,
        agent_id: str,
        significance: str = "low",
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
            significance=significance,
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
        significance: str | None = None,
        occurred_at_from: datetime | None = None,
        occurred_at_to: datetime | None = None,
        limit: int = 50,
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
        significance: str | None = None,
        occurred_at_from: datetime | None = None,
        occurred_at_to: datetime | None = None,
        limit: int = 50,
    ) -> list[ActivityRecord]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100.")
        start = as_utc_datetime(occurred_at_from)
        end = as_utc_datetime(occurred_at_to)
        if start is not None and end is not None and start > end:
            raise ValueError("occurred_at_from must not be after occurred_at_to.")
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
        if significance is not None:
            activities = [item for item in activities if item.significance == significance]
        if start is not None:
            activities = [
                item for item in activities
                if (occurred := as_utc_datetime(item.occurred_at)) is not None and occurred >= start
            ]
        if end is not None:
            activities = [
                item for item in activities
                if (occurred := as_utc_datetime(item.occurred_at)) is not None and occurred <= end
            ]
        activities.sort(
            key=lambda item: (
                -as_utc_datetime(item.processed_at).timestamp(),
                item.activity_id,
            )
        )
        return deepcopy(activities[:limit])
