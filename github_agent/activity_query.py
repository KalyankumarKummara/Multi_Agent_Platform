"""Validated, read-only queries over GitHub activity history."""

import re
from datetime import datetime

from .activity import ActivityRecord, ActivityStore, as_utc_datetime
from .webhook import SUPPORTED_EVENTS


class ActivityQueryError(ValueError):
    """A caller supplied an invalid GitHub activity query filter."""


class GitHubActivityQuery:
    """Small domain query service that only depends on ActivityStore."""

    MAX_LIMIT = 100
    _repository_pattern = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    _status_pattern = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")

    def __init__(self, store: ActivityStore) -> None:
        self._store = store

    @classmethod
    def _validate_limit(cls, limit: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= cls.MAX_LIMIT:
            raise ActivityQueryError("limit must be an integer between 1 and 100.")
        return limit

    @classmethod
    def _validate_repository(cls, repository: str | None) -> str | None:
        if repository is None:
            return None
        if (
            not isinstance(repository, str)
            or not cls._repository_pattern.fullmatch(repository)
            or any(part in {".", ".."} for part in repository.split("/"))
        ):
            raise ActivityQueryError("repository must use the owner/repository format.")
        return repository

    @staticmethod
    def _validate_event_type(event_type: str | None) -> str | None:
        if event_type is None:
            return None
        if not isinstance(event_type, str) or event_type not in SUPPORTED_EVENTS:
            raise ActivityQueryError("event_type is not supported.")
        return event_type

    @classmethod
    def _validate_status(cls, status: str | None) -> str | None:
        if status is None:
            return None
        if not isinstance(status, str) or not cls._status_pattern.fullmatch(status):
            raise ActivityQueryError("status is invalid.")
        return status

    @staticmethod
    def _validate_significance(significance: str | None) -> str | None:
        if significance is None:
            return None
        if not isinstance(significance, str) or significance not in {"low", "medium", "high"}:
            raise ActivityQueryError("significance must be low, medium, or high.")
        return significance

    @staticmethod
    def _validate_time(value: datetime | str | None) -> datetime | None:
        try:
            return as_utc_datetime(value)
        except (TypeError, ValueError):
            raise ActivityQueryError("Activity time filters must be valid datetimes.") from None

    def list_recent_activity(
        self,
        *,
        limit: int = 50,
        repository: str | None = None,
        event_type: str | None = None,
        status: str | None = None,
        significance: str | None = None,
        occurred_at_from: datetime | str | None = None,
        occurred_at_to: datetime | str | None = None,
    ) -> list[ActivityRecord]:
        """Return newest matching activities with deterministic tie-breaking."""
        selected_limit = self._validate_limit(limit)
        start = self._validate_time(occurred_at_from)
        end = self._validate_time(occurred_at_to)
        if start is not None and end is not None and start > end:
            raise ActivityQueryError("occurred_at_from must not be after occurred_at_to.")
        return self._store.list_activities(
            repository=self._validate_repository(repository),
            event_type=self._validate_event_type(event_type),
            status=self._validate_status(status),
            significance=self._validate_significance(significance),
            occurred_at_from=start,
            occurred_at_to=end,
            limit=selected_limit,
        )

    def list_repository_activity(self, repository: str, *, limit: int = 50) -> list[ActivityRecord]:
        return self.list_recent_activity(repository=repository, limit=limit)

    def list_high_significance_activity(self, *, limit: int = 50) -> list[ActivityRecord]:
        return self.list_recent_activity(significance="high", limit=limit)

    def list_activity_by_event_type(
        self,
        event_type: str,
        *,
        limit: int = 50,
    ) -> list[ActivityRecord]:
        return self.list_recent_activity(event_type=event_type, limit=limit)
