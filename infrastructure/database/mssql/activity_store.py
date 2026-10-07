"""MSSQL implementation of the GitHub domain ActivityStore contract."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from github_agent.activity import ActivityRecord, ActivityStore

from .models import GitHubActivityModel


class DuplicateActivityError(ValueError):
    """An activity or webhook event ID already exists in persistent storage."""


class ActivityPersistenceError(RuntimeError):
    """A database operation failed; details are intentionally not exposed."""


def _serialize(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _deserialize(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError("Stored activity JSON must contain an object.")
    return decoded


def _parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        raise ValueError("Activity occurred_at must be a valid ISO-8601 timestamp.") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class MSSQLActivityStore(ActivityStore):
    """Persist activities with SQLAlchemy and return domain records only."""

    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    @staticmethod
    def _to_model(activity: ActivityRecord) -> GitHubActivityModel:
        repository = deepcopy(activity.repository)
        return GitHubActivityModel(
            activity_id=activity.activity_id,
            event_id=activity.event_id,
            source=activity.source,
            agent=activity.agent,
            event_type=activity.event_type,
            action=activity.action,
            repository=_serialize(repository),
            repository_full_name=str(repository.get("full_name", "")),
            actor=_serialize(activity.actor) if activity.actor is not None else None,
            target=_serialize(deepcopy(activity.target)),
            occurred_at=_parse_timestamp(activity.occurred_at),
            processed_at=_as_utc(activity.processed_at),
            status=activity.status,
            activity_metadata=_serialize(deepcopy(activity.metadata)),
        )

    @staticmethod
    def _to_record(model: GitHubActivityModel) -> ActivityRecord:
        occurred_at = model.occurred_at
        processed_at = _as_utc(model.processed_at)
        return ActivityRecord(
            activity_id=model.activity_id,
            event_id=model.event_id,
            source=model.source,
            agent=model.agent,
            event_type=model.event_type,
            action=model.action,
            repository=_deserialize(model.repository) or {},
            actor=_deserialize(model.actor),
            target=_deserialize(model.target) or {},
            occurred_at=_as_utc(occurred_at).isoformat() if occurred_at is not None else None,
            processed_at=processed_at,
            status=model.status,
            metadata=_deserialize(model.activity_metadata) or {},
        )

    @staticmethod
    def _raise_database_error() -> None:
        raise ActivityPersistenceError("GitHub activity database operation failed.") from None

    def save(self, activity: ActivityRecord) -> ActivityRecord:
        if not isinstance(activity, ActivityRecord):
            raise TypeError("Only ActivityRecord values can be saved.")

        session = self._sessions()
        try:
            duplicate = session.scalar(
                select(GitHubActivityModel.activity_id).where(
                    or_(
                        GitHubActivityModel.activity_id == activity.activity_id,
                        GitHubActivityModel.event_id == activity.event_id,
                    )
                )
            )
            if duplicate is not None:
                raise DuplicateActivityError("Activity or event ID already exists.")

            session.add(self._to_model(activity))
            session.commit()
            return deepcopy(activity)
        except DuplicateActivityError:
            session.rollback()
            raise
        except IntegrityError:
            session.rollback()
            # A concurrent insert may win after the pre-check; inspect only
            # the two unique keys to distinguish that case from DB failure.
            try:
                duplicate = session.scalar(
                    select(GitHubActivityModel.activity_id).where(
                        or_(
                            GitHubActivityModel.activity_id == activity.activity_id,
                            GitHubActivityModel.event_id == activity.event_id,
                        )
                    )
                )
            except SQLAlchemyError:
                self._raise_database_error()
            if duplicate is not None:
                raise DuplicateActivityError("Activity or event ID already exists.") from None
            self._raise_database_error()
        except SQLAlchemyError:
            session.rollback()
            self._raise_database_error()
        finally:
            session.close()

    def get(self, activity_id: str) -> ActivityRecord | None:
        session = self._sessions()
        try:
            model = session.get(GitHubActivityModel, activity_id)
            return self._to_record(model) if model is not None else None
        except SQLAlchemyError:
            session.rollback()
            self._raise_database_error()
        finally:
            session.close()

    def list_activities(
        self,
        *,
        repository: str | None = None,
        event_type: str | None = None,
        status: str | None = None,
    ) -> list[ActivityRecord]:
        session = self._sessions()
        try:
            statement = select(GitHubActivityModel)
            if repository is not None:
                statement = statement.where(GitHubActivityModel.repository_full_name == repository)
            if event_type is not None:
                statement = statement.where(GitHubActivityModel.event_type == event_type)
            if status is not None:
                statement = statement.where(GitHubActivityModel.status == status)
            statement = statement.order_by(
                GitHubActivityModel.processed_at.desc(),
                GitHubActivityModel.activity_id.asc(),
            )
            models = session.scalars(statement).all()
            return [self._to_record(model) for model in models]
        except SQLAlchemyError:
            session.rollback()
            self._raise_database_error()
        finally:
            session.close()
