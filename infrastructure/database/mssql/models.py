"""SQL Server ORM schema for minimized GitHub activity history."""

from datetime import datetime

from sqlalchemy import DateTime, Index, String, UnicodeText, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class ActivityBase(DeclarativeBase):
    """Base metadata for the MSSQL activity table only."""


class GitHubActivityModel(ActivityBase):
    """Activity columns store normalized values, never the raw webhook body."""

    __tablename__ = "github_activities"
    __table_args__ = (
        UniqueConstraint("event_id", name="uq_github_activities_event_id"),
        Index("ix_github_activities_repository", "repository_full_name"),
        Index("ix_github_activities_event_type", "event_type"),
        Index("ix_github_activities_occurred_at", "occurred_at"),
        Index("ix_github_activities_status", "status"),
    )

    activity_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    event_id: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    agent: Mapped[str] = mapped_column(String(128), nullable=False)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    action: Mapped[str | None] = mapped_column(String(100), nullable=True)
    repository: Mapped[str] = mapped_column("repository", UnicodeText, nullable=False)
    repository_full_name: Mapped[str] = mapped_column(String(512), nullable=False)
    actor: Mapped[str | None] = mapped_column("actor", UnicodeText, nullable=True)
    target: Mapped[str] = mapped_column("target", UnicodeText, nullable=False)
    occurred_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    processed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    activity_metadata: Mapped[str] = mapped_column("metadata", UnicodeText, nullable=False)
