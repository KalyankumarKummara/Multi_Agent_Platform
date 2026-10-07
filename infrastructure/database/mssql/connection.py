"""SQLAlchemy engine and session setup for the dedicated activity database."""

from dataclasses import dataclass, field
import os
import re

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session, sessionmaker

from .models import ActivityBase, GitHubActivityModel


def _environment_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be set to yes or no.")


def _reject_manager_ai(database_name: str | None) -> None:
    if database_name and database_name.strip().casefold() == "managerai":
        raise ValueError("The GitHub activity store cannot target the ManagerAI database.")


def _database_from_url(url: URL) -> str | None:
    if url.database:
        return url.database
    odbc_connection = url.query.get("odbc_connect")
    if isinstance(odbc_connection, str):
        match = re.search(r"(?:^|;)\s*DATABASE\s*=\s*\{?([^;}]+)", odbc_connection, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


@dataclass(frozen=True)
class DatabaseSettings:
    """Dedicated database settings; URL is excluded from repr to protect secrets."""

    server: str = "localhost"
    database: str = "MultiAgentPlatform"
    driver: str = "ODBC Driver 18 for SQL Server"
    trust_server_certificate: bool = True
    database_url: str | None = field(default=None, repr=False)

    @classmethod
    def from_environment(cls) -> "DatabaseSettings":
        """Load activity database configuration without logging its URL."""
        database_url = os.environ.get("GITHUB_ACTIVITY_DATABASE_URL") or None
        server = os.environ.get("GITHUB_ACTIVITY_DB_SERVER", "localhost")
        database = os.environ.get("GITHUB_ACTIVITY_DB_NAME", "MultiAgentPlatform")
        driver = os.environ.get("GITHUB_ACTIVITY_DB_DRIVER", "ODBC Driver 18 for SQL Server")
        trust_certificate = _environment_bool(
            "GITHUB_ACTIVITY_DB_TRUST_SERVER_CERTIFICATE",
            default=True,
        )

        if database_url:
            parsed_url = make_url(database_url)
            if parsed_url.drivername != "mssql+pyodbc":
                raise ValueError("GITHUB_ACTIVITY_DATABASE_URL must use mssql+pyodbc.")
            _reject_manager_ai(parsed_url.database)
        else:
            _reject_manager_ai(database)

        return cls(
            server=server,
            database=database,
            driver=driver,
            trust_server_certificate=trust_certificate,
            database_url=database_url,
        )

    def to_sqlalchemy_url(self) -> URL:
        """Build an ODBC URL using Windows Authentication and no credentials."""
        if self.database_url:
            url = make_url(self.database_url)
            _reject_manager_ai(_database_from_url(url))
            return url
        _reject_manager_ai(self.database)

        def odbc_value(value: str) -> str:
            return "{" + value.replace("}", "}}") + "}"

        trust_value = "yes" if self.trust_server_certificate else "no"
        connection_string = ";".join(
            (
                f"DRIVER={odbc_value(self.driver)}",
                f"SERVER={odbc_value(self.server)}",
                f"DATABASE={odbc_value(self.database)}",
                "Trusted_Connection=yes",
                "Encrypt=yes",
                f"TrustServerCertificate={trust_value}",
            )
        )
        return URL.create(
            "mssql+pyodbc",
            query={"odbc_connect": connection_string},
        )


def create_database_engine(settings: DatabaseSettings | None = None) -> Engine:
    """Create a quiet pooled engine for only the configured activity database."""
    configured = settings or DatabaseSettings.from_environment()
    _reject_manager_ai(configured.database)
    url = configured.to_sqlalchemy_url()
    _reject_manager_ai(_database_from_url(url))
    return create_engine(url, pool_pre_ping=True, echo=False)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create the reusable SQLAlchemy session factory for a database engine."""
    return sessionmaker(bind=engine, expire_on_commit=False)


def initialize_activity_schema(engine: Engine) -> None:
    """Create the activity table if absent; never drop or recreate objects."""
    _reject_manager_ai(_database_from_url(engine.url))
    ActivityBase.metadata.create_all(
        bind=engine,
        tables=[GitHubActivityModel.__table__],
    )
